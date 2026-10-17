"""Benchmark 2: data versioning and time-travel for multimodal data.

A ground-truth dataset evolves through a scripted sequence of commits that mimic a
real ML data lifecycle (appends, relabels, GDPR deletes, caption fixes, re-embedding
with a new model, image edits, schema evolution). Each strategy applies the same
commits; we record commit latency, cumulative storage, and time-travel read cost, and
verify that every historical version reads back exactly.

Strategies:
  FullCopy     - one Parquet snapshot per version
  FileCAS      - per-sample objects (image / embedding / metadata) in a content-addressed
                 store + per-version manifest (git-LFS / DVC style)
  ChunkCAS     - canonical Parquet snapshot, content-defined chunking (gear hash) and a
                 chunk store (restic / Xet style); with and without Parquet CDC writer
  Delta        - Delta Lake table (copy-on-write MERGE/DELETE), time travel by version
  Lance        - Lance dataset (deletion files, column-level data evolution)
  Delta+Blob   - Delta Lake for metadata/embeddings, images in an immutable blob CAS
"""
import hashlib
import os
import io
import json
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from deltalake import DeltaTable, write_deltalake

import lance
from common import DATA, EMB_DIM, dir_size, file_count, fresh, make_embedding, make_image, make_rows, save_json, timed

OUT = DATA / "versions"
N0 = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
SEED = 7


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Workload: scripted commits over a ground-truth table
# ---------------------------------------------------------------------------

def canonical(t):
    return t.sort_by("id").combine_chunks()


def commit_script(n0):
    """Yield (name, kind, payload, new_state) for each commit; state is the full table."""
    rng = np.random.default_rng(SEED)
    state = canonical(make_rows(n0, 0, seed=1000))
    next_id = n0
    yield "v0 initial load", "init", state, state

    def pick(frac):
        ids = state.column("id").to_numpy()
        return np.sort(rng.choice(ids, size=max(1, int(frac * len(ids))), replace=False))

    def upd(ids, cols, fn):
        nonlocal state
        mask = pc.is_in(state.column("id"), pa.array(ids))
        rows = state.filter(mask)
        new = fn(rows)
        for c in cols:
            rows = rows.set_column(rows.schema.get_field_index(c), c, new[c])
        state = canonical(pa.concat_tables([state.filter(pc.invert(mask)), rows]))
        return rows

    script = ["append", "relabel", "delete", "caption", "reembed", "append", "image_edit",
              "relabel", "delete", "append", "reembed", "add_column"]
    for op in script:
        if op == "append":
            new = make_rows(1000, next_id, seed=int(rng.integers(1 << 30)))
            next_id += 1000
            state = canonical(pa.concat_tables([state, new]))
            yield "append 1k", "append", new, state
        elif op == "relabel":
            ids = pick(0.02)
            rows = upd(ids, ["label"], lambda r: {"label": pa.array(rng.integers(0, 100, r.num_rows).astype(np.int32))})
            yield "relabel 2%", "update", (rows, ["label"]), state
        elif op == "caption":
            ids = pick(0.05)
            rows = upd(ids, ["caption"], lambda r: {"caption": pc.utf8_upper(r.column("caption"))})
            yield "fix captions 5%", "update", (rows, ["caption"]), state
        elif op == "delete":
            ids = pick(0.01)
            state = canonical(state.filter(pc.invert(pc.is_in(state.column("id"), pa.array(ids)))))
            yield "delete 1%", "delete", ids, state
        elif op == "reembed":
            ids = state.column("id").to_numpy()
            e = make_embedding(rng, len(ids))
            rows = upd(ids, ["embedding"], lambda r: {"embedding": pa.FixedSizeListArray.from_arrays(pa.array(e.reshape(-1)), EMB_DIM)})
            yield "re-embed 100%", "update", (rows, ["embedding"]), state
        elif op == "image_edit":
            ids = pick(0.01)
            rows = upd(ids, ["image"], lambda r: {"image": pa.array([make_image(rng, int(l)) for l in r.column("label").to_pylist()], pa.binary())})
            yield "edit images 1%", "update", (rows, ["image"]), state
        elif op == "add_column":
            col = pa.array(rng.random(state.num_rows, dtype=np.float32))
            state = state.append_column("quality", col)
            yield "add column", "add_column", ("quality", state.select(["id", "quality"])), state


def fingerprint(t):
    """Order-independent digest of a version (used to verify time travel)."""
    t = canonical(t.select(sorted(t.column_names)))
    h = hashlib.sha256()
    for c in t.column_names:
        col = t.column(c)
        if pa.types.is_binary(col.type) or pa.types.is_large_binary(col.type):
            col = pa.array([sha(b) for b in col.to_pylist()])
        elif pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type):
            col = pa.array([np.asarray(v, np.float32).tobytes().hex()[:32] for v in col.to_pylist()])
        h.update(c.encode())
        h.update(str(col.to_pylist()).encode())
    return h.hexdigest()


def normalize(t):
    """Normalize types produced by different engines (binary vs large_binary, list vs fixed list)."""
    cols = {}
    for name in t.column_names:
        col = t.column(name).combine_chunks()
        if pa.types.is_large_binary(col.type):
            col = col.cast(pa.binary())
        if pa.types.is_large_string(col.type):
            col = col.cast(pa.string())
        if name == "embedding" and not pa.types.is_fixed_size_list(col.type):
            col = pa.FixedSizeListArray.from_arrays(pc.list_flatten(col).cast(pa.float32()), EMB_DIM)
        cols[name] = col
    return pa.table(cols)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class FullCopy:
    name = "Full copy"

    def __init__(self, p):
        self.p, self.v = p, 0

    def commit(self, kind, payload, state):
        pq.write_table(state, self.p / f"v{self.v:03d}.parquet", compression="snappy")
        self.v += 1

    def read(self, v):
        return pq.read_table(self.p / f"v{v:03d}.parquet")

    def lookup(self, v, ids):
        return pq.read_table(self.p / f"v{v:03d}.parquet", filters=[("id", "in", list(map(int, ids)))])


class FileCAS:
    """Per-sample objects keyed by content hash; a manifest per version."""
    name = "File-level CAS"

    def __init__(self, p):
        self.p, self.v = p, 0
        (p / "objects").mkdir(parents=True)
        self.man = None  # current manifest: id -> (img, emb, meta)

    def _put(self, b):
        h = sha(b)
        d = self.p / "objects" / h[:2]
        d.mkdir(exist_ok=True)
        f = d / h
        if not f.exists():
            f.write_bytes(b)
        return h

    def _get(self, h):
        return (self.p / "objects" / h[:2] / h).read_bytes()

    def _encode_rows(self, t, cols=None):
        meta_cols = [c for c in t.column_names if c not in ("image", "embedding")]
        ids = t.column("id").to_pylist()
        out = {}
        imgs = t.column("image").to_pylist() if (cols is None or "image" in cols) else None
        emb = t.column("embedding").combine_chunks().flatten().to_numpy().reshape(-1, EMB_DIM) \
            if (cols is None or "embedding" in cols) else None
        metas = t.select(meta_cols).to_pylist() if (cols is None or set(cols) & set(meta_cols)) else None
        for i, k in enumerate(ids):
            old = self.man.get(k, (None, None, None)) if self.man else (None, None, None)
            out[k] = (self._put(imgs[i]) if imgs is not None else old[0],
                      self._put(emb[i].tobytes()) if emb is not None else old[1],
                      self._put(json.dumps(metas[i], sort_keys=True).encode()) if metas is not None else old[2])
        return out

    def commit(self, kind, payload, state):
        if kind == "init":
            self.man = self._encode_rows(payload)
        elif kind == "append":
            self.man.update(self._encode_rows(payload))
        elif kind == "update":
            rows, cols = payload
            self.man.update(self._encode_rows(rows, cols))
        elif kind == "delete":
            for k in payload.tolist():
                self.man.pop(k, None)
        elif kind == "add_column":  # metadata objects change for every row
            self.man.update(self._encode_rows(state, ["quality"]))
        ids = sorted(self.man)
        m = pa.table({"id": ids, "img": [self.man[k][0] for k in ids],
                      "emb": [self.man[k][1] for k in ids], "meta": [self.man[k][2] for k in ids]})
        pq.write_table(m, self.p / f"manifest-v{self.v:03d}.parquet")
        self.v += 1

    def _materialize(self, m):
        metas = [json.loads(self._get(h)) for h in m.column("meta").to_pylist()]
        t = pa.Table.from_pylist(metas)
        emb = np.stack([np.frombuffer(self._get(h), np.float32) for h in m.column("emb").to_pylist()])
        t = t.append_column("embedding", pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1)), EMB_DIM))
        return t.append_column("image", pa.array([self._get(h) for h in m.column("img").to_pylist()], pa.binary()))

    def read(self, v):
        return self._materialize(pq.read_table(self.p / f"manifest-v{v:03d}.parquet"))

    def lookup(self, v, ids):
        m = pq.read_table(self.p / f"manifest-v{v:03d}.parquet", filters=[("id", "in", list(map(int, ids)))])
        return self._materialize(m)


GEAR = np.random.default_rng(123).integers(0, 1 << 32, 256, dtype=np.uint64).astype(np.uint32)


def cdc_boundaries(buf, avg_bits=16, min_size=16 << 10, max_size=256 << 10, window=32):
    """Content-defined chunk boundaries using a vectorized gear hash over a 32-byte window."""
    b = np.frombuffer(buf, np.uint8)
    g = GEAR[b]
    h = np.zeros(len(b), np.uint32)
    for j in range(window):
        h[j:] += g[:len(b) - j] << np.uint32(j)
    mask = np.uint32((1 << avg_bits) - 1) << np.uint32(32 - avg_bits)  # use high bits
    cand = np.nonzero((h & mask) == 0)[0] + 1
    cuts, last = [], 0
    ci = 0
    while last < len(b):
        lo, hi = last + min_size, last + max_size
        ci = np.searchsorted(cand, lo, side="left")
        nxt = cand[ci] if ci < len(cand) and cand[ci] <= hi else min(hi, len(b))
        nxt = min(nxt, len(b))
        cuts.append(nxt)
        last = nxt
    return cuts


class ChunkCAS:
    """Canonical Parquet snapshot per version, deduplicated at chunk granularity."""

    def __init__(self, p, parquet_cdc=False, fixed=False):
        self.p, self.v, self.cdc, self.fixed = p, 0, parquet_cdc, fixed
        (p / "chunks").mkdir(parents=True)
        self.name = ("Chunk CAS (fixed 64KB)" if fixed else
                     "Chunk CAS (CDC, Parquet CDC)" if parquet_cdc else "Chunk CAS (CDC)")

    def commit(self, kind, payload, state):
        buf = io.BytesIO()
        pq.write_table(state, buf, compression="snappy", row_group_size=1024,
                       use_content_defined_chunking=self.cdc)
        data = buf.getvalue()
        cuts = list(range(64 << 10, len(data), 64 << 10)) + [len(data)] if self.fixed else cdc_boundaries(data)
        hashes, last = [], 0
        for c in cuts:
            chunk = data[last:c]
            h = sha(chunk)
            f = self.p / "chunks" / h
            if not f.exists():
                f.write_bytes(chunk)
            hashes.append(h)
            last = c
        (self.p / f"v{self.v:03d}.json").write_text(json.dumps(hashes))
        self.v += 1

    def _bytes(self, v):
        hs = json.loads((self.p / f"v{v:03d}.json").read_text())
        return b"".join((self.p / "chunks" / h).read_bytes() for h in hs)

    def read(self, v):
        return pq.read_table(pa.BufferReader(self._bytes(v)))

    def lookup(self, v, ids):
        return pq.read_table(pa.BufferReader(self._bytes(v)), filters=[("id", "in", list(map(int, ids)))])


class Delta:
    name = "Delta Lake"

    def __init__(self, p):
        self.p, self.uri = p, str(p)
        self.vmap = []

    def commit(self, kind, payload, state):
        if kind == "init":
            write_deltalake(self.uri, payload)
        elif kind == "append":
            write_deltalake(self.uri, payload, mode="append")
        elif kind == "update":
            rows, cols = payload
            dt = DeltaTable(self.uri)
            dt.merge(source=rows.select(["id"] + cols), predicate="t.id = s.id",
                     source_alias="s", target_alias="t") \
              .when_matched_update(updates={c: f"s.{c}" for c in cols}).execute()
        elif kind == "delete":
            DeltaTable(self.uri).delete(f"id in ({','.join(map(str, payload.tolist()))})")
        elif kind == "add_column":
            write_deltalake(self.uri, state, mode="overwrite", schema_mode="overwrite")
        self.vmap.append(DeltaTable(self.uri).version())

    def read(self, v):
        return DeltaTable(self.uri, version=self.vmap[v]).to_pyarrow_table()

    def lookup(self, v, ids):
        return DeltaTable(self.uri, version=self.vmap[v]).to_pyarrow_dataset() \
            .to_table(filter=pc.field("id").isin(list(map(int, ids))))


class LanceStrategy:
    name = "Lance"

    def __init__(self, p):
        self.p, self.uri, self.vmap = p, str(p), []

    def commit(self, kind, payload, state):
        if kind == "init":
            ds = lance.write_dataset(payload, self.uri, max_rows_per_group=1024)
        elif kind == "append":
            ds = lance.write_dataset(payload, self.uri, mode="append")
        elif kind == "update":
            rows, cols = payload
            ds = lance.dataset(self.uri)
            if len(cols) == 1 and rows.num_rows == ds.count_rows():
                # whole-column replacement: column-level data evolution, images untouched
                ds.drop_columns(cols)
                ds = lance.dataset(self.uri)
                ds.merge(rows.select(["id"] + cols), left_on="id")
            else:
                ds.merge_insert("id").when_matched_update_all().execute(
                    _full_rows(ds, rows, cols))
        elif kind == "delete":
            lance.dataset(self.uri).delete(f"id IN ({','.join(map(str, payload.tolist()))})")
        elif kind == "add_column":
            name, tbl = payload
            lance.dataset(self.uri).merge(tbl, left_on="id")
        self.vmap.append(lance.dataset(self.uri).version)

    def read(self, v):
        return lance.dataset(self.uri, version=self.vmap[v]).to_table()

    def lookup(self, v, ids):
        return lance.dataset(self.uri, version=self.vmap[v]).to_table(
            filter=f"id IN ({','.join(map(str, ids))})")


def _full_rows(ds, rows, cols):
    """merge_insert needs complete rows: current values with the changed columns replaced."""
    cur = ds.to_table(filter=f"id IN ({','.join(map(str, rows.column('id').to_pylist()))})")
    cur = canonical(cur)
    rows = canonical(rows)
    for c in cols:
        cur = cur.set_column(cur.schema.get_field_index(c), c, rows.column(c).cast(cur.schema.field(c).type))
    return cur


class DeltaBlob:
    """Delta Lake for small/mutable columns; image bytes in an immutable content-addressed
    blob store referenced by hash. Updates never rewrite image bytes."""
    name = "Delta + blob CAS"

    def __init__(self, p):
        self.p, self.uri = p, str(p / "table")
        (p / "blobs").mkdir(parents=True)
        self.vmap = []

    def _strip(self, t):
        hs = []
        for b in t.column("image").to_pylist():
            h = sha(b)
            f = self.p / "blobs" / h
            if not f.exists():
                f.write_bytes(b)
            hs.append(h)
        return t.drop_columns(["image"]).append_column("image_ref", pa.array(hs))

    def commit(self, kind, payload, state):
        if kind == "init":
            write_deltalake(self.uri, self._strip(payload))
        elif kind == "append":
            write_deltalake(self.uri, self._strip(payload), mode="append")
        elif kind == "update":
            rows, cols = payload
            if "image" in cols:
                rows, cols = self._strip(rows), [c for c in cols if c != "image"] + ["image_ref"]
            DeltaTable(self.uri).merge(source=rows.select(["id"] + cols), predicate="t.id = s.id",
                                       source_alias="s", target_alias="t") \
                .when_matched_update(updates={c: f"s.{c}" for c in cols}).execute()
        elif kind == "delete":
            DeltaTable(self.uri).delete(f"id in ({','.join(map(str, payload.tolist()))})")
        elif kind == "add_column":
            write_deltalake(self.uri, self._strip(state), mode="overwrite", schema_mode="overwrite")
        self.vmap.append(DeltaTable(self.uri).version())

    def _hydrate(self, t):
        imgs = [(self.p / "blobs" / h).read_bytes() for h in t.column("image_ref").to_pylist()]
        return t.drop_columns(["image_ref"]).append_column("image", pa.array(imgs, pa.binary()))

    def read(self, v):
        return self._hydrate(DeltaTable(self.uri, version=self.vmap[v]).to_pyarrow_table())

    def lookup(self, v, ids):
        t = DeltaTable(self.uri, version=self.vmap[v]).to_pyarrow_dataset() \
            .to_table(filter=pc.field("id").isin(list(map(int, ids))))
        return self._hydrate(t)


STRATEGIES = [FullCopy, FileCAS, lambda p: ChunkCAS(p, fixed=True), lambda p: ChunkCAS(p),
              lambda p: ChunkCAS(p, parquet_cdc=True), Delta, LanceStrategy, DeltaBlob]


def main(only=None):
    commits = list(commit_script(N0))
    truth = [fingerprint(s) for _, _, _, s in commits]
    logical = []
    for _, _, _, s in commits:
        buf = io.BytesIO()
        pq.write_table(s, buf, compression="snappy")
        logical.append(len(buf.getvalue()))
    print("logical version sizes (MB):", [round(x / 1e6, 1) for x in logical], flush=True)
    out = {"n0": N0, "commits": [c[0] for c in commits], "logical_bytes": logical, "strategies": []}
    probe = np.random.default_rng(5).choice(commits[0][3].column("id").to_numpy(), 100, replace=False)
    for S in STRATEGIES:
        name = S(fresh(OUT / "_probe")).name
        if only and not any(o.strip() == name for o in only.split(",")):
            continue
        p = fresh(OUT / name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("+", "plus"))
        p.mkdir(parents=True)
        s = S(p)
        rec = {"name": name, "commit_s": [], "bytes": [], "files": []}
        for cname, kind, payload, state in commits:
            t0 = time.perf_counter()
            s.commit(kind, payload, state)
            rec["commit_s"].append(time.perf_counter() - t0)
            rec["bytes"].append(dir_size(p))
            rec["files"].append(file_count(p))
        # verify every version and time the full time-travel read
        rec["read_s"], ok = [], []
        for v in range(len(commits)):
            t0 = time.perf_counter()
            t = s.read(v)
            rec["read_s"].append(time.perf_counter() - t0)
            ok.append(fingerprint(normalize(t)) == truth[v])
        rec["verified"] = ok
        # point lookups (100 ids that exist in v0) against the oldest and the newest version
        rec["lookup_v0_s"], _ = timed(lambda: s.lookup(0, probe))
        live = [k for k in probe if k in set(commits[-1][3].column("id").to_pylist())]
        rec["lookup_latest_s"], _ = timed(lambda: s.lookup(len(commits) - 1, live))
        out["strategies"].append(rec)
        print(f"{name:30s} total={rec['bytes'][-1] / 1e6:8.1f}MB files={rec['files'][-1]:6d} "
              f"commit_total={sum(rec['commit_s']):6.1f}s read_v0={rec['read_s'][0]:.2f}s "
              f"read_last={rec['read_s'][-1]:.2f}s lookup_v0={rec['lookup_v0_s'] * 1e3:.0f}ms "
              f"verified={sum(ok)}/{len(ok)}", flush=True)
        print("   commit_s:", [round(x, 2) for x in rec["commit_s"]], flush=True)
    save_json(os.environ.get("MMSTORE_OUT", "version.json"), out)


if __name__ == "__main__":
    main(sys.argv[2] if len(sys.argv) > 2 else None)
