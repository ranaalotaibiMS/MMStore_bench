"""Benchmark 1: storage layouts for multimodal AI workloads.

Layouts:   file-per-sample, JSONL, WebDataset-style tar shards, HDF5, Arrow IPC,
           Parquet (row-group size x codec), Lance, and a decoupled
           "Parquet + blob file" layout.
Workloads: W1 full scan, W2 random-access mini-batches, W3 embedding projection,
           W4 metadata scan, W5 selective filter (unsorted vs. clustered), ingest.
"""
import base64
import io
import json
import struct
import sys
import tarfile
import time

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from common import DATA, EMB_DIM, dir_size, file_count, fresh, save_json, timed

try:
    import lance
except ImportError:  # pragma: no cover
    lance = None

OUT = DATA / "layouts"
META = ["id", "label", "split", "score", "width", "height", "caption"]
RNG = np.random.default_rng(42)
BATCH = 64
N_RANDOM = 2048
FILTER_LABELS = 5  # label < 5  -> ~5% selectivity


def touch(col):
    """Force every image byte to be read (mmap'd Arrow data is otherwise lazy)."""
    tot = 0
    for ch in col.chunks:
        buf = ch.buffers()[2]
        if buf is not None:
            a = np.frombuffer(buf, np.uint8)
            tot += int(a[::64].sum()) + len(a)  # one byte per cache line pulls every page
    return tot


def emb_np(col):
    return col.combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, EMB_DIM)


# ---------------------------------------------------------------------------
# Layout implementations
# ---------------------------------------------------------------------------

class Files:
    name = "Files (per sample)"

    def write(self, t, p):
        p.mkdir(parents=True)
        meta = t.select(META).to_pylist()
        emb = emb_np(t.column("embedding"))
        imgs = t.column("image").to_pylist()
        for i, m in enumerate(meta):
            k = m["id"]
            sub = p / f"{k // 1000:04d}"
            sub.mkdir(exist_ok=True)
            (sub / f"{k}.jpg").write_bytes(imgs[i])
            (sub / f"{k}.json").write_text(json.dumps(m))
            (sub / f"{k}.emb").write_bytes(emb[i].tobytes())
        self.p, self.ids = p, [m["id"] for m in meta]

    def _get(self, k, parts=("jpg", "json", "emb")):
        sub = self.p / f"{k // 1000:04d}"
        out = {}
        if "jpg" in parts:
            out["image"] = (sub / f"{k}.jpg").read_bytes()
        if "json" in parts:
            out["meta"] = json.loads((sub / f"{k}.json").read_text())
        if "emb" in parts:
            out["emb"] = np.frombuffer((sub / f"{k}.emb").read_bytes(), np.float32)
        return out

    def scan(self):
        return sum(len(self._get(k)["image"]) for k in self.ids)

    def take(self, ids):
        out = [self._get(int(k)) for k in ids]
        self.io_req += 3 * len(ids)
        self.io_bytes += sum(len(o["image"]) + 4 * EMB_DIM + 200 for o in out)
        return out

    def embeddings(self):
        return np.stack([self._get(k, ("emb",))["emb"] for k in self.ids])

    def metadata(self):
        return [self._get(k, ("json",))["meta"] for k in self.ids]

    def filter(self):
        return [self._get(k, ("jpg",))["image"] for k in self.ids
                if self._get(k, ("json",))["meta"]["label"] < FILTER_LABELS]


class JSONL:
    name = "JSONL (base64)"

    def write(self, t, p):
        p.mkdir(parents=True)
        f = p / "data.jsonl"
        emb = emb_np(t.column("embedding"))
        offs = []
        with open(f, "wb") as fh:
            for i, r in enumerate(t.select(META + ["image"]).to_pylist()):
                r["image"] = base64.b64encode(r["image"]).decode()
                r["embedding"] = emb[i].round(6).tolist()
                offs.append(fh.tell())
                fh.write(json.dumps(r).encode() + b"\n")
        np.save(p / "offsets.npy", np.array(offs, np.int64))
        self.f, self.offs = f, np.array(offs)

    def _rows(self):
        with open(self.f, "rb") as fh:
            for line in fh:
                yield json.loads(line)

    def scan(self):
        return sum(len(base64.b64decode(r["image"])) for r in self._rows())

    def take(self, ids):
        out = []
        with open(self.f, "rb") as fh:
            for k in ids:
                fh.seek(self.offs[k])
                line = fh.readline()
                self.io_req += 1
                self.io_bytes += len(line)
                r = json.loads(line)
                r["image"] = base64.b64decode(r["image"])
                out.append(r)
        return out

    def embeddings(self):
        return np.array([r["embedding"] for r in self._rows()], np.float32)

    def metadata(self):
        return [{k: r[k] for k in META} for r in self._rows()]

    def filter(self):
        return [base64.b64decode(r["image"]) for r in self._rows() if r["label"] < FILTER_LABELS]


class Tar:
    name = "Tar shards (WebDataset)"
    SHARD = 1000

    def write(self, t, p):
        p.mkdir(parents=True)
        meta = t.select(META).to_pylist()
        emb = emb_np(t.column("embedding"))
        imgs = t.column("image").to_pylist()
        self.index = {}
        for s in range(0, len(meta), self.SHARD):
            path = p / f"shard-{s // self.SHARD:05d}.tar"
            with tarfile.open(path, "w") as tf:
                for i in range(s, min(s + self.SHARD, len(meta))):
                    k = meta[i]["id"]
                    for ext, data in (("jpg", imgs[i]), ("json", json.dumps(meta[i]).encode()),
                                      ("emb", emb[i].tobytes())):
                        ti = tarfile.TarInfo(f"{k}.{ext}")
                        ti.size = len(data)
                        tf.addfile(ti, io.BytesIO(data))
            with tarfile.open(path) as tf:  # build random-access index (offset_data)
                for m in tf.getmembers():
                    k, ext = m.name.split(".")
                    self.index.setdefault(int(k), {})[ext] = (path, m.offset_data, m.size)
        self.p = p

    def _iter(self, want=("jpg", "json", "emb")):
        for path in sorted(self.p.glob("*.tar")):
            with tarfile.open(path) as tf:  # sequential streaming read, as WebDataset does
                for m in tf:
                    ext = m.name.split(".")[1]
                    if ext in want:
                        yield ext, tf.extractfile(m).read()

    def scan(self):
        return sum(len(d) for e, d in self._iter() if e == "jpg")

    def take(self, ids):
        out, handles = [], {}
        for k in ids:
            rec = {}
            for ext, (path, off, size) in self.index[int(k)].items():
                fh = handles.get(path) or handles.setdefault(path, open(path, "rb"))
                fh.seek(off)
                rec[ext] = fh.read(size)
                self.io_req += 1
                self.io_bytes += size
            out.append(rec)
        for fh in handles.values():
            fh.close()
        return out

    def embeddings(self):
        return np.stack([np.frombuffer(d, np.float32) for e, d in self._iter(("emb",)) if e == "emb"])

    def metadata(self):
        return [json.loads(d) for e, d in self._iter(("json",)) if e == "json"]

    def filter(self):
        out, keep = [], False
        for e, d in self._iter(("jpg", "json")):
            if e == "jpg":
                img = d
            else:  # json follows jpg within a sample
                if json.loads(d)["label"] < FILTER_LABELS:
                    out.append(img)
        return out


class HDF5:
    name = "HDF5"

    def write(self, t, p):
        p.mkdir(parents=True)
        self.f = p / "data.h5"
        with h5py.File(self.f, "w") as h:
            for c in ["id", "label", "score", "width", "height"]:
                h[c] = t.column(c).to_numpy()
            h.create_dataset("split", data=t.column("split").to_pylist(), dtype=h5py.string_dtype())
            h.create_dataset("caption", data=t.column("caption").to_pylist(), dtype=h5py.string_dtype())
            h.create_dataset("embedding", data=emb_np(t.column("embedding")), chunks=(256, EMB_DIM))
            dt = h5py.vlen_dtype(np.uint8)
            ds = h.create_dataset("image", (t.num_rows,), dtype=dt, chunks=(256,))
            for i, b in enumerate(t.column("image").to_pylist()):
                ds[i] = np.frombuffer(b, np.uint8)

    def scan(self):
        with h5py.File(self.f) as h:
            _ = [h[c][:] for c in ["id", "label", "score", "caption", "embedding"]]
            return sum(len(x) for x in h["image"][:])

    def take(self, ids):
        with h5py.File(self.f) as h:
            ds, emb, lab, cap = h["image"], h["embedding"], h["label"], h["caption"]
            return [(ds[k], emb[k], lab[k], cap[k]) for k in map(int, ids)]

    def embeddings(self):
        with h5py.File(self.f) as h:
            return h["embedding"][:]

    def metadata(self):
        with h5py.File(self.f) as h:
            return [h[c][:] for c in ["id", "label", "score", "width", "height", "split", "caption"]]

    def filter(self):
        with h5py.File(self.f) as h:
            idx = np.nonzero(h["label"][:] < FILTER_LABELS)[0]
            ds = h["image"]
            return [ds[i] for i in idx]


class ArrowIPC:
    name = "Arrow IPC (mmap)"

    def write(self, t, p):
        p.mkdir(parents=True)
        self.f = p / "data.arrow"
        feather.write_feather(t, self.f, compression="uncompressed", chunksize=1024)
        self.t = feather.read_table(self.f, memory_map=True)  # opened once, like other handles

    def _t(self, cols=None):
        return feather.read_table(self.f, columns=cols, memory_map=True)

    def scan(self):
        t = self._t()
        return touch(t.column("image"))

    def take(self, ids):
        out = self.t.take(pa.array(ids))
        self.io_bytes += out.nbytes  # mmap: only touched rows are paged in
        return out

    def embeddings(self):
        return emb_np(self._t(["embedding"]).column("embedding"))

    def metadata(self):
        return self._t(META)

    def filter(self):
        t = self._t(["id", "label", "image"])
        return t.filter(pc.less(t.column("label"), FILTER_LABELS))


class Parquet:
    def __init__(self, rg, codec, sort=False, page_index=True):
        self.rg, self.codec, self.sort = rg, codec, sort
        self.name = f"Parquet ({codec}, rg={rg if rg < 20000 else 'all'}{', sorted' if sort else ''})"

    def write(self, t, p):
        p.mkdir(parents=True)
        if self.sort:
            t = t.sort_by("label")
        self.f = p / "data.parquet"
        pq.write_table(t, self.f, row_group_size=self.rg, compression=self.codec,
                       write_page_index=True, use_dictionary=["split"])
        self.pf = pq.ParquetFile(self.f)
        rgs = [self.pf.metadata.row_group(i).num_rows for i in range(self.pf.num_row_groups)]
        self.rg_start = np.concatenate([[0], np.cumsum(rgs)])
        # position of each id (rows may be re-ordered by sorting)
        ids = t.column("id").to_numpy()
        self.pos = np.empty(len(ids), np.int64)
        self.pos[ids] = np.arange(len(ids))

    def scan(self):
        t = pq.read_table(self.f)
        return touch(t.column("image"))

    def take(self, ids):
        pos = self.pos[np.asarray(ids)]
        rg = np.searchsorted(self.rg_start, pos, side="right") - 1
        out = []
        for g in np.unique(rg):
            t = self.pf.read_row_group(int(g))
            md = self.pf.metadata.row_group(int(g))
            self.io_req += md.num_columns
            self.io_bytes += sum(md.column(c).total_compressed_size for c in range(md.num_columns))
            out.append(t.take(pa.array(pos[rg == g] - self.rg_start[g])))
        return out

    def embeddings(self):
        return emb_np(pq.read_table(self.f, columns=["embedding"]).column("embedding"))

    def metadata(self):
        return pq.read_table(self.f, columns=META)

    def filter(self):
        return pq.read_table(self.f, columns=["id", "label", "image"],
                             filters=[("label", "<", FILTER_LABELS)])


class Lance:
    def __init__(self, sort=False):
        self.sort = sort
        self.name = "Lance" + (" (sorted)" if sort else "")

    def write(self, t, p):
        if self.sort:
            t = t.sort_by("label")
        self.p = p
        self.ds = lance.write_dataset(t, p, max_rows_per_group=1024)
        ids = t.column("id").to_numpy()
        self.pos = np.empty(len(ids), np.int64)
        self.pos[ids] = np.arange(len(ids))

    def scan(self):
        t = self.ds.to_table()
        return touch(t.column("image"))

    def take(self, ids):
        return self.ds.take(self.pos[np.asarray(ids)].tolist())

    def embeddings(self):
        return emb_np(self.ds.to_table(columns=["embedding"]).column("embedding"))

    def metadata(self):
        return self.ds.to_table(columns=META)

    def filter(self):
        return self.ds.to_table(columns=["id", "label", "image"], filter=f"label < {FILTER_LABELS}")


class BlobSplit:
    """Decoupled layout: small/mutable columns in Parquet, image bytes in an
    append-only blob file addressed by (offset, length)."""
    name = "Parquet + blob file"

    def write(self, t, p):
        p.mkdir(parents=True)
        imgs = t.column("image").to_pylist()
        offs, lens, off = [], [], 0
        with open(p / "images.blob", "wb") as fh:
            for b in imgs:
                offs.append(off)
                lens.append(len(b))
                fh.write(b)
                off += len(b)
        m = t.drop_columns(["image"]).append_column("img_off", pa.array(offs, pa.int64())) \
             .append_column("img_len", pa.array(lens, pa.int32()))
        self.f, self.blob = p / "meta.parquet", p / "images.blob"
        pq.write_table(m, self.f, row_group_size=1024, compression="snappy")
        self.m = pq.read_table(self.f, columns=["img_off", "img_len"])  # resident index
        self.offs, self.lens = self.m.column("img_off").to_numpy(), self.m.column("img_len").to_numpy()
        self.pf = pq.ParquetFile(self.f)

    def scan(self):
        t = pq.read_table(self.f)
        data = self.blob.read_bytes()
        return len(data) + t.num_rows * 0

    def take(self, ids):
        ids = np.asarray(ids)
        rg = ids // 1024
        meta = []
        for g in np.unique(rg):
            meta.append(self.pf.read_row_group(int(g), columns=META + ["embedding"]).take(pa.array(ids[rg == g] - g * 1024)))
            md = self.pf.metadata.row_group(int(g))
            self.io_req += len(META) + 1
            self.io_bytes += sum(md.column(c).total_compressed_size for c in range(md.num_columns))
        with open(self.blob, "rb") as fh:
            imgs = []
            for k in ids:
                fh.seek(self.offs[k])
                imgs.append(fh.read(self.lens[k]))
                self.io_req += 1
                self.io_bytes += int(self.lens[k])
        return meta, imgs

    def embeddings(self):
        return emb_np(pq.read_table(self.f, columns=["embedding"]).column("embedding"))

    def metadata(self):
        return pq.read_table(self.f, columns=META)

    def filter(self):
        t = pq.read_table(self.f, columns=["id", "label", "img_off", "img_len"],
                          filters=[("label", "<", FILTER_LABELS)])
        with open(self.blob, "rb") as fh:
            out = []
            for o, l in zip(t.column("img_off").to_numpy(), t.column("img_len").to_numpy()):
                fh.seek(o)
                out.append(fh.read(l))
        return out


def layouts():
    L = [Files(), JSONL(), Tar(), HDF5(), ArrowIPC(),
         Parquet(1 << 20, "snappy"), Parquet(1024, "snappy"), Parquet(1024, "zstd"),
         Parquet(1024, "none"), Parquet(128, "snappy"), Parquet(1024, "snappy", sort=True)]
    if lance is not None:
        L += [Lance(), Lance(sort=True)]
    L += [BlobSplit()]
    return L


def main(only=None):
    src = feather.read_table(DATA / "source.arrow")
    n = src.num_rows
    raw = sum(len(b) for b in src.column("image").to_pylist())
    rand = RNG.permutation(n)[:N_RANDOM]
    batches = [rand[i:i + BATCH] for i in range(0, N_RANDOM, BATCH)]
    results = []
    for L in layouts():
        if only and only not in L.name:
            continue
        p = fresh(OUT / L.name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", ""))
        t0 = time.perf_counter()
        L.write(src, p)
        w = time.perf_counter() - t0
        r = {"layout": L.name, "write_s": w, "bytes": dir_size(p), "files": file_count(p)}
        L.scan()  # warm-up (page cache)
        slow = isinstance(L, (Files, JSONL))
        reps = 1 if slow else 3
        r["scan_s"], _ = timed(L.scan, reps)
        L.io_req = L.io_bytes = 0
        _ = [L.take(b) for b in batches]
        r["take_req_per_batch"] = L.io_req / len(batches) if L.io_req else None
        r["take_MB_per_batch"] = L.io_bytes / 1e6 / len(batches) if L.io_bytes else None
        r["take_s"], _ = timed(lambda: [L.take(b) for b in batches], reps)
        r["emb_s"], e = timed(L.embeddings, reps)
        assert e.shape == (n, EMB_DIM), (L.name, e.shape)
        r["meta_s"], _ = timed(L.metadata, reps)
        r["filter_s"], f = timed(L.filter, reps)
        r["filter_rows"] = len(f) if not hasattr(f, "num_rows") else f.num_rows
        r["scan_MBps"] = raw / 1e6 / r["scan_s"]
        r["take_sps"] = N_RANDOM / r["take_s"]
        results.append(r)
        print(f"{L.name:34s} write={w:6.2f}s size={r['bytes'] / 1e6:7.1f}MB files={r['files']:6d} "
              f"scan={r['scan_s']:6.3f}s take={r['take_sps']:9.0f}/s emb={r['emb_s']:6.3f}s "
              f"meta={r['meta_s']:6.3f}s filter={r['filter_s']:6.3f}s ({r['filter_rows']})", flush=True)
    save_json("layout.json" if not only else f"layout_{only}.json",
              {"n": n, "raw_image_bytes": raw, "batch": BATCH, "n_random": N_RANDOM, "rows": results})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
