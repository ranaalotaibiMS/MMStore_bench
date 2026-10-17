"""Benchmark 2b: many small commits, fragmentation, compaction, and history retention.

Starting from 5,000 samples, apply 60 small commits (append 100 rows / relabel 0.5% /
delete 0.2%, round-robin) to Delta Lake and Lance. Every 10 commits record file count,
bytes, latest full-read latency and point-lookup latency. Then compact + vacuum and
record what is reclaimed and whether time travel to old versions survives.
"""
import time
from datetime import timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from deltalake import DeltaTable, write_deltalake

import lance
from common import DATA, dir_size, file_count, fresh, make_rows, save_json, timed

OUT = DATA / "compact"
N0, COMMITS = 5000, 60


def ops(rng):
    nid = N0
    for i in range(COMMITS):
        k = i % 3
        if k == 0:
            yield "append", make_rows(100, nid, seed=10_000 + i)
            nid += 100
        elif k == 1:
            yield "relabel", 0.005
        else:
            yield "delete", 0.002


def run(kind):
    rng = np.random.default_rng(3)
    p = fresh(OUT / kind)
    uri = str(p)
    base = make_rows(N0, 0, seed=99)
    live = set(range(N0))
    if kind == "delta":
        write_deltalake(uri, base)
    else:
        lance.write_dataset(base, uri, max_rows_per_group=1024)
    probe = list(range(0, N0, 50))[:100]

    def read_latest():
        return DeltaTable(uri).to_pyarrow_table() if kind == "delta" else lance.dataset(uri).to_table()

    def lookup():
        ids = [k for k in probe if k in live]
        if kind == "delta":
            return DeltaTable(uri).to_pyarrow_dataset().to_table(filter=pc.field("id").isin(ids))
        return lance.dataset(uri).to_table(filter=f"id IN ({','.join(map(str, ids))})")

    trace = []

    def snap(i, label=None):
        rs, _ = timed(read_latest)
        ls, _ = timed(lookup)
        trace.append({"commit": i, "label": label, "files": file_count(p), "bytes": dir_size(p),
                      "read_s": rs, "lookup_s": ls})
        print(kind, trace[-1], flush=True)

    snap(0)
    commit_s = []
    for i, (op, arg) in enumerate(ops(rng), 1):
        t0 = time.perf_counter()
        if op == "append":
            if kind == "delta":
                write_deltalake(uri, arg, mode="append")
            else:
                lance.write_dataset(arg, uri, mode="append")
            live |= set(arg.column("id").to_pylist())
        else:
            ids = rng.choice(sorted(live), max(1, int(arg * len(live))), replace=False).tolist()
            pred = f"id in ({','.join(map(str, ids))})"
            if op == "delete":
                (DeltaTable(uri).delete(pred) if kind == "delta" else lance.dataset(uri).delete(pred))
                live -= set(ids)
            else:
                if kind == "delta":
                    DeltaTable(uri).update(updates={"label": "(label + 1) % 100"}, predicate=pred)
                else:
                    lance.dataset(uri).update({"label": "(label + 1) % 100"}, where=pred)
        commit_s.append(time.perf_counter() - t0)
        if i % 10 == 0:
            snap(i)
    n_versions = (len(DeltaTable(uri).history()) if kind == "delta" else len(lance.dataset(uri).versions()))

    # compaction + retention
    t0 = time.perf_counter()
    if kind == "delta":
        dt = DeltaTable(uri)
        dt.optimize.compact()
        t_compact = time.perf_counter() - t0
        DeltaTable(uri).vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
    else:
        ds = lance.dataset(uri)
        ds.optimize.compact_files(target_rows_per_fragment=1 << 20)
        t_compact = time.perf_counter() - t0
        lance.dataset(uri).cleanup_old_versions(older_than=timedelta(seconds=0), delete_unverified=True)
    t_total = time.perf_counter() - t0
    snap(COMMITS, "after compaction + vacuum")
    # does time travel to the initial version survive?
    try:
        (DeltaTable(uri, version=0).to_pyarrow_table() if kind == "delta"
         else lance.dataset(uri, version=1).to_table())
        tt = True
    except Exception as e:  # expected: files of old versions were removed
        tt = False
    return {"kind": kind, "trace": trace, "commit_s": commit_s, "versions": n_versions,
            "compact_s": t_compact, "compact_vacuum_s": t_total, "old_version_readable": tt}


if __name__ == "__main__":
    save_json("compact.json", [run("delta"), run("lance")])
