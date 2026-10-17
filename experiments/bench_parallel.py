"""Benchmark 1c: random-access throughput with parallel reader threads.

Each worker thread fetches whole 64-sample mini-batches; handles that are not
thread-safe (ParquetFile, HDF5 files, open file objects) are opened per call.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.feather as feather
import pyarrow.parquet as pq

from bench_layout import (BATCH, OUT, ArrowIPC, BlobSplit, Files, HDF5, Lance, Parquet, Tar)
from common import DATA, fresh, save_json

N_BATCHES = 128
WORKERS = [1, 2, 4, 8]
_local = threading.local()


def thread_safe(L):
    """Give Parquet-based layouts a per-thread ParquetFile handle."""
    if hasattr(L, "pf"):
        f = L.f
        cls = type(L)

        class PF:
            def __getattr__(self, k):
                pf = getattr(_local, str(f), None)
                if pf is None:
                    pf = pq.ParquetFile(f)
                    setattr(_local, str(f), pf)
                return getattr(pf, k)
        L.pf = PF()
    return L


def main():
    src = feather.read_table(DATA / "source.arrow")
    n = src.num_rows
    rng = np.random.default_rng(7)
    batches = [rng.choice(n, BATCH, replace=False) for _ in range(N_BATCHES)]
    layouts = [Files(), Tar(), HDF5(), ArrowIPC(), Parquet(1024, "snappy"), Parquet(128, "snappy"),
               Lance(), BlobSplit()]
    res = []
    for L in layouts:
        p = fresh(OUT / ("par_" + L.name.replace(" ", "_").replace("(", "").replace(")", "")
                         .replace(",", "").replace("=", "")))
        L.write(src, p)
        L = thread_safe(L)
        L.io_req = L.io_bytes = 0
        for b in batches[:8]:  # warm-up
            L.take(b)
        row = {"layout": L.name, "sps": {}}
        for w in WORKERS:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(w) as ex:
                list(ex.map(L.take, batches))
            row["sps"][w] = N_BATCHES * BATCH / (time.perf_counter() - t0)
        res.append(row)
        print(f"{L.name:28s}", {k: round(v) for k, v in row["sps"].items()}, flush=True)
    save_json("parallel.json", {"workers": WORKERS, "batches": N_BATCHES, "rows": res})


if __name__ == "__main__":
    main()
