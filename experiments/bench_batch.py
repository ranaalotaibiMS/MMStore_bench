"""Benchmark 1e: random-take throughput vs. mini-batch size, and per-column byte breakdown."""
import numpy as np
import pyarrow.feather as feather
import pyarrow.parquet as pq

from bench_layout import OUT, ArrowIPC, BlobSplit, Lance, Parquet, Tar
from common import DATA, fresh, save_json, timed

SIZES = [1, 8, 64, 512]
TOTAL = 2048


def main():
    src = feather.read_table(DATA / "source.arrow")
    n = src.num_rows
    rng = np.random.default_rng(21)
    res = {"sizes": SIZES, "rows": [], "columns": []}
    for L in [Tar(), ArrowIPC(), Parquet(1024, "snappy"), Parquet(128, "snappy"), Lance(), BlobSplit()]:
        p = fresh(OUT / ("bs_" + L.name.replace(" ", "_").replace("(", "").replace(")", "")
                         .replace(",", "").replace("=", "")))
        L.write(src, p)
        L.io_req = L.io_bytes = 0
        row = {"layout": L.name, "sps": {}}
        for b in SIZES:
            nb = max(1, TOTAL // b) if b < 512 else 8
            batches = [rng.choice(n, b, replace=False) for _ in range(nb)]
            t, _ = timed(lambda: [L.take(x) for x in batches])
            row["sps"][b] = nb * b / t
        res["rows"].append(row)
        print(row, flush=True)
    # bytes per column in a Parquet file (snappy, rg=1024)
    f = DATA / "layouts" / "Parquet_snappy_rg1024" / "data.parquet"
    md = pq.ParquetFile(f).metadata
    tot = {}
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for c in range(rg.num_columns):
            col = rg.column(c)
            tot[col.path_in_schema.split(".")[0]] = tot.get(col.path_in_schema.split(".")[0], 0) + col.total_compressed_size
    res["columns"] = tot
    print(tot)
    save_json("batch.json", res)


if __name__ == "__main__":
    main()
