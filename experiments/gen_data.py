"""Generate the synthetic multimodal corpus used by both benchmarks."""
import sys
import time

import pyarrow as pa
import pyarrow.feather as feather

from common import DATA, make_rows

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000

if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    parts = [make_rows(2_000, start_id=i, seed=i) for i in range(0, N, 2_000)]
    tbl = pa.concat_tables(parts)
    feather.write_feather(tbl, DATA / "source.arrow", compression="uncompressed")
    img = sum(len(b) for b in tbl.column("image").to_pylist())
    print(f"{tbl.num_rows} rows in {time.perf_counter() - t0:.1f}s; "
          f"image bytes {img / 1e6:.1f} MB (avg {img / tbl.num_rows / 1e3:.1f} KB); "
          f"arrow file {(DATA / 'source.arrow').stat().st_size / 1e6:.1f} MB")
