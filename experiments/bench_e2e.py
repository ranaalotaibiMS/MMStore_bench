"""Benchmark 1d: end-to-end loader = random take + JPEG decode + resize to 128px,
with 1 and 8 worker threads (Pillow releases the GIL while decoding)."""
import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from PIL import Image

from bench_layout import BATCH, OUT, ArrowIPC, BlobSplit, Files, Lance, Parquet, Tar
from bench_parallel import thread_safe
from common import DATA, fresh, save_json

N_BATCHES = 64


def images_of(out):
    if isinstance(out, tuple):                       # BlobSplit -> (meta, imgs)
        return out[1]
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return [o.get("image") or o.get("jpg") for o in out]   # Files / Tar
    if isinstance(out, list):                        # Parquet -> list of tables
        return [b for t in out for b in t.column("image").to_pylist()]
    return out.column("image").to_pylist()           # Arrow / Lance


def decode(b):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    im.draft("RGB", (128, 128))
    return np.asarray(im.resize((128, 128)))


def main():
    src = feather.read_table(DATA / "source.arrow")
    rng = np.random.default_rng(11)
    batches = [rng.choice(src.num_rows, BATCH, replace=False) for _ in range(N_BATCHES)]
    res = {"decode_only": {}, "rows": []}
    imgs = src.column("image").to_pylist()

    def dec_only(b):
        return [decode(imgs[i]) for i in b]
    for w in (1, 8):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(w) as ex:
            list(ex.map(dec_only, batches))
        res["decode_only"][w] = N_BATCHES * BATCH / (time.perf_counter() - t0)
    print("decode only", res["decode_only"], flush=True)

    for L in [Files(), Tar(), ArrowIPC(), Parquet(1024, "snappy"), Lance(), BlobSplit()]:
        p = fresh(OUT / ("e2e_" + L.name.replace(" ", "_").replace("(", "").replace(")", "")
                         .replace(",", "").replace("=", "")))
        L.write(src, p)
        L = thread_safe(L)
        L.io_req = L.io_bytes = 0

        def step(b):
            return [decode(x) for x in images_of(L.take(b))]
        step(batches[0])
        row = {"layout": L.name}
        for w in (1, 8):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(w) as ex:
                list(ex.map(step, batches))
            row[w] = N_BATCHES * BATCH / (time.perf_counter() - t0)
        res["rows"].append(row)
        print(row, flush=True)
    save_json("e2e.json", res)


if __name__ == "__main__":
    main()
