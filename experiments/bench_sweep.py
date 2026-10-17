"""Benchmark 1b: Parquet row-group sweep, embedding precision, and JPEG decode cost."""
import io
import time

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
from PIL import Image

from bench_layout import BATCH, N_RANDOM, Parquet, emb_np
from common import DATA, EMB_DIM, dir_size, fresh, save_json, timed

OUT = DATA / "sweep"


def main():
    src = feather.read_table(DATA / "source.arrow")
    n = src.num_rows
    rand = np.random.default_rng(42).permutation(n)[:N_RANDOM]
    batches = [rand[i:i + BATCH] for i in range(0, N_RANDOM, BATCH)]
    res = {"rg": [], "decode": {}, "emb_precision": []}

    for rg in [32, 64, 128, 256, 512, 1024, 2048, 4096, 20000]:
        L = Parquet(rg, "snappy")
        p = fresh(OUT / f"rg{rg}")
        t0 = time.perf_counter()
        L.write(src, p)
        w = time.perf_counter() - t0
        L.scan()
        r = {"rg": rg, "write_s": w, "bytes": dir_size(p),
             "footer_bytes": L.pf.metadata.serialized_size}
        r["scan_s"], _ = timed(L.scan)
        L.io_req = L.io_bytes = 0
        _ = [L.take(b) for b in batches]
        r["take_MB_per_batch"] = L.io_bytes / 1e6 / len(batches)
        r["take_req_per_batch"] = L.io_req / len(batches)
        r["take_s"], _ = timed(lambda: [L.take(b) for b in batches])
        r["take_sps"] = N_RANDOM / r["take_s"]
        r["emb_s"], _ = timed(L.embeddings)
        res["rg"].append(r)
        print(r, flush=True)

    # JPEG decode cost (the consumer side of a training pipeline)
    imgs = src.column("image").take(pa.array(rand)).to_pylist()
    dec_s, _ = timed(lambda: [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in imgs])
    res["decode"] = {"images": len(imgs), "seconds": dec_s, "per_s": len(imgs) / dec_s}
    print(res["decode"], flush=True)

    # Embedding precision: storage vs. fidelity (cosine to fp32)
    emb = emb_np(src.column("embedding"))
    for name, arr in [("float32", emb), ("float16", emb.astype(np.float16)),
                      ("int8", np.clip(np.round(emb * 127 / np.abs(emb).max()), -127, 127).astype(np.int8))]:
        t = pa.table({"embedding": pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1)), EMB_DIM)})
        p = fresh(OUT / f"emb_{name}.parquet")
        pq.write_table(t, p, compression="zstd")
        back = emb_np(pq.read_table(p).column("embedding")).astype(np.float32)
        cos = float(np.mean(np.sum(back * emb, 1) / (np.linalg.norm(back, axis=1) * np.linalg.norm(emb, axis=1))))
        # recall@10 of brute-force nearest neighbours for 200 queries
        q = emb[:200]
        top_true = np.argsort(-q @ emb.T, axis=1)[:, 1:11]
        top_q = np.argsort(-q @ back.T, axis=1)[:, 1:11]
        recall = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top_true, top_q)]))
        rd, _ = timed(lambda: pq.read_table(p))
        res["emb_precision"].append({"dtype": name, "bytes": p.stat().st_size, "cosine": cos,
                                     "recall10": recall, "read_s": rd})
        print(res["emb_precision"][-1], flush=True)
    save_json("sweep.json", res)


if __name__ == "__main__":
    main()
