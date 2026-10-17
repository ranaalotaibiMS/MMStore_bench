# Blobs, Columns, and Versions — MMStore-Bench

**Blobs, Columns, and Versions: Benchmarking Storage Layouts and Time-Travel**

Multimodal AI datasets combine large binary objects (images), dense embeddings, free text and small metadata, and they change all the time through relabeling, re-embedding, deduplication and deletion requests. **MMStore-Bench** is an open, reproducible benchmark that answers two questions on the same corpus:

1. **Which physical layout** best serves access patterns: shuffled random access for training, column projections for embedding jobs, and selective scans for curation?
2. **How should the data be versioned** so that any past state can be reproduced exactly without multiplying storage?

## Key results

- Random-access throughput differs by **two orders of magnitude** across layouts. Parquet, which reads whole row groups, reads up to **361×** more bytes than a training batch needs.
- Clustering by the filter attribute speeds up selective curation scans by **4.9×**.
- After 13 versions, storage ranges from **1.42×** (Lance) to **11.8×** (full snapshots) the size of the latest version. The main factor is whether a strategy rewrites the unchanged image bytes when mutable columns such as embeddings change. Image bytes are 83.8% of the data, embeddings 15.9%.
- Separating images from the table pays off: Delta Lake with a separate blob store needs 2.42×, compared with 9.2× for plain Delta Lake.

## Benchmark design

**Corpus**: synthetic image–text samples that match web-scale vision–language statistics. Each sample has an id, a label (one of 100 classes), a split, a quality score, width and height, an 8–20-word caption, a unit-norm 512-d float32 embedding, and a 224×224 JPEG (quality 85, about 10.8 KB). The layout benchmark uses 20,000 samples (216 MB of images). The versioning benchmark starts from 10,000.

**Layouts (14)**: files per sample, JSONL (base64 images), WebDataset tar shards, HDF5, Arrow IPC (memory-mapped Feather v2), Parquet (row-group size 128 / 1,024 / one group per file; codec none / Snappy / Zstd; with or without clustering), Lance (with or without clustering), and Parquet + a separate blob file.

**Workloads**: W1 full scan · W2 random take (32 batches × 64 ids, with bytes and requests counted per batch) · W3 embedding projection · W4 metadata scan · W5 selective filter (`label < 5`, 5.2%) · W6 ingest (time, size, file count).

**Versioning lifecycle**: an initial load followed by 12 commits: appends, relabels, GDPR-style deletes, caption fixes, two full re-embeddings, image edits, and a new column. The benchmark tests **8 strategies**: full snapshots, file-level CAS, chunk-level CAS (fixed-size, content-defined, and Parquet's content-defined writer), Delta Lake, Lance, and Delta Lake + blob store. Every historical version is read back and checked against a SHA-256 digest of the ground truth, so a wrong implementation cannot report results.

## Contents

- `experiments/`: benchmark code (about 1,500 lines of Python, in a Python 3.12 venv at `experiments/.venv`)
  - `common.py`: shared helpers (paths, row generator, timing, sizes)
  - `gen_data.py [N]`: generates the synthetic corpus (`data/source.arrow`, default 20,000 samples)
  - `bench_layout.py [layout]`: 14 layouts × workloads W1–W6 (pass a layout name to run only that one)
  - `bench_sweep.py`: Parquet row-group sweep, JPEG decode rate, embedding precision
  - `bench_parallel.py`: random-access scaling with threads
  - `bench_e2e.py`: end-to-end data loader including JPEG decode
  - `bench_batch.py`: effect of batch size on random take
  - `bench_version.py [N0] [strategy]`: 13-version lifecycle × 8 strategies, checks every version (default 10,000 initial samples)
  - `bench_compact.py`: 60 small commits, then compaction and vacuum (Delta, Lance)
  - `bench_blobsize.py`: versioning sensitivity to image size (128 / 224 / 448 px)
  - `make_figs.py`: turns `results/*.json` into `tables/*.tex` and the macros
  - `results/`: raw JSON results. `results_*.log` holds the run logs.

## Environment

The paper's numbers come from an Apple M3 Pro (12 cores, 36 GB RAM, NVMe SSD) running macOS 15.6, with Python 3.12, PyArrow 26.0, Lance 13.0, deltalake (delta-rs) 1.6.6, h5py 3.16, Pillow 12.3 and NumPy. Readers are single-threaded unless a script says otherwise. Each value is the median of 3 warm runs, so timings reflect format and CPU cost. The byte and request counters reflect cold or remote (object-storage) I/O.

## Reproduce

Takes about 15 minutes:

```bash
cd experiments && .venv/bin/python gen_data.py 20000 && .venv/bin/python bench_layout.py && .venv/bin/python bench_sweep.py && .venv/bin/python bench_parallel.py && .venv/bin/python bench_e2e.py && .venv/bin/python bench_batch.py && .venv/bin/python bench_version.py 10000 && .venv/bin/python bench_compact.py && .venv/bin/python bench_blobsize.py && .venv/bin/python make_figs.py
```

Generated data goes in `experiments/data/`. It takes several GB and is safe to delete.

## Extending

- **New layout**: implement `write`, `scan`, `take`, `embeddings`, `metadata` and `filter` over an Arrow table. Request and byte counters for random access are optional.
- **New versioning strategy**: implement `commit` (which receives the operation type, its payload and the full new state), `read(v)` and `lookup(v, ids)`. The harness replays the lifecycle, times each commit and verifies every version.
