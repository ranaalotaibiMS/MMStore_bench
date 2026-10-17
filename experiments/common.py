"""Shared utilities: synthetic multimodal dataset, timing, sizes."""
import io
import json
import os
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
EMB_DIM = 512
IMG_SIZE = int(os.environ.get("MMSTORE_IMG", 224))
N_LABELS = 100

WORDS = ("a photo of the small large red blue green dog cat car tree house street beach "
         "mountain person bike city night sunset food table window river bird flower sky "
         "boat train plane chair book phone laptop field snow forest road bridge market").split()


def make_image(rng, label):
    """Smooth random field + mild noise, JPEG-encoded (~10-20 KB at q=85)."""
    base = rng.integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
    base[..., label % 3] = np.clip(base[..., label % 3].astype(int) + 60, 0, 255)  # label tint
    img = Image.fromarray(base).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(img).astype(np.int16) + rng.integers(-12, 13, size=(IMG_SIZE, IMG_SIZE, 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def make_caption(rng):
    return " ".join(rng.choice(WORDS, size=rng.integers(8, 20)))


def make_embedding(rng, n, label=None, seed_shift=0):
    e = rng.standard_normal((n, EMB_DIM), dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e


SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("label", pa.int32()),
    ("split", pa.string()),
    ("score", pa.float32()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("caption", pa.string()),
    ("embedding", pa.list_(pa.float32(), EMB_DIM)),
    ("image", pa.binary()),
])


def make_rows(n, start_id=0, seed=0):
    """Return a pyarrow Table with n multimodal samples."""
    rng = np.random.default_rng(seed)
    ids = np.arange(start_id, start_id + n, dtype=np.int64)
    labels = rng.integers(0, N_LABELS, size=n).astype(np.int32)
    split = rng.choice(["train", "val", "test"], size=n, p=[0.9, 0.05, 0.05])
    score = rng.random(n, dtype=np.float32) * 10
    caps = [make_caption(rng) for _ in range(n)]
    emb = make_embedding(rng, n)
    imgs = [make_image(rng, int(l)) for l in labels]
    return pa.table({
        "id": ids, "label": labels, "split": split, "score": score,
        "width": np.full(n, IMG_SIZE, np.int32), "height": np.full(n, IMG_SIZE, np.int32),
        "caption": caps,
        "embedding": pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1)), EMB_DIM),
        "image": pa.array(imgs, pa.binary()),
    }, schema=SCHEMA)


def dir_size(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def file_count(p):
    return sum(1 for f in Path(p).rglob("*") if f.is_file())


def fresh(p):
    p = Path(p)
    if p.exists():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    return p


def timed(fn, reps=3):
    """Run fn reps times; return (median seconds, last result)."""
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out


def save_json(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=float))
