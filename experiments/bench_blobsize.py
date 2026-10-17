"""Benchmark 2c: sensitivity of version storage to blob (image) size.
Runs the lifecycle at 5,000 initial samples for three image resolutions."""
import os
import subprocess
import sys

STRATS = "Full copy,File-level CAS,Delta Lake,Lance,Delta + blob CAS"
for size in (128, 224, 448):
    env = dict(os.environ, MMSTORE_IMG=str(size), MMSTORE_OUT=f"blob_{size}.json")
    subprocess.run([sys.executable, "bench_version.py", "5000", STRATS], env=env, check=True)
