"""Turn results/*.json into LaTeX tables, pgfplots coordinates and macros."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent.parent / "tables"
OUT.mkdir(exist_ok=True)
macros = {}


def fmt(x, d=1):
    if x is None:
        return "--"
    s = f"{x:,.{d}f}"
    return s.replace(",", "{,}")


def short(name):
    return (name.replace("Parquet ", "PQ ").replace("snappy", "snap").replace("rg=all", "rg=all")
            .replace("Tar shards (WebDataset)", "Tar (WebDataset)").replace("Files (per sample)", "Files/sample")
            .replace("Arrow IPC (mmap)", "Arrow IPC").replace("Parquet + blob file", "PQ + blob file"))


# ---------------------------------------------------------------- layout
L = json.loads((R / "layout.json").read_text())
rows = L["rows"]
useful_mb = L["raw_image_bytes"] / L["n"] * L["batch"] / 1e6
lines = []
for r in rows:
    amp = r["take_MB_per_batch"] / useful_mb if r["take_MB_per_batch"] else None
    lines.append(" & ".join([
        short(r["layout"]), fmt(r["bytes"] / 1e6, 0), fmt(r["files"], 0), fmt(r["write_s"], 2),
        fmt(r["scan_s"] * 1e3, 0), fmt(r["take_sps"], 0), (fmt(amp, 0) + r"$\times$") if amp else "n/i",
        fmt(r["emb_s"] * 1e3, 0), fmt(r["meta_s"] * 1e3, 0), fmt(r["filter_s"] * 1e3, 0)]) + r" \\")
(OUT / "layout_table.tex").write_text("\n".join(lines) + "\n")

srt = sorted(rows, key=lambda r: r["take_sps"])
(OUT / "take_bars.tex").write_text(
    r"\addplot coordinates {" + " ".join(f"({r['take_sps']:.0f},{i})" for i, r in enumerate(srt)) + "};\n")
(OUT / "take_labels.tex").write_text(r"\pgfplotsset{takelabels/.style={yticklabels={" +
                                     ",".join("{" + short(r["layout"]) + "}" for r in srt) + "}}}\n")
by = {r["layout"]: r for r in rows}
macros.update({
    "N": f"{L['n']:,}".replace(",", "{,}"),
    "RawMB": f"{L['raw_image_bytes'] / 1e6:.0f}",
    "AvgKB": f"{L['raw_image_bytes'] / L['n'] / 1e3:.1f}",
    "LanceTake": fmt(by["Lance"]["take_sps"], 0),
    "TarTake": fmt(by["Tar shards (WebDataset)"]["take_sps"], 0),
    "PqTakeAll": fmt(by["Parquet (snappy, rg=all)"]["take_sps"], 0),
    "PqTakeK": fmt(by["Parquet (snappy, rg=1024)"]["take_sps"], 0),
    "PqAmpK": fmt(by["Parquet (snappy, rg=1024)"]["take_MB_per_batch"] / useful_mb, 0),
    "BlobAmp": fmt(by["Parquet + blob file"]["take_MB_per_batch"] / useful_mb, 0),
    "UsefulMB": f"{useful_mb:.2f}",
    "JsonlMB": fmt(by["JSONL (base64)"]["bytes"] / 1e6, 0),
    "FilesScan": fmt(by["Files (per sample)"]["scan_s"], 2),
    "ArrowScan": fmt(by["Arrow IPC (mmap)"]["scan_s"] * 1e3, 0),
    "PqFilterUnsorted": fmt(by["Parquet (snappy, rg=1024)"]["filter_s"] * 1e3, 0),
    "PqFilterSorted": fmt(by["Parquet (snappy, rg=1024, sorted)"]["filter_s"] * 1e3, 0),
    "FilterSpeedup": fmt(by["Parquet (snappy, rg=1024)"]["filter_s"] / by["Parquet (snappy, rg=1024, sorted)"]["filter_s"], 1),
    "TarEmb": fmt(by["Tar shards (WebDataset)"]["emb_s"] * 1e3, 0),
    "PqEmb": fmt(by["Parquet (snappy, rg=1024)"]["emb_s"] * 1e3, 0),
    "EmbSpeedup": fmt(by["Tar shards (WebDataset)"]["emb_s"] / by["Parquet (snappy, rg=1024)"]["emb_s"], 0),
})

# ---------------------------------------------------------------- sweep
S = json.loads((R / "sweep.json").read_text())
rg = S["rg"]
(OUT / "rg_take.tex").write_text(r"\addplot coordinates {" + " ".join(f"({r['rg']},{r['take_sps']:.0f})" for r in rg) + "};\n")
(OUT / "rg_mb.tex").write_text(r"\addplot coordinates {" + " ".join(f"({r['rg']},{r['take_MB_per_batch']:.2f})" for r in rg) + "};\n")
(OUT / "rg_scan.tex").write_text(r"\addplot coordinates {" + " ".join(f"({r['rg']},{r['scan_s'] * 1e3:.1f})" for r in rg) + "};\n")
(OUT / "rg_footer.tex").write_text(r"\addplot coordinates {" + " ".join(f"({r['rg']},{r['footer_bytes'] / 1e3:.1f})" for r in rg) + "};\n")
macros.update({
    "DecodeRate": fmt(S["decode"]["per_s"], 0),
    "RgSmallTake": fmt(rg[0]["take_sps"], 0), "RgSmall": str(rg[0]["rg"]),
    "RgSmallFooter": fmt(rg[0]["footer_bytes"] / 1e3, 0), "RgBigFooter": fmt(rg[-1]["footer_bytes"] / 1e3, 1),
    "RgSmallScan": fmt(rg[0]["scan_s"] * 1e3, 0), "RgMidScan": fmt(min(r["scan_s"] for r in rg) * 1e3, 0),
})
el = []
for e in S["emb_precision"]:
    el.append(f"{e['dtype']} & {e['bytes'] / 1e6:.1f} & {e['cosine']:.5f} & {e['recall10']:.3f} & {e['read_s'] * 1e3:.0f} \\\\")
(OUT / "emb_table.tex").write_text("\n".join(el) + "\n")

# ---------------------------------------------------------------- versioning
vp = R / "version.json"
if vp.exists():
    V = json.loads(vp.read_text())
    commits = V["commits"]
    latest = V["logical_bytes"][-1]
    full = next(s for s in V["strategies"] if s["name"] == "Full copy")
    vl = []
    colors = {}
    for s in V["strategies"]:
        c = s["commit_s"]
        idx = {n: i for i, n in enumerate(commits)}
        reembed = [c[i] for i, n in enumerate(commits) if n == "re-embed 100%"]
        small = [c[i] for i, n in enumerate(commits) if n in ("relabel 2%", "delete 1%", "fix captions 5%")]
        vl.append(" & ".join([
            s["name"], fmt(s["bytes"][-1] / 1e6, 0), fmt(s["bytes"][-1] / latest, 2) + r"$\times$",
            fmt(s["files"][-1], 0), fmt(sum(small) / len(small) * 1e3, 0), fmt(sum(reembed) / len(reembed), 2),
            fmt(sum(c), 1), fmt(s["read_s"][0], 2), fmt(s["read_s"][-1], 2),
            fmt(s["lookup_v0_s"] * 1e3, 0), f"{sum(s['verified'])}/{len(s['verified'])}"]) + r" \\")
        s["_coords"] = " ".join(f"({i},{b / 1e6:.1f})" for i, b in enumerate(s["bytes"]))
        s["_read"] = " ".join(f"({i},{t:.3f})" for i, t in enumerate(s["read_s"]))
    (OUT / "version_table.tex").write_text("\n".join(vl) + "\n")
    (OUT / "version_growth.tex").write_text("\n".join(
        r"\addplot+[mark size=1.2pt] coordinates {" + s["_coords"] + "};\n" + r"\addlegendentry{" + s["name"] + "}"
        for s in V["strategies"]) + "\n" +
        r"\addplot+[black, dashed, mark=none] coordinates {" + " ".join(f"({i},{b / 1e6:.1f})" for i, b in enumerate(V["logical_bytes"])) + "};\n" + r"\addlegendentry{latest version only}")
    (OUT / "version_read.tex").write_text("\n".join(
        r"\addplot+[mark size=1.2pt] coordinates {" + s["_read"] + "};" for s in V["strategies"]) + "\n")
    # per-commit latency / storage-delta matrix
    abbrev = {"Full copy": "Full copy", "File-level CAS": "File CAS", "Chunk CAS (fixed 64KB)": "Fixed chunks",
              "Chunk CAS (CDC)": "CDC", "Chunk CAS (CDC, Parquet CDC)": "CDC + PQ-CDC", "Delta Lake": "Delta",
              "Lance": "Lance", "Delta + blob CAS": "Delta + blob"}
    head = " & ".join(["Commit"] + [abbrev[s["name"]] for s in V["strategies"]]) + r" \\"
    cm = [head, r"\midrule"]
    for i, n in enumerate(commits):
        cells = []
        for s in V["strategies"]:
            d = (s["bytes"][i] - (s["bytes"][i - 1] if i else 0)) / 1e6
            cells.append(f"{s['commit_s'][i]:.2f} / {d:.0f}")
        cm.append(" & ".join([f"{i}: {n.replace('v0 ', '').replace('%', chr(92) + '%')}"] + cells) + r" \\")
    (OUT / "commit_matrix.tex").write_text("\n".join(cm) + "\n")
    by = {s["name"]: s for s in V["strategies"]}
    macros.update({
        "NZero": f"{V['n0']:,}".replace(",", "{,}"),
        "NVersions": str(len(commits)),
        "LatestMB": fmt(latest / 1e6, 0),
        "FullGB": fmt(full["bytes"][-1] / 1e9, 2),
        "FullAmp": fmt(full["bytes"][-1] / latest, 1),
        "LanceAmp": fmt(by["Lance"]["bytes"][-1] / latest, 2),
        "DeltaAmp": fmt(by["Delta Lake"]["bytes"][-1] / latest, 1),
        "DeltaBlobAmp": fmt(by["Delta + blob CAS"]["bytes"][-1] / latest, 2),
        "FileCASAmp": fmt(by["File-level CAS"]["bytes"][-1] / latest, 2),
        "FixedAmp": fmt(by["Chunk CAS (fixed 64KB)"]["bytes"][-1] / latest, 1),
        "CDCAmp": fmt(by["Chunk CAS (CDC)"]["bytes"][-1] / latest, 2),
        "PCDCAmp": fmt(by["Chunk CAS (CDC, Parquet CDC)"]["bytes"][-1] / latest, 2),
        "FileCASFiles": fmt(by["File-level CAS"]["files"][-1], 0),
        "FileCASRead": fmt(by["File-level CAS"]["read_s"][0], 2),
        "FullRead": fmt(full["read_s"][0], 2),
    })

(OUT / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
print(f"wrote {len(macros)} macros")

# ---------------------------------------------------------------- parallel readers
P = json.loads((R / "parallel.json").read_text())
pl = []
for r in P["rows"]:
    pl.append(r"\addplot+[mark size=1.3pt] coordinates {" + " ".join(f"({w},{v:.0f})" for w, v in r["sps"].items()) + "};\n"
              + r"\addlegendentry{" + short(r["layout"]) + "}")
(OUT / "parallel.tex").write_text("\n".join(pl) + "\n")
pb = {r["layout"]: r["sps"] for r in P["rows"]}
macros.update({
    "LanceParMax": fmt(max(pb["Lance"].values()), 0),
    "TarParEight": fmt(pb["Tar shards (WebDataset)"]["8"], 0),
    "TarParOne": fmt(pb["Tar shards (WebDataset)"]["1"], 0),
    "BlobParEight": fmt(pb["Parquet + blob file"]["8"], 0),
    "BlobScale": fmt(pb["Parquet + blob file"]["8"] / pb["Parquet + blob file"]["1"], 1),
    "PqParEight": fmt(pb["Parquet (snappy, rg=1024)"]["8"], 0),
})

# ---------------------------------------------------------------- object-store model
# Per epoch of random access over the layout corpus, from the instrumented counts (W2).
GET_MS, CONC, GBPS = 30.0, 16, 1.0
ml = []
for r in rows:
    if not r["take_req_per_batch"]:
        continue
    nb = L["n"] / L["batch"]
    req = r["take_req_per_batch"] * nb
    gb = r["take_MB_per_batch"] * nb / 1e3
    t_batch = (r["take_req_per_batch"] / CONC) * GET_MS / 1e3 + r["take_MB_per_batch"] / 1e3 / GBPS
    ml.append(f"{short(r['layout'])} & {fmt(req / 1e3, 1)} & {fmt(gb, 1)} & {fmt(t_batch * 1e3, 0)} & {fmt(L['batch'] / t_batch, 0)} \\\\")
(OUT / "s3model.tex").write_text("\n".join(ml) + "\n")

# ---------------------------------------------------------------- compaction
C = json.loads((R / "compact.json").read_text())
cl = []
for r in C:
    t0, t60, tc = r["trace"][0], r["trace"][-2], r["trace"][-1]
    name = "Delta Lake" if r["kind"] == "delta" else "Lance"
    import statistics
    cl.append(" & ".join([name, fmt(statistics.median(r["commit_s"]) * 1e3, 1),
                          f"{t0['files']}/{t60['files']}/{tc['files']}",
                          f"{fmt(t0['bytes'] / 1e6, 0)}/{fmt(t60['bytes'] / 1e6, 0)}/{fmt(tc['bytes'] / 1e6, 0)}",
                          f"{fmt(t0['lookup_s'] * 1e3, 1)}/{fmt(t60['lookup_s'] * 1e3, 1)}/{fmt(tc['lookup_s'] * 1e3, 1)}",
                          fmt(r["compact_vacuum_s"], 2), "no" if not r["old_version_readable"] else "yes"]) + r" \\")
(OUT / "compact_table.tex").write_text("\n".join(cl) + "\n")
cd = {r["kind"]: r for r in C}
macros.update({
    "DeltaSmallGB": fmt(cd["delta"]["trace"][-2]["bytes"] / 1e9, 2),
    "DeltaSmallAmp": fmt(cd["delta"]["trace"][-2]["bytes"] / cd["delta"]["trace"][-1]["bytes"], 0),
    "LanceSmallMB": fmt(cd["lance"]["trace"][-2]["bytes"] / 1e6, 0),
    "LanceFiles": str(cd["lance"]["trace"][-2]["files"]),
    "LanceLookupSlow": fmt(cd["lance"]["trace"][-2]["lookup_s"] / cd["lance"]["trace"][0]["lookup_s"], 1),
})
(OUT / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
print("extended macros", len(macros))

# ---------------------------------------------------------------- end-to-end loader
E = json.loads((R / "e2e.json").read_text())
el = [f"Decode only (in memory) & {fmt(E['decode_only']['1'], 0)} & {fmt(E['decode_only']['8'], 0)} & -- \\\\", r"\midrule"]
for r in E["rows"]:
    tk = {x["layout"]: x["take_sps"] for x in rows}.get(r["layout"])
    el.append(f"{short(r['layout'])} & {fmt(r['1'], 0)} & {fmt(r['8'], 0)} & {fmt(tk, 0)} \\\\")
(OUT / "e2e_table.tex").write_text("\n".join(el) + "\n")
eb = {r["layout"]: r for r in E["rows"]}
macros.update({
    "EtoEDecodeOne": fmt(E["decode_only"]["1"], 0), "EtoEDecodeEight": fmt(E["decode_only"]["8"], 0),
    "EtoETar": fmt(eb["Tar shards (WebDataset)"]["8"], 0), "EtoELance": fmt(eb["Lance"]["8"], 0),
    "EtoEPq": fmt(eb["Parquet (snappy, rg=1024)"]["8"], 0), "EtoEBlob": fmt(eb["Parquet + blob file"]["8"], 0),
    "EtoEGap": fmt(eb["Tar shards (WebDataset)"]["8"] / eb["Parquet (snappy, rg=1024)"]["8"], 1),
})

# ---------------------------------------------------------------- workload-mix analysis
# Normalized time for one "day" of work under three mixes, from measured W1-W5 times.
mixes = {
    "Training-heavy": {"take": 20, "emb": 1, "meta": 1, "filter": 1, "scan": 0},
    "Curation-heavy": {"take": 2, "emb": 2, "meta": 20, "filter": 20, "scan": 1},
    "Embedding-heavy": {"take": 2, "emb": 20, "meta": 2, "filter": 2, "scan": 1},
}
cand = ["Files (per sample)", "Tar shards (WebDataset)", "Arrow IPC (mmap)", "Parquet (snappy, rg=1024)",
        "Parquet (snappy, rg=1024, sorted)", "Lance (sorted)", "Parquet + blob file"]
wl = []
cost = {}
for c in cand:
    r = {x["layout"]: x for x in rows}[c]
    cost[c] = {m: sum(w * r[k + "_s"] for k, w in mix.items()) for m, mix in mixes.items()}
best = {m: min(cost[c][m] for c in cand) for m in mixes}
for c in cand:
    wl.append(short(c) + " & " + " & ".join(
        (r"\textbf{1.0}" if abs(cost[c][m] - best[m]) < 1e-12 else fmt(cost[c][m] / best[m], 1)) for m in mixes) + r" \\")
(OUT / "mix_table.tex").write_text("\n".join(wl) + "\n")
(OUT / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
print("final macros", len(macros))

# ---------------------------------------------------------------- blob-size sensitivity
bl, amp = [], {}
for size in (128, 224, 448):
    B = json.loads((R / f"blob_{size}.json").read_text())
    lat = B["logical_bytes"][-1]
    for s in B["strategies"]:
        amp.setdefault(s["name"], []).append((size, s["bytes"][-1] / lat, lat / B["n0"] / 1e3))
for name, pts in amp.items():
    bl.append(r"\addplot+[mark size=1.5pt] coordinates {" + " ".join(f"({sz},{a:.2f})" for sz, a, _ in pts) + "};\n"
              + r"\addlegendentry{" + name + "}")
(OUT / "blob_amp.tex").write_text("\n".join(bl) + "\n")
macros.update({
    "BlobDeltaSmall": fmt(amp["Delta Lake"][0][1] / amp["Delta + blob CAS"][0][1], 1),
    "BlobDeltaLarge": fmt(amp["Delta Lake"][-1][1] / amp["Delta + blob CAS"][-1][1], 1),
    "BlobLanceLarge": fmt(amp["Lance"][-1][1], 2), "BlobDBLarge": fmt(amp["Delta + blob CAS"][-1][1], 2),
    "BlobDeltaRange": f"{amp['Delta Lake'][0][1]:.1f}--{amp['Delta Lake'][-1][1]:.1f}",
})
(OUT / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
print("blob macros", {k: macros[k] for k in macros if k.startswith("Blob")})

# ---------------------------------------------------------------- batch size + column shares
Bt = json.loads((R / "batch.json").read_text())
bp = []
for r in Bt["rows"]:
    bp.append(r"\addplot+[mark size=1.4pt] coordinates {" + " ".join(f"({b},{v:.0f})" for b, v in r["sps"].items()) + "};\n"
              + r"\addlegendentry{" + short(r["layout"]) + "}")
(OUT / "batch.tex").write_text("\n".join(bp) + "\n")
bb = {r["layout"]: r["sps"] for r in Bt["rows"]}
cols = Bt["columns"]
tot = sum(cols.values())
macros.update({
    "ShareImage": f"{100 * cols['image'] / tot:.1f}", "ShareEmb": f"{100 * cols['embedding'] / tot:.1f}",
    "ShareCaption": f"{100 * cols['caption'] / tot:.2f}",
    "ShareMeta": f"{100 * (tot - cols['image'] - cols['embedding'] - cols['caption']) / tot:.2f}",
    "ArrowBOne": fmt(bb["Arrow IPC (mmap)"]["1"], 0), "ArrowBBig": fmt(bb["Arrow IPC (mmap)"]["512"], 0),
    "PqBOne": fmt(bb["Parquet (snappy, rg=1024)"]["1"], 0), "PqBBig": fmt(bb["Parquet (snappy, rg=1024)"]["512"], 0),
    "LanceBOne": fmt(bb["Lance"]["1"], 0), "TarBOne": fmt(bb["Tar shards (WebDataset)"]["1"], 0),
})
(OUT / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
print("ok", macros["ShareImage"], macros["ShareEmb"])
