#!/usr/bin/env python3
import csv
from collections import defaultdict

ROOT = "."
fn = "bench-results/summary.csv"
rows = []
with open(fn) as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

# normalize transforms field
for r in rows:
    t = r["transforms"].lower()
    if "off" in t:
        r["transforms_norm"] = "off"
    elif "cpu" in t:
        r["transforms_norm"] = "cpu"
    elif "gpu" in t:
        r["transforms_norm"] = "gpu"
    else:
        r["transforms_norm"] = t

# Build nested dict: stats[filter][build][resolution][backend] = mean_ms
stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for r in rows:
    filt = r["filter"]
    build = r["build"]
    res = r["resolution"]
    backend = r["backend"]
    mean = float(r["mean_ms"])
    stats[filt][build][res][backend] = mean

# Print comparisons per filter/build/resolution
print("Comparison CPU vs GPU (mean frame time ms)")
print(
    "Filter | Build | Resolution | CPU ms | GPU ms | Delta ms | % change (GPU vs CPU)"
)
for filt in sorted(stats.keys()):
    for build in sorted(stats[filt].keys()):
        for res in sorted(
            stats[filt][build].keys(),
            key=lambda s: (int(s.split("x")[0]), int(s.split("x")[1])),
        ):
            cpu = stats[filt][build][res].get("cpu", None)
            gpu = stats[filt][build][res].get("gpu", None)
            if cpu is None or gpu is None:
                continue
            delta = gpu - cpu
            pct = ((gpu - cpu) / cpu * 100.0) if cpu != 0 else 0.0
            print(
                f"{filt:6} | {build:11} | {res:9} | {cpu:6.2f} | {gpu:6.2f} | {delta:7.2f} | {pct:7.2f}%"
            )

# Resolution effect: compare 1024x768 vs 2048x1536 averaged across backend
print("\nResolution effect (mean frame time ms)")
print(
    "Filter | Build | Backend | 1024x768 ms | 2048x1536 ms | Delta ms | % change (2048 vs 1024)"
)
for filt in sorted(stats.keys()):
    for build in sorted(stats[filt].keys()):
        for backend in ["cpu", "gpu"]:
            m1024 = stats[filt][build].get("1024x768", {}).get(backend, None)
            m2048 = stats[filt][build].get("2048x1536", {}).get(backend, None)
            if m1024 is None or m2048 is None:
                continue
            delta = m2048 - m1024
            pct = (delta / m1024 * 100.0) if m1024 != 0 else 0.0
            print(
                f"{filt:6} | {build:11} | {backend:6} | {m1024:10.2f} | {m2048:12.2f} | {delta:8.2f} | {pct:7.2f}%"
            )

# Debug vs Release: compare mean across resolutions and backends per filter
print("\nBuild effect (Debug vs Release) averaged across res/backend")
print(
    "Filter | Debug mean ms | Release mean ms | Delta ms | % change (Release vs Debug)"
)
for filt in sorted(stats.keys()):
    debug_vals = []
    release_vals = []
    for res in stats[filt].get("build-debug", {}):
        for bk in stats[filt]["build-debug"][res]:
            debug_vals.append(stats[filt]["build-debug"][res][bk])
    for res in stats[filt].get("build-release", {}):
        for bk in stats[filt]["build-release"][res]:
            release_vals.append(stats[filt]["build-release"][res][bk])
    if not debug_vals or not release_vals:
        continue
    d_mean = sum(debug_vals) / len(debug_vals)
    r_mean = sum(release_vals) / len(release_vals)
    delta = r_mean - d_mean
    pct = (delta / d_mean * 100.0) if d_mean != 0 else 0.0
    print(f"{filt:6} | {d_mean:12.2f} | {r_mean:15.2f} | {delta:8.2f} | {pct:7.2f}%")
