#!/usr/bin/env python3
"""
Aggregate plots: (A) mean frame time per FILTER (averaged across builds/backends/resolutions/transforms)
(B) mean frame time per BACKEND (CPU vs GPU) averaged across filters/resolutions/builds/transforms
Writes PNGs to bench-results/plots/
"""
import csv
import os
import statistics
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMMARY = os.path.join(ROOT, "bench-results", "summary.csv")
PLOTS_DIR = os.path.join(ROOT, "bench-results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(SUMMARY):
    print("summary.csv not found at", SUMMARY)
    raise SystemExit(1)

rows = []
with open(SUMMARY, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

# Normalize backend and filter names
for r in rows:
    r["filter"] = r["filter"].strip()
    r["backend"] = r["backend"].strip()

# Aggregate by filter
filter_vals = defaultdict(list)
for r in rows:
    try:
        m = float(r["mean_ms"])
    except Exception:
        continue
    filter_vals[r["filter"]].append(m)

filters = sorted(filter_vals.keys())
filter_means = [statistics.mean(filter_vals[f]) for f in filters]
filter_stds = [
    statistics.stdev(filter_vals[f]) if len(filter_vals[f]) > 1 else 0.0
    for f in filters
]

# Plot filter means
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    filters,
    filter_means,
    yerr=filter_stds,
    capsize=6,
    color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"],
)
ax.set_ylabel("Mean frame time (ms)")
ax.set_title("Mean frame time by FILTER (averaged across configs)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
outf = os.path.join(PLOTS_DIR, "plot_filters_overall.png")
fig.savefig(outf)
plt.close(fig)
print("Wrote", outf)

# Aggregate by backend (cpu vs gpu)
backend_vals = defaultdict(list)
for r in rows:
    b = r["backend"].lower().strip()
    # normalize known tokens
    if "cpu" in b:
        key = "cpu"
    elif "gpu" in b:
        key = "gpu"
    else:
        key = b
    try:
        m = float(r["mean_ms"])
    except Exception:
        continue
    backend_vals[key].append(m)

backends = sorted(backend_vals.keys())
backend_means = [statistics.mean(backend_vals[b]) for b in backends]
backend_stds = [
    statistics.stdev(backend_vals[b]) if len(backend_vals[b]) > 1 else 0.0
    for b in backends
]

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(
    backends, backend_means, yerr=backend_stds, capsize=8, color=["#E24A33", "#348ABD"]
)
ax.set_ylabel("Mean frame time (ms)")
ax.set_title("Mean frame time by BACKEND (averaged across configs)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
outf = os.path.join(PLOTS_DIR, "plot_backends_overall.png")
fig.savefig(outf)
plt.close(fig)
print("Wrote", outf)

# Export aggregated numbers to CSV for inclusion in reports
import csv

outcsv = os.path.join(ROOT, "bench-results", "aggregated_stats.csv")
with open(outcsv, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["level", "group", "mean_ms", "std_ms", "count"])
    # filters
    for f in filters:
        vals = filter_vals[f]
        writer.writerow(
            [
                "filter",
                f,
                f"{statistics.mean(vals):.6f}",
                f"{statistics.stdev(vals) if len(vals)>1 else 0.0:.6f}",
                len(vals),
            ]
        )
    # backends
    for b in backends:
        vals = backend_vals[b]
        writer.writerow(
            [
                "backend",
                b,
                f"{statistics.mean(vals):.6f}",
                f"{statistics.stdev(vals) if len(vals)>1 else 0.0:.6f}",
                len(vals),
            ]
        )
    # per filter+backend
    # build a mapping
    per_fb = defaultdict(list)
    for r in rows:
        try:
            m = float(r["mean_ms"])
        except Exception:
            continue
        filt = r["filter"]
        b = r["backend"].lower().strip()
        key = f"{filt}|{b}"
        per_fb[key].append(m)
    for k, vals in sorted(per_fb.items()):
        writer.writerow(
            [
                "filter|backend",
                k,
                f"{statistics.mean(vals):.6f}",
                f"{statistics.stdev(vals) if len(vals)>1 else 0.0:.6f}",
                len(vals),
            ]
        )

print("Wrote aggregated CSV", outcsv)
