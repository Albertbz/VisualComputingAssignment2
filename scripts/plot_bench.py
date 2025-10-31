#!/usr/bin/env python3
"""
Simple benchmark plotting script for bench-results CSVs.
Generates summary CSV and PNG bar charts per (filter, build).

Usage: python3 scripts/plot_bench.py

Outputs:
 - bench-results/summary.csv
 - bench-results/plots/plot_<filter>_<build>.png
"""
import os
import glob
import csv
import statistics
from collections import defaultdict, OrderedDict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR = os.path.join(ROOT, "bench-results")
PLOTS_DIR = os.path.join(BENCH_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Find bench CSVs
pattern = os.path.join(BENCH_DIR, "bench_*.csv")
files = sorted(glob.glob(pattern))
if not files:
    print("No bench CSVs found in", BENCH_DIR)
    raise SystemExit(1)

# Parse files into a structured dict
# key: (build, filter, backend, res, trans) -> list of frame times
data = defaultdict(list)
meta = {}
for f in files:
    name = os.path.basename(f).replace(".csv", "")
    parts = name.split("_")
    # expect filenames like: bench_<filter>_<backend>_<transforms>_<resolution>_<build>
    # example: bench_edge_cpu_off_1024x768_build-debug.csv
    if len(parts) < 6:
        print("Skipping unexpected filename:", f)
        continue
    # map parts to variables in the order produced by the run scripts
    # parts: ['bench', filter, backend, transforms, resolution, build]
    _, filt, backend, trans, res, build = parts[:6]
    key = (build, filt, backend, res, trans)
    with open(f, "r") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        vals = []
        for row in reader:
            if not row:
                continue
            try:
                vals.append(float(row[0]))
            except Exception:
                pass
    if vals:
        data[key] = vals
        meta[key] = f

# Aggregate stats per key
summary_rows = []
for key, vals in sorted(data.items()):
    build, filt, backend, res, trans = key
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    frames = len(vals)
    summary_rows.append(
        (build, filt, backend, res, trans, frames, mean, stdev, meta[key])
    )

# Write summary CSV
summary_csv = os.path.join(BENCH_DIR, "summary.csv")
with open(summary_csv, "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(
        [
            "build",
            "filter",
            "backend",
            "resolution",
            "transforms",
            "frames",
            "mean_ms",
            "std_ms",
            "csv_path",
        ]
    )
    for row in summary_rows:
        writer.writerow(row)
print("Wrote", summary_csv)

# Organize for plotting
# For each (filter, build) create a bar chart where x is resolution and bars are backend+trans combos
from itertools import product

# derive filters/resolutions/combo keys directly from summary_rows

# Build nested dict: plot_data[filter][build][(backend,trans)][res] = mean
plot_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
resolutions = set()
legend_labels = set()
for row in summary_rows:
    build, filt, backend, res, trans, frames, mean, stdev, path = row
    plot_data[filt][build][(backend, trans)][res] = mean
    resolutions.add(res)
    legend_labels.add((backend, trans))
resolutions = sorted(
    resolutions, key=lambda s: (int(s.split("x")[0]), int(s.split("x")[1]))
)
legend_labels = sorted(legend_labels)

# Plotting
for filt, builds in plot_data.items():
    for build, combos_dict in builds.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(resolutions))
        width = 0.8 / max(1, len(combos_dict))
        offsets = []
        combo_keys = sorted(combos_dict.keys())
        start = -0.4 + width / 2
        for i, combo in enumerate(combo_keys):
            means = [combos_dict[combo].get(res, None) for res in resolutions]
            xpos = [xi + start + i * width for xi in x]
            # replace None with 0 and mark missing with hatch (but better to skip)
            barlist = ax.bar(
                xpos,
                [m if m is not None else 0 for m in means],
                width=width,
                label=f"{combo[0]}|{combo[1]}",
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(resolutions, rotation=45)
        ax.set_ylabel("Mean frame time (ms)")
        ax.set_title(f"Filter={filt}  Build={build}")
        ax.legend(
            title="backend|transforms", bbox_to_anchor=(1.02, 1), loc="upper left"
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        outpng = os.path.join(PLOTS_DIR, f"plot_{filt}_{build}.png")
        fig.savefig(outpng)
        plt.close(fig)
        print("Wrote", outpng)

print("Done. Plots are in", PLOTS_DIR)
