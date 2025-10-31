#!/usr/bin/env python3
import glob, csv, statistics, os

ROOT = "."
pattern = "bench-results/*_detailed_1024x768_build-release.csv.detailed.csv"
files = sorted(glob.glob(pattern))
if not files:
    print("No detailed files found for pattern", pattern)
    raise SystemExit(1)
print("Found", len(files), "detailed files")
rows_all = {}
for f in files:
    name = os.path.basename(f)
    key = name.replace("_detailed_1024x768_build-release.csv.detailed.csv", "")
    vals = {
        "total": [],
        "capture": [],
        "process": [],
        "transform": [],
        "upload": [],
        "draw": [],
    }
    with open(f) as fh:
        r = csv.DictReader(fh)
        for row in r:
            vals["total"].append(float(row["total_ms"]))
            vals["capture"].append(float(row["capture_ms"]))
            vals["process"].append(float(row["process_ms"]))
            vals["transform"].append(float(row["transform_ms"]))
            vals["upload"].append(float(row["upload_ms"]))
            vals["draw"].append(float(row["draw_ms"]))
    rows_all[key] = vals

print("\nPer-file averages (ms):")
print("file,total,capture,process,transform,upload,draw")
for k, v in rows_all.items():
    print(
        k,
        ",",
        ",".join(
            f"{statistics.mean(v[x]):.3f}"
            for x in ["total", "capture", "process", "transform", "upload", "draw"]
        ),
    )

# aggregate by filter/backend from key
print("\nAggregated by filter+backend:")
for k, v in rows_all.items():
    parts = k.split("_")
    if len(parts) >= 2:
        filt = parts[0]
        backend = parts[1]
    else:
        filt = k
        backend = ""
    print(
        f"{filt:8} {backend:4} -> total={statistics.mean(v['total']):.2f} ms, capture={statistics.mean(v['capture']):.2f} ms, process={statistics.mean(v['process']):.2f} ms, upload={statistics.mean(v['upload']):.2f} ms, draw={statistics.mean(v['draw']):.2f} ms"
    )
