#!/usr/bin/env zsh

ROOT="/Users/albertbz/Library/CloudStorage/OneDrive-Personal/Dokumenter/University/7. semester/Visual Computing/Assignment2"
mkdir -p "$ROOT/bench-results" "$ROOT/bench-results/plots"

for BUILD in build-debug build-release; do
  BIN="$ROOT/$BUILD/Webcam"
  if [ ! -x "$BIN" ]; then
    echo "Missing binary: $BIN"
    exit 1
  fi
  for FILTER in edge pixelate; do
    for BACKEND in cpu gpu; do
      if [ "$BACKEND" = cpu ]; then
        TRANSFORMS_LIST="off cpu"
      else
        TRANSFORMS_LIST="off gpu"
      fi
      for TRANSFORMS in $TRANSFORMS_LIST; do
        for RES in 1024x768 2048x1536; do
          OUT="bench-results/bench_${FILTER}_${BACKEND}_${TRANSFORMS}_${RES}_${BUILD}.csv"
          echo "Running: $BUILD $FILTER $BACKEND $TRANSFORMS $RES -> $OUT"
          (cd "$ROOT/Webcam" && "$BIN" --benchmark --out "../$OUT" \
            --filter "$FILTER" --backend "$BACKEND" --transforms "$TRANSFORMS" \
            --resolution "$RES" --frames 120 \
            --translateU 0.12 --translateV -0.08 --scale 1.25 --rotation 15)
          sleep 0.5
        done
      done
    done
  done
done

echo "ALL DONE"
