#!/usr/bin/env zsh

# Full benchmark matrix runner
# Runs all valid combinations: filters x backends x transforms (consistent) x resolutions x builds
# Writes CSV outputs to bench-results/

ROOT="/Users/albertbz/Library/CloudStorage/OneDrive-Personal/Dokumenter/University/7. semester/Visual Computing/Assignment2"
mkdir -p "$ROOT/bench-results" "$ROOT/bench-results/plots"

# Frames per run (adjust if you want longer/shorter runs)
FRAMES=120

# Preset transforms to apply when transforms are enabled
T_U=0.12
T_V=-0.08
SCALE=1.25
ROT=15

for BUILD in build-debug build-release; do
  BIN="$ROOT/$BUILD/Webcam"
  if [ ! -x "$BIN" ]; then
    echo "Missing binary: $BIN"
    exit 1
  fi

  for FILTER in none gray edge pixelate; do
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
          # execute from Webcam/ so shader relative paths resolve correctly
          (cd "$ROOT/Webcam" && "$BIN" --benchmark --out "../$OUT" \
            --filter "$FILTER" --backend "$BACKEND" --transforms "$TRANSFORMS" \
            --resolution "$RES" --frames "$FRAMES" \
            --translateU $T_U --translateV $T_V --scale $SCALE --rotation $ROT)
          # brief pause between runs to let system settle
          sleep 0.5
        done
      done
    done
  done
done

echo "FULL BENCHMARK COMPLETE"
