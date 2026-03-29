#!/bin/bash
# Experiment 1: S2-Pro Voice Clone Small-Sample Representativeness (EN, shuffle only)
#
# Prerequisites:
#   1. pip install -e ".[s2pro,dev]"
#   2. python -m benchmarks.dataset.prepare --dataset seedtts
#
# Usage:
#   Terminal 1 — start server:
#     python -m sglang_omni.cli.cli serve \
#       --model-path fishaudio/s2-pro \
#       --config examples/configs/s2pro_tts.yaml \
#       --port 8000
#
#   Terminal 2 — run this script:
#     bash scripts/run_experiment1.sh

set -e

PORT=8000
SEED=42
MODEL="fishaudio/s2-pro"
SIZES=(6 100 300)
META="seedtts_testset/en/meta.lst"
BASE="results/representativeness/s2pro_vc"
ASR_DEVICE="cuda:1"

# ------------------------------------------------------------------
# Preflight checks
# ------------------------------------------------------------------
if [ ! -f "$META" ]; then
  echo "[ERROR] Dataset not found: $META"
  echo "Run: python -m benchmarks.dataset.prepare --dataset seedtts"
  exit 1
fi

echo "=== Experiment 1: S2-Pro Voice Clone Small-Sample Representativeness ==="
echo "Samples sizes: ${SIZES[*]}"
echo "Dataset: $META"
echo ""

for n in "${SIZES[@]}"; do
  echo "=============================="
  echo "  Sample size: $n"
  echo "=============================="

  # --- Performance: streaming ---
  echo "[1/2] Performance streaming (n=$n) ..."
  python -m benchmarks.eval.s2pro_tts_speed \
    --model $MODEL --port $PORT \
    --testset "$META" --max-samples $n --stream \
    --shuffle --sample-seed $SEED \
    --output-dir "${BASE}/perf_stream/en/shuf_${n}"

  # --- WER accuracy ---
  echo "[2/2] WER accuracy (n=$n) ..."
  python -m benchmarks.eval.voice_clone_s2pro \
    --model $MODEL --port $PORT --device $ASR_DEVICE \
    --meta "$META" --lang en --max-samples $n \
    --shuffle --sample-seed $SEED \
    --output-dir "${BASE}/wer/en/shuf_${n}"

  echo ""
done

echo "=== All 6 runs completed ==="
echo "Results saved to: ${BASE}/"
echo ""
echo "Directory layout:"
echo "  ${BASE}/perf_stream/en/shuf_{6,100,300}/speed_results.json"
echo "  ${BASE}/wer/en/shuf_{6,100,300}/wer_results.json"
