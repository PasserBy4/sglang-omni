#!/bin/bash
# Experiment 2: Validate Small-Sample Representativeness via Codec Degradation
#
# Runs WER at multiple sample sizes against the degraded model (port 8000).
# Compare the results with Experiment 1 (normal model) to check whether
# a small sample detects the same WER increase as the full dataset.
#
# Prerequisites:
#   1. pip install -e ".[s2pro,dev]"
#   2. python -m benchmarks.dataset.prepare --dataset seedtts
#   3. python scripts/degrade_codec.py --model-path fishaudio/s2-pro \
#          --output-dir /tmp/s2pro-degraded
#
# Usage:
#   Terminal 1 — degraded model server (port 8000):
#     python -m sglang_omni.cli.cli serve \
#       --model-path /tmp/s2pro-degraded \
#       --config examples/configs/s2pro_tts.yaml \
#       --port 8000
#
#   Terminal 2 — run this script:
#     bash scripts/run_experiment2.sh

set -e

PORT=8000
SEED=42
MODEL="/tmp/s2pro-degraded"
SIZES=(6 100 300)
META="seedtts_testset/en/meta.lst"
BASE="results/representativeness/s2pro_vc_degraded"
ASR_DEVICE="cuda:1"

# ------------------------------------------------------------------
# Preflight checks
# ------------------------------------------------------------------
if [ ! -f "$META" ]; then
  echo "[ERROR] Dataset not found: $META"
  echo "Run: python -m benchmarks.dataset.prepare --dataset seedtts"
  exit 1
fi

if [ ! -d "$MODEL" ]; then
  echo "[ERROR] Degraded model not found: $MODEL"
  echo "Run: python scripts/degrade_codec.py --model-path fishaudio/s2-pro --output-dir $MODEL"
  exit 1
fi

echo "=== Experiment 2: Degraded Model (INT4 Semantic Codebook) ==="
echo "Sample sizes: ${SIZES[*]}"
echo "Model: $MODEL"
echo ""

for n in "${SIZES[@]}"; do
  echo "=============================="
  echo "  Sample size: $n"
  echo "=============================="

  python -m benchmarks.eval.voice_clone_s2pro \
    --model "$MODEL" --port $PORT --device $ASR_DEVICE \
    --meta "$META" --lang en --max-samples $n \
    --shuffle --sample-seed $SEED \
    --output-dir "${BASE}/wer/en/shuf_${n}"

  echo ""
done

echo "=== All runs completed ==="
echo "Results saved to: ${BASE}/wer/"
echo ""
echo "Compare with normal model (Experiment 1):"
echo "  Normal:   results/representativeness/s2pro_vc/wer/en/shuf_{6,100,300}/"
echo "  Degraded: ${BASE}/wer/en/shuf_{6,100,300}/"
