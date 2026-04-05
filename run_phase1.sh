#!/usr/bin/env bash
# Run multiple phase 1 passes with different seeds.
# Usage: ./run_phase1.sh [model] [api-base] [count-per-pass]
# Defaults to the Gemma model on spark-farm.

MODEL="${1:-nvidia/Gemma-4-31B-IT-NVFP4}"
API_BASE="${2:-http://spark-farm:8000/v1}"
COUNT="${3:-1000}"
SEEDS=(1111 2222 3333 4444 5555 6666 7777 8888 9999 1010 2020 3030 4040 5050 6060)

echo "Model:    $MODEL"
echo "API:      $API_BASE"
echo "Count:    $COUNT per pass"
echo "Seeds:    ${SEEDS[*]}"
echo "---"

for SEED in "${SEEDS[@]}"; do
  echo "Starting pass seed=$SEED ..."
  uv run python -m generation.generate \
    --model "$MODEL" \
    --api-base "$API_BASE" \
    --count "$COUNT" \
    --batch-size 50 \
    --seed "$SEED" \
    --output-dir ~/data/generation/phase1/
  echo "Done seed=$SEED"
  echo "---"
done

echo "All passes complete."
