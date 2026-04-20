#!/usr/bin/env bash
# Run multiple phase 2 passes with different seeds.
# Usage: ./run_phase2.sh [--model MODEL] [--api-base API_BASE] [--count COUNT] [--num-runs NUM_RUNS] [--output-dir OUTPUT_DIR]
# Defaults to the Gemma model on spark-farm.

# Default values
# Switch to 31B for higher quality: MODEL="nvidia/Gemma-4-31B-IT-NVFP4"
MODEL="google/gemma-4-E4B-IT"
API_BASE="http://spark-farm:8000/v1"
COUNT=500
NUM_RUNS=5
OUTPUT_DIR=""  # empty = derive from concepts.json version at runtime

# Parse named parameters
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --api-base)
      API_BASE="$2"
      shift 2
      ;;
    --count)
      COUNT="$2"
      shift 2
      ;;
    --num-runs)
      NUM_RUNS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run_phase2.sh [--model MODEL] [--api-base API_BASE] [--count COUNT] [--num-runs NUM_RUNS] [--output-dir OUTPUT_DIR]"
      exit 1
      ;;
  esac
done

SEEDS=()
for ((i=0; i<NUM_RUNS; i++)); do
  SEEDS+=($RANDOM)
done

echo "Model:    $MODEL"
echo "API:      $API_BASE"
echo "Count:    $COUNT per pass"
echo "Output:   ${OUTPUT_DIR:-"(derived from concepts version)"}"
echo "Seeds:    ${SEEDS[*]}"
echo "---"

for SEED in "${SEEDS[@]}"; do
  OUTPUT_ARGS=()
  [[ -n "$OUTPUT_DIR" ]] && OUTPUT_ARGS=(--output-dir "$OUTPUT_DIR")
  uv run python -m generation.generate \
    --phase 2 \
    --model "$MODEL" \
    --api-base "$API_BASE" \
    --count "$COUNT" \
    --batch-size 128 \
    --seed "$SEED" \
    "${OUTPUT_ARGS[@]}"
  echo "Done seed=$SEED"
  echo "---"
done
wait

echo "All passes complete."
