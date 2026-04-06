#!/usr/bin/env bash
# Run multiple phase 1 passes with different seeds.
# Usage: ./run_phase1.sh [--model MODEL] [--api-base API_BASE] [--count COUNT] [--num-runs NUM_RUNS] [--output-dir OUTPUT_DIR]
# Defaults to the Gemma model on spark-farm.

# Default values
MODEL="nvidia/Gemma-4-31B-IT-NVFP4"
API_BASE="http://spark-farm:8000/v1"
COUNT=1000
NUM_RUNS=5
OUTPUT_DIR="$HOME/data/generation/phase1"

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
      echo "Usage: ./run_phase1.sh [--model MODEL] [--api-base API_BASE] [--count COUNT] [--num-runs NUM_RUNS] [--output-dir OUTPUT_DIR]"
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
echo "Output:   $OUTPUT_DIR"
echo "Seeds:    ${SEEDS[*]}"
echo "---"

for SEED in "${SEEDS[@]}"; do
  uv run python -m generation.generate \
    --model "$MODEL" \
    --api-base "$API_BASE" \
    --count "$COUNT" \
    --batch-size 256 \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"
  echo "Done seed=$SEED"
  echo "---"
done
wait

echo "All passes complete."
