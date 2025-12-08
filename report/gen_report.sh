#!/bin/bash
#
# Generate all experiment data and visualizations for the expert-lru report.
#
# Usage:
#   ./report/gen_report.sh <model_path>
#   CACHE_SIZES=64,80,96 ./report/gen_report.sh <model_path>
#
# Outputs:
#   - img/<model_name>/*.png (visualizations)
#   - /tmp/data/report/ (raw data)
#

set -e

# Determine script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL=${1:?Usage: $0 <model_path>}
MODEL_NAME=$(basename "$MODEL")
DATA_DIR=/tmp/data/report
IMG_DIR=$PROJECT_DIR/img/$MODEL_NAME
IFS=',' read -ra CACHE_SIZES <<< "${CACHE_SIZES:-80,96}"
N_TOKENS=512

PROMPT_POETRY="Write 5 poems about the ocean in different styles"
PROMPT_CODE="Write a Python function to sort a list"

# Use the largest cache size for cross-prompt experiments
MAX_CACHE=${CACHE_SIZES[${#CACHE_SIZES[@]}-1]}

echo "============================================================"
echo "Report Generation for: $MODEL_NAME"
echo "Cache sizes: ${CACHE_SIZES[*]}"
echo "Max cache (for cross-prompt): $MAX_CACHE"
echo "Output: $IMG_DIR"
echo "============================================================"

# Cleanup and setup
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR" "$IMG_DIR"

echo ""
echo "============================================================"
echo "Phase 1: Cold runs"
echo "============================================================"

# Cold runs for poetry at all cache sizes (for hitrate charts)
for cache_size in "${CACHE_SIZES[@]}"; do
    echo ""
    echo "=== Poetry prompt - Cold (cache=$cache_size) ==="
    python "$PROJECT_DIR/scripts/generate.py" -m "$MODEL" -c "$cache_size" \
        -p "$PROMPT_POETRY" -n $N_TOKENS \
        -A "$DATA_DIR/cold_poetry_c${cache_size}"
done

# Cold run for code at max cache size (for cross-prompt comparison)
echo ""
echo "=== Code prompt - Cold (cache=$MAX_CACHE) ==="
python "$PROJECT_DIR/scripts/generate.py" -m "$MODEL" -c "$MAX_CACHE" \
    -p "$PROMPT_CODE" -n $N_TOKENS \
    -A "$DATA_DIR/cold_code_c${MAX_CACHE}"

echo ""
echo "============================================================"
echo "Phase 2: Warmup runs"
echo "============================================================"

# Random and warm runs for poetry at all cache sizes
for cache_size in "${CACHE_SIZES[@]}"; do
    echo ""
    echo "=== Poetry prompt - Random warmup (cache=$cache_size) ==="
    python "$PROJECT_DIR/scripts/generate.py" -m "$MODEL" -c "$cache_size" \
        -p "$PROMPT_POETRY" -n $N_TOKENS \
        -R 42 \
        -A "$DATA_DIR/random_poetry_c${cache_size}"

    echo ""
    echo "=== Poetry prompt - Warm from own log (cache=$cache_size) ==="
    python "$PROJECT_DIR/scripts/generate.py" -m "$MODEL" -c "$cache_size" \
        -p "$PROMPT_POETRY" -n $N_TOKENS \
        -W "$DATA_DIR/cold_poetry_c${cache_size}" \
        -A "$DATA_DIR/warm_poetry_c${cache_size}"
done

# Merge access logs for cross-prompt warmup
echo ""
echo "=== Merging access logs ==="
python "$SCRIPT_DIR/merge_access_logs.py" \
    "$DATA_DIR/cold_poetry_c${MAX_CACHE}" "$DATA_DIR/cold_code_c${MAX_CACHE}" \
    -o "$DATA_DIR/merged" -v

# Cross-prompt warmup variants (at max cache size)
echo ""
echo "=== Poetry - Warm from code ==="
python "$PROJECT_DIR/scripts/generate.py" -m "$MODEL" -c "$MAX_CACHE" \
    -p "$PROMPT_POETRY" -n $N_TOKENS \
    -W "$DATA_DIR/cold_code_c${MAX_CACHE}" \
    -A "$DATA_DIR/warm_poetry_from_code"

echo ""
echo "=== Poetry - Warm from merged ==="
python "$PROJECT_DIR/scripts/generate.py" -m "$MODEL" -c "$MAX_CACHE" \
    -p "$PROMPT_POETRY" -n $N_TOKENS \
    -W "$DATA_DIR/merged" \
    -A "$DATA_DIR/warm_poetry_from_merged"

echo ""
echo "=== Code - Warm from own log ==="
python "$PROJECT_DIR/scripts/generate.py" -m "$MODEL" -c "$MAX_CACHE" \
    -p "$PROMPT_CODE" -n $N_TOKENS \
    -W "$DATA_DIR/cold_code_c${MAX_CACHE}" \
    -A "$DATA_DIR/warm_code_from_code"

echo ""
echo "=== Code - Warm from poetry ==="
python "$PROJECT_DIR/scripts/generate.py" -m "$MODEL" -c "$MAX_CACHE" \
    -p "$PROMPT_CODE" -n $N_TOKENS \
    -W "$DATA_DIR/cold_poetry_c${MAX_CACHE}" \
    -A "$DATA_DIR/warm_code_from_poetry"

echo ""
echo "=== Code - Warm from merged ==="
python "$PROJECT_DIR/scripts/generate.py" -m "$MODEL" -c "$MAX_CACHE" \
    -p "$PROMPT_CODE" -n $N_TOKENS \
    -W "$DATA_DIR/merged" \
    -A "$DATA_DIR/warm_code_from_merged"

echo ""
echo "=== Code - Random warmup ==="
python "$PROJECT_DIR/scripts/generate.py" -m "$MODEL" -c "$MAX_CACHE" \
    -p "$PROMPT_CODE" -n $N_TOKENS \
    -R 42 \
    -A "$DATA_DIR/random_code_c${MAX_CACHE}"

echo ""
echo "============================================================"
echo "Phase 3: Visualizations"
echo "============================================================"

# Build directory arrays for visualization
cold_dirs=()
warm_dirs=()
for cache_size in "${CACHE_SIZES[@]}"; do
    cold_dirs+=("$DATA_DIR/cold_poetry_c${cache_size}")
    warm_dirs+=("$DATA_DIR/warm_poetry_c${cache_size}")
done

echo ""
echo "=== Cold hitrate charts ==="
python "$SCRIPT_DIR/visualize_hitrate.py" "${cold_dirs[@]}" -o "$IMG_DIR" -p cold

echo ""
echo "=== Warm hitrate charts ==="
python "$SCRIPT_DIR/visualize_hitrate.py" "${warm_dirs[@]}" -o "$IMG_DIR" -p warm

echo ""
echo "=== Expert overlap chart ==="
python "$SCRIPT_DIR/compare_experts.py" \
    "$DATA_DIR/cold_poetry_c${MAX_CACHE}" "$DATA_DIR/cold_code_c${MAX_CACHE}" \
    -o "$IMG_DIR"

echo ""
echo "=== Warmup comparison chart ==="
python "$SCRIPT_DIR/visualize_warmup_comparison.py" \
    --poetry-cold "$DATA_DIR/cold_poetry_c${MAX_CACHE}" \
    --poetry-random "$DATA_DIR/random_poetry_c${MAX_CACHE}" \
    --poetry-same "$DATA_DIR/warm_poetry_c${MAX_CACHE}" \
    --poetry-other "$DATA_DIR/warm_poetry_from_code" \
    --poetry-merged "$DATA_DIR/warm_poetry_from_merged" \
    --code-cold "$DATA_DIR/cold_code_c${MAX_CACHE}" \
    --code-random "$DATA_DIR/random_code_c${MAX_CACHE}" \
    --code-same "$DATA_DIR/warm_code_from_code" \
    --code-other "$DATA_DIR/warm_code_from_poetry" \
    --code-merged "$DATA_DIR/warm_code_from_merged" \
    -o "$IMG_DIR"

echo ""
echo "============================================================"
echo "Done! Images saved to: $IMG_DIR"
echo "============================================================"
