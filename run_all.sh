#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-full_run}"
SEARCH_MODE="${SEARCH_MODE:-grid}"
COARSE_SEARCH_EPOCHS="${COARSE_SEARCH_EPOCHS:-10}"
COARSE_SEARCH_MAX_TRIALS="${COARSE_SEARCH_MAX_TRIALS:-36}"
COARSE_SEARCH_NAME="${COARSE_SEARCH_NAME:-${RUN_NAME}_coarse_search}"
FINE_SEARCH_EPOCHS="${FINE_SEARCH_EPOCHS:-15}"
FINE_SEARCH_MAX_TRIALS="${FINE_SEARCH_MAX_TRIALS:-18}"
FINE_SEARCH_NAME="${FINE_SEARCH_NAME:-${RUN_NAME}_fine_search}"
AUTO_CLEAN="${AUTO_CLEAN:-1}"
TRAIN_FROM_FINE_SEARCH_BEST="${TRAIN_FROM_FINE_SEARCH_BEST:-1}"

COARSE_LEARNING_RATES="${COARSE_LEARNING_RATES:-0.03,0.01,0.003}"
COARSE_HIDDEN_DIMS="${COARSE_HIDDEN_DIMS:-128,256,512}"
COARSE_WEIGHT_DECAYS="${COARSE_WEIGHT_DECAYS:-0,1e-4}"
COARSE_ACTIVATIONS="${COARSE_ACTIVATIONS:-relu,tanh,sigmoid}"
SEARCH_TOP_K="${SEARCH_TOP_K:-10}"
FINE_LEARNING_RATES="${FINE_LEARNING_RATES:-}"
FINE_HIDDEN_DIMS="${FINE_HIDDEN_DIMS:-}"
FINE_WEIGHT_DECAYS="${FINE_WEIGHT_DECAYS:-}"
FINE_ACTIVATIONS="${FINE_ACTIVATIONS:-}"

IMAGE_SIZE="${IMAGE_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-30}"
HIDDEN_DIM1="${HIDDEN_DIM1:-256}"
HIDDEN_DIM2="${HIDDEN_DIM2:-128}"
ACTIVATION="${ACTIVATION:-relu}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
LR_DECAY="${LR_DECAY:-0.95}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
LOG_DIR="results/logs"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"

mkdir -p results/curves results/weights results/errors results/search results/checkpoints "${LOG_DIR}"

cleanup_old_results() {
  echo "Cleaning old artifacts for run name: ${RUN_NAME}"
  rm -f "results/checkpoints/${RUN_NAME}_best.npz"
  rm -f "results/checkpoints/${RUN_NAME}"_epoch_*.npz
  rm -f "results/curves/${RUN_NAME}_history.json"
  rm -f "results/curves/${RUN_NAME}_curves.png"
  rm -f "results/weights/${RUN_NAME}_first_layer_weights.png"
  rm -f "results/errors/${RUN_NAME}_test_summary.json"
  rm -f "results/errors/${RUN_NAME}_test_confusion_matrix.png"
  rm -f "results/errors/${RUN_NAME}_test_misclassified.png"
  rm -f "results/errors/${RUN_NAME}_viz_confusion_matrix.png"
  rm -f "results/logs/${RUN_NAME}.log"

  for SEARCH_PREFIX in "${COARSE_SEARCH_NAME}" "${FINE_SEARCH_NAME}"; do
    echo "Cleaning old search artifacts for search name: ${SEARCH_PREFIX}"
    rm -f "results/search/${SEARCH_PREFIX}_results.csv"
    rm -f "results/search/${SEARCH_PREFIX}_summary.json"
    rm -f "results/search/${SEARCH_PREFIX}"_top*.csv
    rm -f "results/search/${SEARCH_PREFIX}_best_val_accuracy.png"
    rm -f "results/search/${SEARCH_PREFIX}"_groupby_*.csv
    rm -f "results/search/${SEARCH_PREFIX}"_groupby_*.png
    rm -f "results/search/${SEARCH_PREFIX}"_heatmap_*.png
    rm -f "results/checkpoints/${SEARCH_PREFIX}"_trial_*_best.npz
    rm -f "results/curves/${SEARCH_PREFIX}"_trial_*_history.json
    rm -f "results/curves/${SEARCH_PREFIX}"_trial_*_curves.png
    rm -f "results/weights/${SEARCH_PREFIX}"_trial_*_first_layer_weights.png
  done
}

if [[ "${AUTO_CLEAN}" == "1" ]]; then
  cleanup_old_results
fi

exec > >(tee -a "$LOG_PATH") 2>&1

echo "===== EuroSAT MLP Pipeline ====="
echo "Run name: ${RUN_NAME}"
echo "Coarse search name: ${COARSE_SEARCH_NAME}"
echo "Fine search name: ${FINE_SEARCH_NAME}"
echo "Log file: ${LOG_PATH}"
echo "Auto clean: ${AUTO_CLEAN}"
echo "Train from fine search best: ${TRAIN_FROM_FINE_SEARCH_BEST}"
echo "================================"

echo "[1/7] Installing dependencies if needed"
"$PYTHON_BIN" -m pip install -r requirements.txt

extract_best_params() {
  local summary_path="$1"
  "$PYTHON_BIN" - <<PY
import json
with open("${summary_path}", "r", encoding="utf-8") as f:
    data = json.load(f)
best = data["best_result"]
print(best["learning_rate"])
print(best["hidden_dim1"])
print(best["hidden_dim2"])
print(best["weight_decay"])
print(best["activation"])
PY
}

load_lines_into_array() {
  local array_name="$1"
  local command_output="$2"
  local old_ifs="${IFS}"
  IFS=$'\n'
  local lines=()
  for line in $command_output; do
    lines+=("$line")
  done
  IFS="${old_ifs}"
  eval "$array_name=()"
  local idx=0
  for line in "${lines[@]}"; do
    eval "$array_name[$idx]=\"\$line\""
    idx=$((idx + 1))
  done
}

generate_fine_space() {
  local summary_path="$1"
  "$PYTHON_BIN" - <<PY
import json

def uniq(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out

def format_float_list(values):
    cleaned = []
    for v in values:
        if abs(v) < 1e-12:
            cleaned.append("0")
        elif v >= 1e-3:
            cleaned.append(f"{v:.6f}".rstrip("0").rstrip("."))
        else:
            cleaned.append(f"{v:.0e}")
    return ",".join(cleaned)

with open("${summary_path}", "r", encoding="utf-8") as f:
    data = json.load(f)
best = data["best_result"]
lr = float(best["learning_rate"])
hidden = int(best["hidden_dim1"])
wd = float(best["weight_decay"])
activation = str(best["activation"])

lr_candidates = uniq(sorted([max(lr * 0.5, 1e-4), lr, min(lr * 2.0, 0.1)]))
hidden_candidates = uniq(sorted([max(64, hidden // 2), hidden, max(64, int(hidden * 1.5))]))
if wd == 0.0:
    wd_candidates = [0.0, 1e-5, 1e-4]
else:
    wd_candidates = uniq(sorted([max(wd * 0.5, 1e-6), wd, min(wd * 5.0, 1e-2)]))

print(format_float_list(lr_candidates))
print(",".join(str(v) for v in hidden_candidates))
print(format_float_list(wd_candidates))
print(activation)
PY
}

COARSE_SEARCH_SUMMARY_PATH="results/search/${COARSE_SEARCH_NAME}_summary.json"
echo "[2/7] Running coarse hyperparameter search"
"$PYTHON_BIN" code/search.py \
  --mode "$SEARCH_MODE" \
  --epochs "$COARSE_SEARCH_EPOCHS" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-trials "$COARSE_SEARCH_MAX_TRIALS" \
  --search-name "$COARSE_SEARCH_NAME" \
  --learning-rates "$COARSE_LEARNING_RATES" \
  --hidden-dims "$COARSE_HIDDEN_DIMS" \
  --weight-decays "$COARSE_WEIGHT_DECAYS" \
  --activations "$COARSE_ACTIVATIONS" \
  --top-k "$SEARCH_TOP_K"

COARSE_BEST_RAW="$(extract_best_params "${COARSE_SEARCH_SUMMARY_PATH}")"
load_lines_into_array COARSE_BEST_PARAMS "${COARSE_BEST_RAW}"
echo "Best coarse-search hyperparameters:"
echo "  learning_rate=${COARSE_BEST_PARAMS[0]}"
echo "  hidden_dim1=${COARSE_BEST_PARAMS[1]}"
echo "  hidden_dim2=${COARSE_BEST_PARAMS[2]}"
echo "  weight_decay=${COARSE_BEST_PARAMS[3]}"
echo "  activation=${COARSE_BEST_PARAMS[4]}"

if [[ -z "${FINE_LEARNING_RATES}" || -z "${FINE_HIDDEN_DIMS}" || -z "${FINE_WEIGHT_DECAYS}" || -z "${FINE_ACTIVATIONS}" ]]; then
  AUTO_FINE_SPACE_RAW="$(generate_fine_space "${COARSE_SEARCH_SUMMARY_PATH}")"
  load_lines_into_array AUTO_FINE_SPACE "${AUTO_FINE_SPACE_RAW}"
  [[ -z "${FINE_LEARNING_RATES}" ]] && FINE_LEARNING_RATES="${AUTO_FINE_SPACE[0]}"
  [[ -z "${FINE_HIDDEN_DIMS}" ]] && FINE_HIDDEN_DIMS="${AUTO_FINE_SPACE[1]}"
  [[ -z "${FINE_WEIGHT_DECAYS}" ]] && FINE_WEIGHT_DECAYS="${AUTO_FINE_SPACE[2]}"
  [[ -z "${FINE_ACTIVATIONS}" ]] && FINE_ACTIVATIONS="${AUTO_FINE_SPACE[3]}"
fi

echo "Auto-generated fine-search space:"
echo "  learning_rates=${FINE_LEARNING_RATES}"
echo "  hidden_dims=${FINE_HIDDEN_DIMS}"
echo "  weight_decays=${FINE_WEIGHT_DECAYS}"
echo "  activations=${FINE_ACTIVATIONS}"

FINE_SEARCH_SUMMARY_PATH="results/search/${FINE_SEARCH_NAME}_summary.json"
echo "[3/7] Running fine hyperparameter search"
"$PYTHON_BIN" code/search.py \
  --mode "$SEARCH_MODE" \
  --epochs "$FINE_SEARCH_EPOCHS" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-trials "$FINE_SEARCH_MAX_TRIALS" \
  --search-name "$FINE_SEARCH_NAME" \
  --learning-rates "$FINE_LEARNING_RATES" \
  --hidden-dims "$FINE_HIDDEN_DIMS" \
  --weight-decays "$FINE_WEIGHT_DECAYS" \
  --activations "$FINE_ACTIVATIONS" \
  --top-k "$SEARCH_TOP_K"

if [[ "${TRAIN_FROM_FINE_SEARCH_BEST}" == "1" && -f "${FINE_SEARCH_SUMMARY_PATH}" ]]; then
  FINE_BEST_RAW="$(extract_best_params "${FINE_SEARCH_SUMMARY_PATH}")"
  load_lines_into_array FINE_BEST_PARAMS "${FINE_BEST_RAW}"
  LEARNING_RATE="${FINE_BEST_PARAMS[0]}"
  HIDDEN_DIM1="${FINE_BEST_PARAMS[1]}"
  HIDDEN_DIM2="${FINE_BEST_PARAMS[2]}"
  WEIGHT_DECAY="${FINE_BEST_PARAMS[3]}"
  ACTIVATION="${FINE_BEST_PARAMS[4]}"
  echo "Using best fine-search hyperparameters for final training:"
  echo "  learning_rate=${LEARNING_RATE}"
  echo "  hidden_dim1=${HIDDEN_DIM1}"
  echo "  hidden_dim2=${HIDDEN_DIM2}"
  echo "  weight_decay=${WEIGHT_DECAY}"
  echo "  activation=${ACTIVATION}"
fi

echo "[4/7] Training final model"
"$PYTHON_BIN" code/train.py \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --hidden-dim1 "$HIDDEN_DIM1" \
  --hidden-dim2 "$HIDDEN_DIM2" \
  --activation "$ACTIVATION" \
  --learning-rate "$LEARNING_RATE" \
  --lr-decay "$LR_DECAY" \
  --weight-decay "$WEIGHT_DECAY" \
  --run-name "$RUN_NAME"

CHECKPOINT_PATH="results/checkpoints/${RUN_NAME}_best.npz"
HISTORY_PATH="results/curves/${RUN_NAME}_history.json"

echo "[5/7] Evaluating best checkpoint"
"$PYTHON_BIN" code/test.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --run-name "${RUN_NAME}_test"

echo "[6/7] Regenerating visualizations"
"$PYTHON_BIN" code/visualize.py \
  --history "$HISTORY_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --test-summary "results/errors/${RUN_NAME}_test_summary.json" \
  --run-name "${RUN_NAME}_viz"

echo "[7/7] Finalizing summaries"

echo "Pipeline finished."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "History: $HISTORY_PATH"
echo "Log: $LOG_PATH"

TEST_SUMMARY_PATH="results/errors/${RUN_NAME}_test_summary.json"

echo
echo "===== Experiment Summary ====="
echo "Best checkpoint: ${CHECKPOINT_PATH}"
if [[ -f "$TEST_SUMMARY_PATH" ]]; then
  TEST_ACCURACY="$("$PYTHON_BIN" - <<PY
import json
with open("${TEST_SUMMARY_PATH}", "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"{data['test_accuracy']:.4f}")
PY
)"
  echo "Test accuracy: ${TEST_ACCURACY}"
else
  echo "Test accuracy: summary file not found"
fi

if [[ -f "$COARSE_SEARCH_SUMMARY_PATH" ]]; then
  "$PYTHON_BIN" - <<PY
import json
with open("${COARSE_SEARCH_SUMMARY_PATH}", "r", encoding="utf-8") as f:
    data = json.load(f)
best = data.get("best_result", {})
print("Best coarse search result:")
print(
    f"  trial={best.get('trial')} "
    f"lr={best.get('learning_rate')} "
    f"hidden_dim1={best.get('hidden_dim1')} "
    f"hidden_dim2={best.get('hidden_dim2')} "
    f"weight_decay={best.get('weight_decay')} "
    f"activation={best.get('activation')} "
    f"best_val_accuracy={best.get('best_val_accuracy'):.4f}"
)
print(f"  checkpoint={best.get('checkpoint_path')}")
PY
else
  echo "Best coarse search result: summary file not found"
fi

if [[ -f "$FINE_SEARCH_SUMMARY_PATH" ]]; then
  "$PYTHON_BIN" - <<PY
import json
with open("${FINE_SEARCH_SUMMARY_PATH}", "r", encoding="utf-8") as f:
    data = json.load(f)
best = data.get("best_result", {})
print("Best fine search result:")
print(
    f"  trial={best.get('trial')} "
    f"lr={best.get('learning_rate')} "
    f"hidden_dim1={best.get('hidden_dim1')} "
    f"hidden_dim2={best.get('hidden_dim2')} "
    f"weight_decay={best.get('weight_decay')} "
    f"activation={best.get('activation')} "
    f"best_val_accuracy={best.get('best_val_accuracy'):.4f}"
)
print(f"  checkpoint={best.get('checkpoint_path')}")
PY
else
  echo "Best fine search result: summary file not found"
fi
echo "=============================="
