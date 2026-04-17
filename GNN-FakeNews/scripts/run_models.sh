#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
#   bash scripts/run_models.sh
#   bash scripts/run_models.sh gin_base
#   bash scripts/run_models.sh gin_topo gin_topo_pr
#
# Optional environment overrides:
#   DATASET=politifact FEATURE=bert DEVICE=cpu BATCH_SIZE=64 LR=0.001 \
#   WEIGHT_DECAY=0.0005 NHID=128 DROPOUT=0.5 EPOCHS=50 \
#   bash scripts/run_models.sh gin_topo

DATASET="${DATASET:-politifact}"
FEATURE="${FEATURE:-bert}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
NHID="${NHID:-128}"
DROPOUT="${DROPOUT:-0.5}"
EPOCHS="${EPOCHS:-50}"

MODELS_DIR="gnn/models"

declare -A MODEL_FILES=(
  ["gin_base"]="${MODELS_DIR}/gin_base.py"
  ["gin_topo"]="${MODELS_DIR}/gin_topo.py"
  ["gin_topo_pr"]="${MODELS_DIR}/gin_topo_pr.py"
  ["gin_topo_temp"]="${MODELS_DIR}/gin_topo_temp.py"
)

print_usage() {
  cat <<EOF
Usage:
  bash scripts/run_models.sh [model1 model2 ...]

Available models:
  gin_base
  gin_topo
  gin_topo_pr
  gin_topo_temp

Examples:
  bash scripts/run_models.sh
  bash scripts/run_models.sh gin_base
  bash scripts/run_models.sh gin_topo gin_topo_pr
  DATASET=gossipcop bash scripts/run_models.sh gin_base gin_topo_temp
EOF
}

validate_model() {
  local model="$1"
  if [[ -z "${MODEL_FILES[$model]:-}" ]]; then
    echo "Error: unknown model '$model'"
    print_usage
    exit 1
  fi
  if [[ ! -f "${MODEL_FILES[$model]}" ]]; then
    echo "Error: file not found: ${MODEL_FILES[$model]}"
    exit 1
  fi
}

run_model() {
  local model="$1"
  local script="${MODEL_FILES[$model]}"

  echo "============================================================"
  echo "Running model: ${model}"
  echo "Script: ${script}"
  echo "Dataset: ${DATASET}"
  echo "Feature: ${FEATURE}"
  echo "Device: ${DEVICE}"
  echo "Batch size: ${BATCH_SIZE}"
  echo "LR: ${LR}"
  echo "Weight decay: ${WEIGHT_DECAY}"
  echo "Hidden size: ${NHID}"
  echo "Dropout: ${DROPOUT}"
  echo "Epochs: ${EPOCHS}"
  echo "============================================================"

  python "${script}" \
    --dataset "${DATASET}" \
    --feature "${FEATURE}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --nhid "${NHID}" \
    --dropout_ratio "${DROPOUT}" \
    --epochs "${EPOCHS}"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
  fi

  local selected_models=()

  if [[ "$#" -eq 0 ]]; then
    selected_models=("gin_base" "gin_topo" "gin_topo_pr" "gin_topo_temp")
  else
    selected_models=("$@")
  fi

  for model in "${selected_models[@]}"; do
    validate_model "${model}"
  done

  mkdir -p results/logs

  for model in "${selected_models[@]}"; do
    run_model "${model}" 2>&1 | tee "results/logs/${model}_${DATASET}.log"
    echo
  done
}

main "$@"