#!/usr/bin/env bash
# Run scripts/run_evaluation.py N times with distinct seeds and output dirs, then
# aggregate mean/std/min/max via scripts/summarize_evaluation_runs.py.
#
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   ./scripts/run_evaluation_repeated.sh
#   ./scripts/run_evaluation_repeated.sh -n 10 -o results/evaluation_runs/my_batch
#   ./scripts/run_evaluation_repeated.sh -- --queries Query1 --model gpt-4o
#
# Extra arguments after -- are passed to run_evaluation.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

N=3
OUTPUT_BASE=""
BASE_SEED=42
PYTHON_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- EXTRA_ARGS_FOR_run_evaluation.py]

Options:
  -n, --runs N          Number of evaluation runs (default: 3)
  -o, --output-base DIR Directory for this batch (default: results/evaluation_runs/TIMESTAMP)
  -s, --base-seed S     Seed for run 1; run i uses S + i - 1 (default: 42)
  -h, --help            Show this help

Each run writes to DIR/run_01, DIR/run_02, ... and a summary.json is written to DIR.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--runs)
      N="$2"
      shift 2
      ;;
    -o|--output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    -s|--base-seed)
      BASE_SEED="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      PYTHON_ARGS+=("$@")
      break
      ;;
    *)
      PYTHON_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! [[ "${N}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: -n must be a positive integer, got: ${N}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_BASE}" ]]; then
  OUTPUT_BASE="${PROJECT_ROOT}/results/evaluation_runs/$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${OUTPUT_BASE}"

# Paths relative to project root for --output-dir
REL_BASE="${OUTPUT_BASE#${PROJECT_ROOT}/}"
if [[ "${REL_BASE}" == "${OUTPUT_BASE}" ]]; then
  echo "ERROR: OUTPUT_BASE must be under PROJECT_ROOT (${PROJECT_ROOT})" >&2
  exit 1
fi

echo "Batch directory: ${OUTPUT_BASE}"
echo "Runs: ${N}, base seed: ${BASE_SEED}"
echo

for ((i = 1; i <= N; i++)); do
  RUN_LABEL=$(printf 'run_%02d' "${i}")
  RUN_DIR_REL="${REL_BASE}/${RUN_LABEL}"
  SEED=$((BASE_SEED + i - 1))
  echo "=== ${RUN_LABEL} (seed=${SEED}) -> ${RUN_DIR_REL} ==="
  # "${arr[@]+...}" avoids nounset errors when PYTHON_ARGS is empty (bash + set -u).
  python "${PROJECT_ROOT}/scripts/run_evaluation.py" \
    --seed "${SEED}" \
    --output-dir "${RUN_DIR_REL}" \
    ${PYTHON_ARGS[@]+"${PYTHON_ARGS[@]}"}
  echo
done

SUMMARY_OUT="${OUTPUT_BASE}/summary.json"
echo "=== Aggregating statistics ==="
python "${PROJECT_ROOT}/scripts/summarize_evaluation_runs.py" "${OUTPUT_BASE}" -o "${SUMMARY_OUT}"
echo "Done. Summary: ${SUMMARY_OUT}"
