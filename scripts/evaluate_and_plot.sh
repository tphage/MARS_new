#!/usr/bin/env bash
# Run LLM-as-judge evaluation on a results tree (e.g. results_new/) and build paper_plots/.
#
# Requires OPENAI_API_KEY. Writes per-query eval_*.json and aggregate_results.json under
# <RESULTS_ROOT>/evaluation/, then runs make_paper_plots.py against that aggregate.
#
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   ./scripts/evaluate_and_plot.sh results_new
#   ./scripts/evaluate_and_plot.sh results_new --queries Query1 --model gpt-4o
#
# Any arguments after RESULTS_ROOT are passed to scripts/run_evaluation.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <RESULTS_ROOT> [run_evaluation.py options]

  RESULTS_ROOT   Directory containing QueryN/ folders with mars.json and ablation_*.json
                 (e.g. results_new). Evaluation output goes to RESULTS_ROOT/evaluation/.

  Remaining args are forwarded to scripts/run_evaluation.py (e.g. --queries Query1).

  Figures and CSVs are written to paper_plots/ at the project root.

Environment: OPENAI_API_KEY must be set.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  exit 1
fi

RESULTS_ROOT="${1:?Usage: $(basename "$0") <RESULTS_ROOT> [run_evaluation.py options — try -h]}"
shift

EVAL_DIR="${RESULTS_ROOT%/}/evaluation"

python scripts/run_evaluation.py \
  --results-root "${RESULTS_ROOT}" \
  --output-dir "${EVAL_DIR}" \
  "$@"

python scripts/make_paper_plots.py \
  --aggregate "${EVAL_DIR}/aggregate_results.json"

echo "[done] Evaluation: ${EVAL_DIR}"
echo "[done] Figures:    ${PROJECT_ROOT}/paper_plots/"
