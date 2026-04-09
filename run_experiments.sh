#!/usr/bin/env bash
# run_experiments.sh — Reproduce MARS experiments
#
# Usage:
#   ./run_experiments.sh [OPTIONS]
#
# Options:
#   -q, --queries    Comma-separated query names (default: Query1,Query2,Query3)
#   -a, --ablations  Also run ablation conditions (3agent, 1agent_rag, 1agent_no_rag)
#   -e, --eval       Also run LLM-judge evaluation after pipeline/ablations
#   -c, --condition  Specific ablation condition to run (3agent|1agent_rag|1agent_no_rag)
#                    Only meaningful when --ablations is set. Default: all conditions.
#   -h, --help       Show this help message
#
# Examples:
#   ./run_experiments.sh                              # full MARS only, all 3 queries
#   ./run_experiments.sh -a                           # full MARS + all ablations
#   ./run_experiments.sh -a -e                        # full MARS + ablations + evaluation
#   ./run_experiments.sh -q Query1,Query2 -a          # two queries, MARS + ablations
#   ./run_experiments.sh -q Query1 -a -c 3agent       # one query, MARS + 3agent only
#   ./run_experiments.sh -e                           # evaluation only (re-score existing results)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
QUERIES="Query1,Query2,Query3"
RUN_ABLATIONS=false
RUN_EVAL=false
CONDITION=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -q|--queries)
      QUERIES="$2"; shift 2 ;;
    -a|--ablations)
      RUN_ABLATIONS=true; shift ;;
    -e|--eval)
      RUN_EVAL=true; shift ;;
    -c|--condition)
      CONDITION="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "Unknown option: $1"
      echo "Run './run_experiments.sh --help' for usage."
      exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Locate project root (directory containing this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "MARS Experiment Runner"
echo "============================================================"
echo "  Queries:    $QUERIES"
echo "  MARS:       yes (always)"
echo "  Ablations:  $RUN_ABLATIONS$(if $RUN_ABLATIONS && [[ -n "$CONDITION" ]]; then echo " (condition: $CONDITION)"; fi)"
echo "  Evaluation: $RUN_EVAL"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Full MARS pipeline
# ---------------------------------------------------------------------------
echo "------------------------------------------------------------"
echo "Step 1: Full MARS Pipeline"
echo "------------------------------------------------------------"
python scripts/run_mars.py --queries "$QUERIES"

# ---------------------------------------------------------------------------
# Step 2: Ablation conditions (optional)
# ---------------------------------------------------------------------------
if $RUN_ABLATIONS; then
  echo ""
  echo "------------------------------------------------------------"
  echo "Step 2: Ablation Conditions"
  echo "------------------------------------------------------------"
  ABLATION_ARGS="--queries $QUERIES"
  if [[ -n "$CONDITION" ]]; then
    ABLATION_ARGS="$ABLATION_ARGS --condition $CONDITION"
  fi
  python scripts/run_ablations.py $ABLATION_ARGS
fi

# ---------------------------------------------------------------------------
# Step 3: LLM-judge evaluation (optional)
# ---------------------------------------------------------------------------
if $RUN_EVAL; then
  echo ""
  echo "------------------------------------------------------------"
  echo "Step 3: LLM-Judge Evaluation"
  echo "------------------------------------------------------------"
  python scripts/run_evaluation.py --queries "$QUERIES"
fi

echo ""
echo "============================================================"
echo "Done."
echo "============================================================"
