#!/usr/bin/env bash
# Generate named LaTeX PDFs (MARS + three ablations) and blind Response A–D PDFs
# per query. Uses only the LaTeX/pdflatex path (generate_evaluation_latex_pdf).
#
# Prerequisites: Python env with project on PYTHONPATH (or run from repo root),
# and pdflatex on PATH.
#
# Usage: ./scripts/generate_query_evaluation_pdfs.sh <QueryName> [<QueryName> ...]
# Example: ./scripts/generate_query_evaluation_pdfs.sh Query1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <QueryName> [<QueryName> ...]" >&2
  echo "  Generates under results/<QueryName>/reports/named/ and results/blind_pdfs/." >&2
  exit 1
fi

# Resolve MARS baseline JSON: mars.json, else newest evaluation_*.json in query root,
# else newest artifacts/evaluation_*.json (matches generate_blind_evaluation_pdfs).
resolve_mars_json() {
  local qdir="$1"
  if [[ -f "$qdir/mars.json" ]]; then
    printf '%s\n' "$qdir/mars.json"
    return 0
  fi
  shopt -s nullglob
  local files=( "$qdir"/evaluation_*.json )
  if (( ${#files[@]} )); then
    _pick_newest "${files[@]}"
    shopt -u nullglob
    return 0
  fi
  files=( "$qdir/artifacts"/evaluation_*.json )
  if (( ${#files[@]} )); then
    _pick_newest "${files[@]}"
    shopt -u nullglob
    return 0
  fi
  shopt -u nullglob
  return 1
}

_pick_newest() {
  local newest="$1" f
  for f in "$@"; do
    if [[ "$f" -nt "$newest" ]]; then
      newest="$f"
    fi
  done
  printf '%s\n' "$newest"
}

# Prefer exact filename, else newest prefix*.json in query dir.
resolve_ablation_json() {
  local qdir="$1" exact="$2" prefix="$3"
  if [[ -f "$qdir/$exact" ]]; then
    printf '%s\n' "$qdir/$exact"
    return 0
  fi
  shopt -s nullglob
  local files=( "$qdir/${prefix}"*.json )
  if (( ${#files[@]} )); then
    _pick_newest "${files[@]}"
    shopt -u nullglob
    return 0
  fi
  shopt -u nullglob
  return 1
}

BLIND_OUT="${ROOT}/results/blind_pdfs"

for q in "$@"; do
  QDIR="${ROOT}/results/${q}"
  if [[ ! -d "$QDIR" ]]; then
    echo "error: directory not found: ${QDIR}" >&2
    exit 1
  fi

  MARS_JSON="$(resolve_mars_json "$QDIR")" || {
    echo "error: ${q}: need mars.json, evaluation_*.json in query root, or artifacts/evaluation_*.json" >&2
    exit 1
  }
  A3="$(resolve_ablation_json "$QDIR" "ablation_3agent.json" "ablation_3agent_")" || {
    echo "error: ${q}: missing ablation_3agent.json or ablation_3agent_*.json" >&2
    exit 1
  }
  AR="$(resolve_ablation_json "$QDIR" "ablation_1agent_rag.json" "ablation_1agent_rag_")" || {
    echo "error: ${q}: missing ablation_1agent_rag.json or ablation_1agent_rag_*.json" >&2
    exit 1
  }
  AN="$(resolve_ablation_json "$QDIR" "ablation_1agent_no_rag.json" "ablation_1agent_no_rag_")" || {
    echo "error: ${q}: missing ablation_1agent_no_rag.json or ablation_1agent_no_rag_*.json" >&2
    exit 1
  }

  NAMED="${QDIR}/reports/named"
  mkdir -p "$NAMED"

  echo "==> ${q}: named PDFs -> ${NAMED}"
  python -m src.tools.generate_evaluation_latex_pdf -i "$MARS_JSON" -o "${NAMED}/mars.pdf"
  python -m src.tools.generate_evaluation_latex_pdf -i "$A3" -o "${NAMED}/ablation_3agent.pdf"
  python -m src.tools.generate_evaluation_latex_pdf -i "$AR" -o "${NAMED}/ablation_1agent_rag.pdf"
  python -m src.tools.generate_evaluation_latex_pdf -i "$AN" -o "${NAMED}/ablation_1agent_no_rag.pdf"
done

comma_queries=$(IFS=,; echo "$*")
echo "==> Blind PDFs -> ${BLIND_OUT} (queries: ${comma_queries})"
python -m src.tools.generate_blind_evaluation_pdfs \
  --layout results \
  --results-root "${ROOT}/results" \
  --only-queries "${comma_queries}" \
  --output-dir "${BLIND_OUT}"

echo "Done."
