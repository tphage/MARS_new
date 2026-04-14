# Generating evaluation PDFs (named + blind)

This describes how to turn per-query evaluation JSON exports into **LaTeX PDFs** using [`scripts/generate_query_evaluation_pdfs.sh`](../scripts/generate_query_evaluation_pdfs.sh). All PDFs are produced via [`src/tools/generate_evaluation_latex_pdf.py`](../src/tools/generate_evaluation_latex_pdf.py) (`pdflatex`); there is no ReportLab path in this workflow.

For where the JSON files come from (full MARS run, ablations, judge), see [`ABLATION_AND_EVALUATION.md`](ABLATION_AND_EVALUATION.md) and [`RUN_EVALUATION_REFERENCE.md`](RUN_EVALUATION_REFERENCE.md).

---

## Prerequisites

- **TeX:** `pdflatex` on your `PATH` (e.g. TeX Live or MacTeX). If it is missing, the Python tools fail with an explicit error.
- **Python:** Run commands from the **repository root** so `python -m src.tools...` resolves, or set `PYTHONPATH` to the repo root.

---

## One-command workflow (recommended)

From the repo root:

```bash
./scripts/generate_query_evaluation_pdfs.sh Query1
./scripts/generate_query_evaluation_pdfs.sh Query1 Query2
```

Arguments are **folder names** under `results/`, not full paths.

### What it does

1. **Named PDFs** (full structured report per condition)  
   For each query, writes four files under `results/<QueryName>/reports/named/`:

   | Output file | Source JSON (resolved automatically) |
   |-------------|--------------------------------------|
   | `mars.pdf` | `mars.json`, or newest `evaluation_*.json` in the query folder, or newest `artifacts/evaluation_*.json` |
   | `ablation_3agent.pdf` | `ablation_3agent.json` or newest `ablation_3agent_*.json` |
   | `ablation_1agent_rag.pdf` | `ablation_1agent_rag.json` or newest `ablation_1agent_rag_*.json` |
   | `ablation_1agent_no_rag.pdf` | `ablation_1agent_no_rag.json` or newest `ablation_1agent_no_rag_*.json` |

   Resolution matches [`generate_blind_evaluation_pdfs`](../src/tools/generate_blind_evaluation_pdfs.py) and the judge’s discovery in [`scripts/run_evaluation.py`](../scripts/run_evaluation.py).

2. **Blind PDFs** (for expert review)  
   One extra step runs after all named PDFs: it builds **Response A–D** PDFs with shuffled labels (no pipeline IDs in the body) and writes:

   - `results/blind_pdfs/<QueryName>/Response_A.pdf` … `Response_D.pdf`
   - `results/blind_pdfs/blind_key.json` — maps each label to the internal condition key (`full_mars`, `3agent`, `1agent_rag`, `1agent_no_rag`).

   Blind PDFs omit rejected-candidate detail and the hard-constraints block (same options as the blind generator uses internally).

---

## Running the pieces manually

### Single PDF from one JSON

```bash
python -m src.tools.generate_evaluation_latex_pdf \
  -i results/Query1/mars.json \
  -o /path/to/out.pdf
```

Optional: `--blind` and `--label "Response A"` for a single anonymized-style export (see the module’s `--help`).

### Blind batch only (`results/` layout)

```bash
python -m src.tools.generate_blind_evaluation_pdfs \
  --layout results \
  --results-root results \
  --only-queries Query1,Query2 \
  --output-dir results/blind_pdfs \
  --seed 42
```

Omit `--only-queries` to process **every** `results/<QueryName>/` directory that already contains all four condition JSON files.

### Legacy layout (`pipeline_logs_Query*`)

If you still keep exports under `pipeline_logs_Query1/` at the repo root:

```bash
python -m src.tools.generate_blind_evaluation_pdfs \
  --layout legacy \
  --queries-dir . \
  --output-dir evaluation_results/blind_pdfs
```

---

## Troubleshooting

- **`pdflatex not found`:** Install a LaTeX distribution and ensure `pdflatex` is on `PATH`.
- **Missing JSON:** The shell script exits with a short error if any of the four inputs for a query cannot be resolved. Run MARS and [`scripts/run_ablations.py`](../scripts/run_ablations.py) for that query first.
- **Blind step fails on `--only-queries`:** Each named query must exist under `results/` and include **all four** exports; otherwise the blind generator exits with a descriptive message.
