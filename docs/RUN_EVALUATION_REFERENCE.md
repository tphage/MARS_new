# Reference: `run_evaluation.py` and evaluation-related files

This document complements [`ABLATION_AND_EVALUATION.md`](ABLATION_AND_EVALUATION.md) with a **file inventory**, a **step-by-step walkthrough** of [`main()`](../scripts/run_evaluation.py) in [`scripts/run_evaluation.py`](../scripts/run_evaluation.py), and **judge prompt details** (LLM call count and what the model sees).

For end-to-end pipeline behavior, prompts, and artifacts, prefer [`ABLATION_AND_EVALUATION.md`](ABLATION_AND_EVALUATION.md).

---

## 1. Files involved in evaluation

### 1.1 Automated LLM-as-judge (ablation benchmark)

| Role | Path |
|------|------|
| Judge entry point | [`scripts/run_evaluation.py`](../scripts/run_evaluation.py) |
| Default rubric (overridable with `--rubric`) | [`config/evaluation_rubric.yaml`](../config/evaluation_rubric.yaml) |
| Optional shell orchestration (`-e` / `--eval`) | [`run_experiments.sh`](../run_experiments.sh) |

**Per-query inputs** (not source files; paths under each `results/<QueryName>/`, or legacy `pipeline_logs_<QueryName>/` at repo root):

| Condition key in code | Typical filename |
|----------------------|------------------|
| `evaluation` (full MARS) | `mars.json` or newest `evaluation_*.json` |
| `ablation_3agent` | `ablation_3agent.json` or `ablation_3agent_*.json` |
| `ablation_1agent_rag` | `ablation_1agent_rag.json` or `ablation_1agent_rag_*.json` |
| `ablation_1agent_no_rag` | `ablation_1agent_no_rag.json` or `ablation_1agent_no_rag_*.json` |

Discovery and loading: [`discover_query_dirs`](../scripts/run_evaluation.py), [`load_all_conditions`](../scripts/run_evaluation.py).

**Default judge outputs:** `--output-dir` defaults to `results/evaluation/` — per-query `eval_<QueryName>.json` and `aggregate_results.json`.

### 1.2 Building baseline `mars.json` / `evaluation_<id>.json`

| Role | Path |
|------|------|
| Payload builder and saver | [`src/utils/evaluation_export.py`](../src/utils/evaluation_export.py) (`build_evaluation_payload`, `save_evaluation_export`) |
| Invokes export after a run | [`src/runner.py`](../src/runner.py) |

`build_evaluation_payload` reads artifact JSON under the query artifacts directory (e.g. System 1/2/3 paths from `pipeline_run`, `rejected_candidates.json`).

### 1.3 Ablation JSONs consumed by the judge

| Role | Path |
|------|------|
| Ablation runner | [`scripts/run_ablations.py`](../scripts/run_ablations.py) |
| Shared evaluation-shaped payloads | [`src/utils/ablation_utils.py`](../src/utils/ablation_utils.py) (`build_ablation_evaluation`, query loading) |
| Ablation prompts | [`config/prompts.yaml`](../config/prompts.yaml) (`ablation` section) |
| Queries | [`config/queries.yaml`](../config/queries.yaml) (or `config/ablation_queries.yaml` if the primary file is missing) |

### 1.4 PDF reports from `evaluation_*.json` (optional)

| Role | Path |
|------|------|
| LaTeX PDF from one export | [`src/tools/generate_evaluation_latex_pdf.py`](../src/tools/generate_evaluation_latex_pdf.py) |
| Blind PDFs across conditions | [`src/tools/generate_blind_evaluation_pdfs.py`](../src/tools/generate_blind_evaluation_pdfs.py) |

### 1.5 In-pipeline “evaluation” (not the LLM judge)

These implement answer gating and feasibility-style checks inside MARS, **not** [`run_evaluation.py`](../scripts/run_evaluation.py):

| Role | Path |
|------|------|
| Prompts (e.g. `evaluate_answer`) | [`config/prompts.yaml`](../config/prompts.yaml) |
| Validation / length settings | [`config/config.yaml`](../config/config.yaml) |
| Question-answered checks | [`src/agents/research_manager.py`](../src/agents/research_manager.py) |
| Material requirements stage | [`src/pipelines/material_requirements.py`](../src/pipelines/material_requirements.py) |
| Manufacturability / feasibility wording | [`src/pipelines/manufacturability_assessment.py`](../src/pipelines/manufacturability_assessment.py) |

---

## 2. `main()` in `run_evaluation.py` (step by step)

Implementation: [`main`](../scripts/run_evaluation.py) (from roughly the `def main():` block through the end of the script).

1. **Parse CLI** — `--model`, `--queries`, `--output-dir`, `--rubric`, `--seed`, `--base-url`.
2. **Require `OPENAI_API_KEY`** — exit if unset.
3. **Load rubric** — [`load_rubric`](../scripts/run_evaluation.py); resolve judge model: `args.model` or `rubric["judge_model"]` or `gpt-4o`.
4. **Construct OpenAI client** — optional `OPENAI_BASE_URL` / `--base-url`.
5. **`random.seed(args.seed)`** — reproducible A–D shuffle in [`evaluate_query`](../scripts/run_evaluation.py).
6. **`discover_query_dirs()`** — query folders that contain all four condition files; exit if none.
7. **Filter by `--queries`** if provided; exit if no matches.
8. **Print** judge model, query list, seed.
9. **Ensure output directory** — `PROJECT_ROOT / args.output_dir`.
10. **For each query:** [`evaluate_query`](../scripts/run_evaluation.py) → append result → write `eval_<query_name>.json`.
11. **If any successful results:** [`print_aggregate_summary`](../scripts/run_evaluation.py) to stdout.
12. **Write `aggregate_results.json`** — metadata, `per_query`, and (if valid results exist) `aggregate_scores` and `avg_ranks`.

---

## 3. Per-query loop: LLM calls and judge visibility

The loop in `main()` calls **`evaluate_query` once per query**. Each call performs **one** judge chat completion in the common case.

### 3.1 How many LLM calls?

- **Per query:** one `client.chat.completions.create` in [`call_judge`](../scripts/run_evaluation.py).
- **Retries:** up to **3** attempts on JSON parse failure or API errors (`max_retries=3`); each attempt is an additional API call.
- The four systems are **not** invoked here — only their **saved JSON** on disk is loaded.

**Total judge calls** ≈ **number of queries** (plus any retries).

### 3.2 What the judge model sees

Messages passed to the API:

1. **`system` message** — Fixed instructions: blind evaluator role, four anonymized systems A–D, independence, do not guess identities, critical assessment, hallucination guidance, **JSON-only** reply. See [`build_judge_prompt`](../scripts/run_evaluation.py).

2. **`user` message** — Built by [`build_judge_prompt`](../scripts/run_evaluation.py):
   - **Material substitution query** — `query_sentence` from `systems["evaluation"]["query"]["sentence"]`.
   - **Evaluation rubric** — each dimension: name, weight, and rubric text from the YAML.
   - **Four blocks `SYSTEM A` … `SYSTEM D`** — for each label, the JSON for the **randomly mapped** condition, pretty-printed. Content is passed through [`strip_raw_responses`](../scripts/run_evaluation.py):
     - Removes top-level `raw_responses`.
     - In `metadata`, drops `ablation_condition` and `pipeline_run_id`.
   - **Required output format** — illustrative JSON schema: per-label scores (1–10) and reasoning per dimension, `overall_comment`, global `ranking` and `ranking_reasoning`.

**What the judge does not see:** which real pipeline corresponds to A–D (shuffle is random per query); `raw_responses`; the stripped metadata fields above.

---

## 4. Related documentation

- [`ABLATION_AND_EVALUATION.md`](ABLATION_AND_EVALUATION.md) — full ablation and judge workflow, discovery rules, rubric dimensions.
- [`README.md`](../README.md) — quick start including `python scripts/run_evaluation.py`.
- [`MARS_AND_KNOWLEDGE_SOURCES.md`](MARS_AND_KNOWLEDGE_SOURCES.md) — Systems 1–3 and knowledge sources (baseline only).
