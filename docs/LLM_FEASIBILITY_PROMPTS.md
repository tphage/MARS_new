# What the LLM sees for feasibility (System 2 vs System 3)

This document describes **which prompts** are used and **which structured fields** are passed into the model when the pipeline evaluates feasibility. It reflects the implementation in `src/agents/research_manager.py`, `src/pipelines/material_discovery.py`, and `src/pipelines/manufacturability_assessment.py`, with prompt text defined in `config/prompts.yaml`.

---

## System 2 — application feasibility (`validate_feasibility`)

**Purpose:** Decide whether the proposed **substitute material** fits the **application** requirements (properties + constraints), using evidence gathered after validation queries.

**Where it runs:** `ResearchManager.validate_feasibility` is invoked from the material discovery pipeline after RAG retrieval and `answer_question` have been run for each validation query (`src/pipelines/material_discovery.py`, step `[2d]`).

### System message

- **YAML path:** `agents.research_manager.validate_feasibility`
- **Content:** Instructions for hard constraints, critical properties, positive evidence, and overall feasibility (see `config/prompts.yaml`).

### User message (template)

- **YAML path:** `agents.research_manager.validate_feasibility_user_prompt`
- **Placeholders filled by code:**

| Placeholder | Source (conceptually) |
|-------------|------------------------|
| `{candidate_name}` | `candidate_Z["material_name"]` |
| `{evidence_list}` | Built in code **per validation query** — **not** raw retrieved document text |
| `{kg_evidence_section}` | Optional KG summary / formatted paths from the material-informed subgraph |
| `{property_list}` | `properties_W["required"]` (+ `target_values` when present) |
| `{constraints_section}` | `constraints_U` (bulleted list) |
| `{kg_consideration}` | Extra instruction line when KG evidence or paths are present |

### What `{evidence_list}` actually contains

For each item in `evidence_I`, the code appends (`src/agents/research_manager.py`, `validate_feasibility`):

- `[Evidence i]`
- `Query:` the validation question string
- `Answer:` the **ResearchManager `answer_question` output**, truncated to **300 characters** (not the raw RAG chunks)
- `Documents Retrieved:` **count only** (`len(rag_results)`), not document bodies

So the **feasibility judge does not see the original patent/MaterialDB chunks** — only **short distilled answers** plus **how many** documents were retrieved per query.

Optional **knowledge graph** context is added separately: path summaries from `_find_paths_in_subgraph` when a subgraph and embeddings are available, plus a short note from `kg_evidence` (e.g. connected nodes) when present.

### Truncation

- The full user prompt may be cut to `agents.research_manager.max_prompt_chars` via `_truncate_prompt` (see `config/config.yaml`).

### Expected model output format

The user prompt asks for a fixed text layout ending with:

`FEASIBLE: [YES/NO]`, `CONSTRAINTS_VIOLATED:`, `REASONING:`

The parser in `validate_feasibility` reads these lines from the model response.

---

## System 3 — lab-scale manufacturability (three related LLM calls)

System 3 assesses whether the candidate can be **manufactured at lab scale**, using **manufacturing-oriented** RAG (textbooks, patents, material DB, optional spec sheets). This is **not** the same call as System 2 `validate_feasibility`.

**Implementation:** `src/pipelines/manufacturability_assessment.py`.

### Special case: no process evidence

If **Step 1** (process retrieval) yields **zero** documents after deduplication/capping, the pipeline returns **`blocked`** immediately with a fixed message. **No** `generate_feasibility_questions`, **no** `answer_feasibility_question`, and **no** `assess_manufacturability_feasibility` LLM call runs in that case.

---

### Step A — Generate feasibility questions (not a pass/fail verdict)

**Method:** `ResearchManager.generate_feasibility_questions`

| Part | Detail |
|------|--------|
| **System prompt** | `pipelines.manufacturability_assessment.generate_feasibility_questions` (includes `{num_questions}`) |
| **User prompt** | `pipelines.manufacturability_assessment.generate_feasibility_questions_user_prompt` |
| **Inputs** | `material_name`, `material_class`, `application_Y`, comma-separated **required properties**, **constraints** (multi-line bullet list), **`rag_context`** = formatted Step-1 process retrieval hits, **`evidence_coverage`** = JSON string (constituents with/without evidence, combination-query stats, etc.) |
| **RAG formatting** | `_format_rag_context(..., max_chars_per_result=self.max_chars_per_result_feasibility)` — default cap per document from `agents.research_manager.formatting.max_chars_per_result_feasibility` (often **2000** characters per chunk; see `config/config.yaml`) |
| **Output** | JSON: `{ "questions": [ ... ] }` with exactly `num_questions` strings (config: `pipelines.manufacturability_assessment.num_feasibility_questions`, default **4**) |

This step **only generates questions**; it does not output feasible/blocked.

---

### Step B — Answer each feasibility question (evidence-heavy)

**Method:** `ResearchManager.answer_feasibility_question` — called **once per question**.

| Part | Detail |
|------|--------|
| **System prompt** | `pipelines.manufacturability_assessment.answer_feasibility_question` |
| **User prompt** | `pipelines.manufacturability_assessment.answer_feasibility_question_user_prompt` |
| **Inputs** | `material_name`, `material_class`, the **single** `question`, **`rag_context`** = documents retrieved for **that question only** via `process_analyst.analyze_question(question)` |
| **RAG formatting** | Same `_format_rag_context` with `max_chars_per_result_feasibility` |

The model returns JSON: `answer`, `confidence` (`high` | `medium` | `low` | `insufficient_evidence`), `evidence_used`.

---

### Step C — Aggregate into final manufacturability verdict (**feasibility evaluation** proper)

**Method:** `ResearchManager.assess_manufacturability_feasibility`

This is the System 3 **final feasibility** LLM call that maps Q&A → `feasible`, `blocking_constraints`, `feedback_to_system2`.

| Part | Detail |
|------|--------|
| **System prompt** | `pipelines.manufacturability_assessment.assess_feasibility` |
| **User prompt** | `pipelines.manufacturability_assessment.assess_feasibility_user_prompt` |
| **Inputs** | `material_name`, `material_class`, `application_Y`, **required properties** (comma-separated), **constraints** (bulleted), **`question_answers`** = formatted string built from Step B (each block: question, answer, confidence, evidence_used), **`evidence_coverage`** = same JSON as in Step A |

**Important:** This aggregation step **does not** re-include the full RAG document text. It only sees the **summarized Q&A** and **confidence labels** plus **evidence_coverage** metadata.

Expected model output: a single JSON object with `feasible`, `blocking_constraints`, `feedback_to_system2` (parsed in code).

Additional **Python** logic after the LLM (e.g. composite combination-query heuristics, hard vs soft blocking types) can override or adjust `feasible` — see `src/pipelines/manufacturability_assessment.py` after `assess_manufacturability_feasibility` returns.

---

## Quick comparison

| Aspect | System 2 `validate_feasibility` | System 3 manufacturability |
|--------|---------------------------------|----------------------------|
| **Question** | Does the material **meet application requirements**? | Can it be **made at lab scale**? |
| **Primary evidence in the judge prompt** | Short **answers** per validation query + doc **counts**; optional KG paths | Step B sees **full** formatted RAG per question; Step C sees **Q&A summaries** only |
| **Raw RAG chunks in the final judge call?** | **No** | **No** (only in `answer_feasibility_question`, not in `assess_feasibility`) |
| **Prompt keys in YAML** | `agents.research_manager.validate_feasibility` + `validate_feasibility_user_prompt` | `pipelines.manufacturability_assessment.answer_feasibility_question` / `assess_feasibility` (+ question generator) |

---

## Config knobs that shape context size

- `agents.research_manager.max_prompt_chars` — safety truncation for System 2 feasibility user prompt.
- `agents.research_manager.formatting.max_chars_per_result_feasibility` — per-document cap when formatting RAG for System 3 Steps A and B (default commonly **2000**).
- `pipelines.manufacturability_assessment.num_feasibility_questions` — number of Q&A rounds (default **4**).

---

## Source files

- Prompt templates: `config/prompts.yaml` (`validate_feasibility`, `validate_feasibility_user_prompt`, `generate_feasibility_questions`, `answer_feasibility_question`, `assess_feasibility`, etc.).
- System 2 feasibility assembly: `src/agents/research_manager.py` → `validate_feasibility`.
- System 2 orchestration: `src/pipelines/material_discovery.py` → steps `[2c]`–`[2d]`.
- System 3 pipeline: `src/pipelines/manufacturability_assessment.py`.
