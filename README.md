# MARS: Hierarchical Multi-Agent Reasoning for Manufacturability-Aware Material Substitution

MARS is a three-system LLM pipeline that discovers substitute materials using knowledge graphs, retrieval-augmented generation, and iterative multi-agent reasoning. Given a material-substitution query, it extracts required properties, proposes and validates candidates against two domain knowledge graphs and four RAG corpora, and assesses lab-scale manufacturability — looping until a viable candidate is found.

```
System 1 (Property Extraction)
        │
        ▼
System 2 (Material Discovery) ◄──┐
        │                         │ feedback
        ▼                         │
System 3 (Manufacturability) ─────┘
        │
        ▼
   Manufacturable candidate
```

## Repository Structure

```
├── config/
│   ├── config.yaml              # LLM, embeddings, data paths, agent hyperparameters
│   ├── prompts.yaml             # All LLM system/user prompts
│   ├── queries.yaml             # Benchmark queries (Query1–3)
│   └── evaluation_rubric.yaml   # LLM judge + expert PDF: 12 subsystem criteria, 1–5 scale
├── src/
│   ├── runner.py                # Pipeline orchestrator (initialize + run_query)
│   ├── agents/                  # ResearchManager, ResearchScientist, …
│   ├── pipelines/               # System 1, 2, 3 pipeline functions
│   ├── config/                  # YAML loader with ${ENV_VAR} interpolation
│   └── utils/                   # LLM wrapper, embeddings, ChromaDB helpers, KG tools, …
├── scripts/
│   ├── run_mars.py              # Run full MARS pipeline (batch)
│   ├── run_ablations.py         # Run ablation conditions (batch)
│   └── run_evaluation.py        # LLM-as-judge blind evaluation
├── notebooks/
│   └── walkthrough.ipynb        # Interactive single-query demo + visualization
└── results/                     # Frozen experiment outputs
    ├── Query1/                  # mars.json + ablation_*.json + artifacts/
    ├── Query2/
    ├── Query3/
    └── evaluation/              # Per-query and aggregate judge scores
```

## Prerequisites

- Python 3.10+
- An OpenAI-compatible LLM endpoint (configured in `config/config.yaml`)
- Domain data: knowledge graphs (`.graphml` + embedding `.pkl` files) and ChromaDB databases
- The `GraphReasoning` package (see installation below)

## Installation

```bash
git clone <repo-url> && cd MARS
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` to set:

- **`llm.base_url`** / **`llm.api_key`** — your LLM endpoint
- **`data.graphs.kg_dir`** — directory containing the three `.graphml` + `.pkl` files
- **`data.chromadb.base_path`** — directory containing the ChromaDB databases
- **`data.material_database.path`** — path to `internal_material_database.json`

The config loader supports `${ENV_VAR}` and `${ENV_VAR:-default}` interpolation, so you can set paths via environment variables instead of editing the file directly.

**Strict data loading:** `initialize()` fails immediately if any required path is missing or invalid: the three knowledge-graph file pairs under `data.graphs.kg_dir`, the four Chroma persist directories (`pfas`, `patents`, `materialdb`, `manufacturing_textbooks` under `data.chromadb.base_path`), and `data.material_database.path`. Optional **`data.chromadb.spec_sheets`** is only loaded when `enabled: true`; in that case the spec-sheet Chroma path must exist or initialization raises.

## Reproducing the Experiments

Three commands reproduce all results:

```bash
# 1. Run the full MARS pipeline for all 3 benchmark queries
python scripts/run_mars.py

# 2. Run all ablation conditions (3-agent, 1-agent+RAG, 1-agent no-RAG)
python scripts/run_ablations.py

# 3. Run the blind LLM-as-judge evaluation
export OPENAI_API_KEY="sk-..."
python scripts/run_evaluation.py
```

Each script accepts `--queries Query1,Query2` to run a subset. Results are written to `results/<QueryName>/`.

For a detailed description of how ablations and the LLM-as-judge evaluation work at runtime (artifacts, retrieval behavior, and rubric), see [`docs/ABLATION_AND_EVALUATION.md`](docs/ABLATION_AND_EVALUATION.md).

### Ablation Conditions

| Condition | Description |
|-----------|-------------|
| **MARS (Full Pipeline)** | System 1 → System 2 ↔ System 3 with RAG + dual-KG reasoning |
| **3-agent** | 3 sequential LLM calls mimicking S1→S2→S3, no RAG or KG |
| **1-agent + RAG/KG** | Single LLM call with pre-retrieved RAG + KG context |
| **1-agent (no RAG/KG)** | Single LLM call, purely parametric knowledge |

### LLM-Judge Evaluation

`scripts/run_evaluation.py` randomizes system labels (A–D) for blind scoring across **12 subsystem criteria** (four each for Systems 1–3, matching the paper table) on a **1–5** ordinal scale. See `config/evaluation_rubric.yaml` for the full rubric (shared with expert LaTeX PDF forms).

## Interactive Walkthrough

Open `notebooks/walkthrough.ipynb` to run the pipeline interactively for a single query and inspect:

- Execution timeline (Gantt chart)
- Agent chat logs (System 1, 2, 3)
- Knowledge graph subgraph visualizations
- RAG retrieval results
- Iteration-by-iteration candidate proposals and feedback

## Frozen Results

The `results/` directory contains pre-computed outputs for all three benchmark queries. These are the exact outputs used in the paper.

## Citation

```bibtex
@article{mars2026,
  title  = {MARS: Hierarchical Multi-Agent Reasoning Systems Enable Manufacturability-Aware Material Substitution Using Knowledge Graphs},
  author = {TODO},
  year   = {2026},
}
```

## License

See [LICENSE](LICENSE).
