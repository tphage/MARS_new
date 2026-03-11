# Material Discovery Pipeline

An AI-powered system for discovering substitute materials using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Knowledge Graphs. This pipeline helps identify viable material substitutes based on property requirements, application constraints, and knowledge graph insights.

## Overview

The Material Discovery Pipeline consists of three main systems:

1. **System 1: Property Extraction** - Extracts material property requirements from natural language queries
2. **System 2: Material Discovery** - Iteratively proposes and validates candidate materials based on requirements
3. **System 3: Manufacturing Assessment** - Assesses whether a candidate material is manufacturable; uses initial user query/inputs and System 2 output (not System 1)

The pipeline uses a multi-agent architecture with specialized agents for different tasks, combined with RAG retrieval from knowledge bases and knowledge graph reasoning.

## Features

- **Multi-Agent Architecture**: Specialized agents for research, analysis, and material discovery
- **RAG Integration**: Retrieval-Augmented Generation using ChromaDB for document search
- **Knowledge Graph Reasoning**: Leverages material and property relationships from knowledge graphs
- **Iterative Validation**: Closed-loop system that proposes candidates and validates them against requirements
- **PFAS Detection**: Automatic detection and rejection of PFAS (per- and polyfluoroalkyl substances) materials
- **Material Grounding**: Maps lab materials to knowledge graph nodes for enhanced reasoning
- **Subgraph Processing**: Filters and processes relevant subgraphs from knowledge graphs

## Architecture

### Agents

- **ResearchManager**: Generates research questions, proposes candidates, and validates feasibility
- **ResearchAnalyst**: Performs RAG retrieval and analysis from knowledge bases
- **ResearchScientist**: Extracts material classes and finds connections in knowledge graphs
- **ResearchAssistant**: Extracts material property keywords from queries
- **MaterialScientist**: Reasons about material substitutions using multiple evidence sources
- **RejectedCandidateTracker**: Tracks rejected candidates to avoid repetition

### Pipelines

- **Material Requirements Pipeline** (`material_requirements.py`): Extracts material property requirements from natural language
- **Material Discovery Pipeline** (`material_discovery.py`): Iterative candidate proposal and validation workflow
- **Manufacturing Assessment Pipeline** (`manufacturability_assessment.py`): Assesses manufacturability of candidate materials; outputs manufacturable (true/false); supports closed-loop re-run of System 2 on rejection

### Utilities

- **LLM Wrapper**: Unified interface for LLM interactions
- **Embeddings**: Sentence transformer-based embeddings for semantic search
- **Material Database**: Interface to internal material database
- **Subgraph Processor**: Filters and processes knowledge graph subgraphs
- **Material Grounding**: Maps materials to knowledge graph nodes
- **Property Mapper**: Maps user-specified properties to database properties

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for embeddings)
- Access to LLM API (configured in `config/config.yaml`)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd material-discovery-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system:
   - Edit `config/config.yaml` to set LLM API endpoints and model names
   - Update data paths for knowledge graphs, embeddings, and ChromaDB databases
   - Adjust pipeline parameters (max_iterations, similarity thresholds, etc.)

## Configuration

The system is configured via `config/config.yaml`. Key configuration sections:

- **llm**: LLM API settings (base_url, model_name, max_tokens, temperature, seed, timeout)
- **embeddings**: Embedding model configuration
- **data.graphs**: Knowledge graph and embedding locations
  - `kg_dir`: Base directory for all graph files and (some) embeddings
  - `material_properties`: `graph_file`, `embedding_file`
  - `pfas`: `graph_file`, `embedding_file`
  - `patents`: `graph_file`, `embedding_file`
- **data.chromadb**: ChromaDB databases used for RAG
  - `base_path`: Base directory for all ChromaDB databases
  - `pfas`, `materialdb`, `patents`, `manufacturing_textbooks`, `spec_sheets`:
    - `database_path`: Path relative to `base_path`
    - `collection_name`: Optional collection name (if `null`, the first collection is used)
  - `material_database`: Path to internal material database
    - `path`: Full path to the material database JSON file
- **pipelines**: Pipeline-specific parameters
  - `material_requirements`: Settings for System 1
  - `material_discovery`: Settings for System 2 (e.g., `max_iterations`)
  - `manufacturability_assessment`: Settings for future manufacturability features (`enabled`, `n_results_per_source`, `max_process_families`)
- **subgraph_processing**: Subgraph filtering parameters (similarity thresholds, max nodes/edges)

## Usage

### System 1: Property Extraction

See `notebooks/01_system1_property_extraction.ipynb` for examples of extracting material property requirements from natural language queries.

### System 2: Material Discovery

See `notebooks/02_system2_material_discovery.ipynb` for the complete material discovery workflow.

### System 3: Manufacturing Assessment

System 3 is run only from `notebooks/03_manufacturing_assessment.ipynb`, and only after you have run `01_system1_property_extraction.ipynb` and `02_system2_material_discovery.ipynb` first. The notebook loads outputs from those runs (e.g. from `pipeline_logs/system1_*.json` and a saved System 2 result), then runs the manufacturability pipeline.

- **Output status = manufacturable**: Returns a high-level lab-scale process recipe (steps, conditions, equipment class, inputs), evidence pointers. Legacy key `manufacturable=True` and `info_text` summarize the recipe.
- **Output status = blocked**: Returns structured blocking constraints (type, severity, description, optional mitigation and evidence), a concise `feedback_to_system2` string for the material discovery system. If a tracker is provided, the rejected candidate is added with `source="manufacturability"` so you can re-run System 2 later with this constraint in mind. There is no automatic closed-loop orchestrator; re-running System 2 after a manufacturability block is done manually by the user.

## Workflow

### Material Discovery Pipeline

1. **Material Substitution Step**:
   - Load and filter subgraph from System 1
   - Ground lab materials in knowledge graph
   - Retrieve material-property relationships
   - Merge into material-informed subgraph
   - Extract material classes from knowledge graphs

2. **Iterative Candidate Proposal and Validation** (up to `max_iterations`):
   - **Propose Candidate**: ResearchManager proposes a candidate material (avoiding rejected ones)
   - **Generate Validation Queries**: Create queries to validate the candidate
   - **Retrieve Evidence**: ResearchAnalyst retrieves evidence via RAG
   - **Validate Feasibility**: ResearchManager validates against requirements and constraints
   - **If Feasible**: Return candidate
   - **If Not Feasible**: Record constraints, add to rejected list, continue

### Full Pipeline with Manufacturing Assessment (Closed-Loop)

1. Run System 1 → properties W
2. Run System 2 → candidate Z
3. Run System 3 (Manufacturing Assessment) on candidate Z
4. If manufacturable: return candidate + info
5. If not manufacturable: add to rejected tracker with constraints, re-run System 2 (avoids rejected, uses constraints)
6. Repeat until manufacturable candidate found

## Data Sources

The pipeline integrates multiple data sources:

- **Material Properties Knowledge Graph**: Material-property relationships
- **PFAS Knowledge Graph**: PFAS-related research and materials
- **Patent Knowledge Graph**: Patent-derived material information
- **ChromaDB Collections**: 
  - PFAS papers
  - Material database
  - Patents
- **Internal Material Database**: Lab materials with properties

## Output

The pipeline returns a dictionary with:

- `success`: Boolean indicating if a viable candidate was found
- `candidate`: Candidate material dictionary (if successful)
- `iterations`: Number of iterations performed
- `rejected_candidates`: List of rejected candidate names
- `final_constraints`: Constraints that prevented success (if failed)
- `evidence_summary`: Summary of evidence gathered
- `property_mapping`: Property mapping results
- `iteration_history`: Detailed history of each iteration
- `substitution_result`: Results from substitution step

## Notebooks

- `01_system1_property_extraction.ipynb`: Property extraction examples
- `02_system2_material_discovery.ipynb`: Complete material discovery workflow
- `03_manufacturing_assessment.ipynb`: Manufacturing assessment (System 3); inputs: user context + System 2 output
- `chat_visualization.ipynb`: Visualization of agent interactions
- `pipeline_visualization.ipynb`: Timeline and metrics for full pipeline runs
- `evaluation_visualization.ipynb`: Visualization of structured evaluation logs (`evaluation_*.json`)

### Evaluation report PDF export

You can turn a single structured evaluation log (e.g. `pipeline_logs/evaluation_2026031003.json`) into a science-facing PDF report that matches the SG MIT evaluation framework:

- System 1: required material properties and hard constraints
- System 2: final candidate + rejected candidates
- System 3: manufacturing status, process recipe (if present), and blocking constraints

Install dependencies (includes `reportlab` for PDF generation):

```bash
pip install -r requirements.txt
```

Generate a PDF from the command line:

```bash
python -m src.tools.generate_evaluation_pdf \
  -i pipeline_logs/evaluation_2026031003.json \
  -o pipeline_logs/reports/evaluation_2026031003.pdf
```

Useful flags:

- `--label "Response A"`: set a blind-evaluation label shown in the PDF header.
- `--blind`: hide pipeline run ID and timestamps inside the PDF (for blind expert scoring).

From a notebook, you can call the same functionality programmatically:

```python
from src.tools.generate_evaluation_pdf import generate_evaluation_pdf

pdf_path = generate_evaluation_pdf(
    "pipeline_logs/evaluation_2026031003.json",
    label="Response A",
    blind=True,
)
pdf_path
```

## Project Structure

```
material-discovery-pipeline/
├── config/
│   ├── config.yaml          # Main configuration file
│   └── prompts.yaml         # Agent prompts and system messages
├── notebooks/               # Jupyter notebooks for examples
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── research_manager.py
│   │   ├── research_analyst.py
│   │   ├── research_scientist.py
│   │   ├── research_assistant.py
│   │   ├── material_scientist.py
│   │   └── tracker.py
│   ├── pipelines/           # Pipeline workflows
│   │   ├── material_requirements.py
│   │   ├── material_discovery.py
│   │   └── manufacturability_assessment.py
│   ├── utils/               # Utility modules
│   │   ├── llm_wrapper.py
│   │   ├── embeddings.py
│   │   ├── material_database.py
│   │   ├── subgraph_processor.py
│   │   ├── material_grounding.py
│   │   └── property_mapper.py
│   └── config/              # Configuration loading
│       └── loader.py
└── requirements.txt         # Python dependencies
```

## Dependencies

- `autogen`: Multi-agent framework
- `openai`: LLM API client
- `numpy`, `pandas`: Data processing
- `networkx`: Knowledge graph operations
- `torch`: PyTorch for embeddings
- `sentence-transformers`: Embedding models
- `chromadb`: Vector database for RAG
- `pyyaml`: Configuration file parsing

## Notes

- The pipeline automatically rejects PFAS materials as a hard constraint
- Knowledge graphs must be pre-processed and embeddings must be generated
- ChromaDB databases must be populated before running the pipeline
- The system uses deterministic temperature (0) by default for reproducibility

## License

[Add license information here]

## Contact

[Add contact information here]
