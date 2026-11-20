# Milestone 2 — LaTeX Parsing Pipeline Execution Guide

This document provides step-by-step instructions for running the **Hierarchical Parsing and Standardization** pipeline (Milestone 2) developed by Group 13.

---

## 1. Environment Setup

Make sure your virtual environment is activated and dependencies are installed.

### Activate Virtual Environment

- **On Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

- **On Linux / WSL / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Verify Dependencies

Install or update all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `pyarrow`, `pytest`, `tqdm`, `psutil`, `matplotlib`, `seaborn`, `jupyter`.

---

## 2. Running Automated Tests

Before running the full pipeline, verify that all components function properly:

### Run Milestone 2 Tests Only
```bash
pytest tests/latexParser/ -v
```

### Run Full Test Suite (Milestones 1 & 2)
```bash
pytest -v
```

All 43 unit and integration tests should pass (`100%`).

---

## 3. Running the Parsing Pipeline (Student 24127559)

The entry point for student **Nguyen Thanh Toan** (`24127559`) is `src.latexParser.parse_24127559`.

### Basic Execution (Default: All Stages)

Parse all assigned publications (`2512.03231` through `2512.08230`) and export Parquet tables:

```bash
python -m src.latexParser.parse_24127559
```

### Quick Test Run (Limit Papers)

To process only a small batch (e.g. 5 papers) for verification:

```bash
python -m src.latexParser.parse_24127559 --limit 5
```

### Running Specific Stages

You can run individual stages using the `--stage` flag:

1. **Parse only** (generates structured JSON in `data/intermediate/`):
   ```bash
   python -m src.latexParser.parse_24127559 --stage parse
   ```

2. **Export only** (reads existing `data/intermediate/` and produces Parquet tables):
   ```bash
   python -m src.latexParser.parse_24127559 --stage export
   ```

3. **Both stages** (default):
   ```bash
   python -m src.latexParser.parse_24127559 --stage all
   ```

### Offset & Subsets

To resume from a specific offset or process a slice:

```bash
# Skip first 100 papers, process the next 50
python -m src.latexParser.parse_24127559 --offset 100 --limit 50
```

### Force Re-Parsing

By default, papers with existing `parsed.json` in `data/intermediate/` are cached and skipped. To force a full re-parse:

```bash
python -m src.latexParser.parse_24127559 --force
```

---

## 4. Pipeline Outputs and Data Structure

The pipeline generates two tiers of data:

### Tier 1: Intermediate Stores (`data/intermediate/`)

Each publication has a dedicated subfolder with a deduplicated hierarchical JSON:
```
data/intermediate/
└── 2512.03231/
    └── parsed.json
```

Structure of `parsed.json`:
- `arxiv_id`: Base publication identifier.
- `versions`: Metadata for each version (parsing success, element count, bib count).
- `elements`: Deduplicated list of distinct elements across all versions.
- `hierarchy`: Edge list of `(arxiv_id, version, element_id, parent_id)` node relationships.
- `bibtex`: Unified, deduplicated BibTeX records.

### Tier 2: Standardized Parquet Tables (`data/processed/24127559/`)

Exported to `data/processed/<student_id>/`:
```
data/processed/24127559/
├── elements.parquet
├── hierarchy.parquet
└── bibtex.parquet
```

#### Table Schemas

| Table | Columns | Types / Constraints | Description |
|---|---|---|---|
| `elements.parquet` | `element_id`, `arxiv_id`, `kind`, `content` | `string`, `string`, `category`, `string` | Distinct elements (full-text deduplicated across versions). |
| `hierarchy.parquet` | `arxiv_id`, `version`, `element_id`, `parent_id` | `string`, `int64`, `string`, `string` | Node-parent relationship edges for each version. |
| `bibtex.parquet` | `arxiv_id`, `citation_key`, `entry_type`, `title`, `authors`, `year`, `doi`, `journal`, `url` | `string`, `string`, `category`, `string`, `list[string]`, `Int64`, `string`, `string`, `string` | Unified and deduplicated bibliography entries with merged fields. |

---

## 5. Inspecting Output Tables

You can inspect the generated Parquet files using Python and pandas:

```python
import pandas as pd

# Load elements table
elements_df = pd.read_parquet("data/processed/24127559/elements.parquet")
print(elements_df.head())
print(elements_df["kind"].value_counts())

# Load hierarchy table
hierarchy_df = pd.read_parquet("data/processed/24127559/hierarchy.parquet")
print(hierarchy_df.head())

# Load bibtex table
bibtex_df = pd.read_parquet("data/processed/24127559/bibtex.parquet")
print(bibtex_df.head())
```

---

## 6. Running the Exploratory Data Analysis (EDA) Notebook

The Jupyter notebook provides data exploration, distribution plots, and performance metrics.

### Interactive Execution

Start JupyterLab or Jupyter Notebook:
```bash
jupyter lab
# or
jupyter notebook
```
Open and run `notebooks/02_latex_parsing_eda.ipynb` top-to-bottom.

### Headless Batch Execution

To execute the notebook headlessly and update all figures in-place:
```bash
jupyter nbconvert --to notebook --execute notebooks/02_latex_parsing_eda.ipynb --output 02_latex_parsing_eda.ipynb
```

---

## 7. Performance & Resource Benchmarking

Per the course guidelines, the parser is optimized for single-threaded CPU execution (Google Colab compatibility):

- **Throughput**: ~1.5 to 2.5 seconds per publication (including multi-version transitive parsing, AST construction, and deduplication).
- **RAM footprint**: ~100 MB to 150 MB baseline RAM, well within standard hardware limits.
- **Disk storage**: Zero external downloads during parsing; intermediate JSONs and Parquet compressions achieve significant space savings compared to raw TeX archives.
