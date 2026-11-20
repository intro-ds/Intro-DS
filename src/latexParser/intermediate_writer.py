"""Intermediate writer — persists parsed hierarchical structures to JSON.

Saves structured publication data (elements, hierarchy, and BibTeX entries)
to data/intermediate/<arxiv_id>/parsed.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_INTERMEDIATE_DIR = Path("data/intermediate")


def save_publication(
    arxiv_id: str,
    elements_rows: List[Dict[str, Any]],
    hierarchy_rows: List[Dict[str, Any]],
    bibtex_rows: List[Dict[str, Any]],
    version_meta: Dict[int, Dict[str, Any]],
    output_dir: Path = DEFAULT_INTERMEDIATE_DIR,
) -> Path:
    """Save parsed publication data to a structured JSON file.

    Args:
        arxiv_id: Publication arXiv ID.
        elements_rows: List of elements table records.
        hierarchy_rows: List of hierarchy table records.
        bibtex_rows: List of unified BibTeX table records.
        version_meta: Per-version parsing metadata (e.g. status, element count).
        output_dir: Destination root directory.

    Returns:
        Path to the saved parsed.json file.
    """
    pub_dir = output_dir / arxiv_id
    pub_dir.mkdir(parents=True, exist_ok=True)
    out_file = pub_dir / "parsed.json"

    data = {
        "arxiv_id": arxiv_id,
        "versions": {str(v): meta for v, meta in sorted(version_meta.items())},
        "elements": elements_rows,
        "hierarchy": hierarchy_rows,
        "bibtex": bibtex_rows,
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_file


def load_publication(
    arxiv_id: str,
    intermediate_dir: Path = DEFAULT_INTERMEDIATE_DIR,
) -> Optional[Dict[str, Any]]:
    """Load parsed publication data from intermediate storage if available.

    Args:
        arxiv_id: Publication arXiv ID.
        intermediate_dir: Intermediate root directory.

    Returns:
        Loaded publication dict if file exists, else None.
    """
    file_path = intermediate_dir / arxiv_id / "parsed.json"
    if not file_path.exists():
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
