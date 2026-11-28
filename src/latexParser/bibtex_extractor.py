"""BibTeX extraction hook used by run.py (Milestone 2).

run.py looks for `extract_bibtex` in this module. For ONE version it collects
raw reference entries from, in this order:
  1. \\bibitem entries inside the compiled .tex text
  2. every .bib file under the version folder
  3. compiled .bbl files, only when 1 and 2 found nothing
"""
from pathlib import Path
from typing import Any, Dict, List

from src.latexParser.standardizer import (
    extract_bibitems_from_tex,
    extract_citation_keys,
    parse_bib_file,
)


def extract_bibtex(
    compiled_tex: str,
    version_dir: Path,
    arxiv_id: str = "",
    version: int = 1,
) -> List[Dict[str, Any]]:
    """Return the raw (not yet deduplicated) BibTeX entries of one version."""
    version_dir = Path(version_dir)
    entries: List[Dict[str, Any]] = list(extract_bibitems_from_tex(compiled_tex))

    # Only used to skip uncited entries of huge database-sized .bib files
    cited_keys = set(extract_citation_keys(compiled_tex))
    for bib_file in sorted(version_dir.rglob("*.bib")):
        entries.extend(parse_bib_file(bib_file, filter_keys=cited_keys))

    if not entries:
        for bbl_file in sorted(version_dir.rglob("*.bbl")):
            text = bbl_file.read_text(encoding="utf-8", errors="replace")
            entries.extend(extract_bibitems_from_tex(text))

    return [e for e in entries if str(e.get("citation_key") or "").strip()]