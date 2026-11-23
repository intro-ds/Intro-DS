"""LaTeX parsing pipeline orchestrator.

Parses every version of assigned publications following compilation paths,
builds hierarchical document trees, extracts and unifies BibTeX references,
and persists intermediate structured stores.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from src.latexParser.deduplicator import deduplicate_bibtex, deduplicate_elements
from src.latexParser.hierarchy_builder import Element, build_hierarchy, flatten_tree
from src.latexParser.intermediate_writer import DEFAULT_INTERMEDIATE_DIR, save_publication
from src.latexParser.standardizer import (
    extract_bibitems_from_tex,
    extract_citation_keys,
    parse_bib_file,
)
from src.latexParser.tex_gatherer import find_root_tex, gather_tex_files

DEFAULT_CORPUS_DIR = Path("data/raw/corpus")


def _discover_versions(paper_dir: Path, arxiv_id: str) -> List[int]:
    """Identify all version numbers present for a paper."""
    version_nums: Set[int] = set()

    # 1. From metadata.json if present
    meta_path = paper_dir / "metadata.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            for v_str in meta.get("versions", []):
                match = re.search(r"v(\d+)$", v_str)
                if match:
                    version_nums.add(int(match.group(1)))
        except (OSError, json.JSONDecodeError):
            pass

    # 2. From filesystem subdirectories
    for child in paper_dir.glob("v*"):
        if child.is_dir():
            match = re.match(r"^v(\d+)$", child.name)
            if match:
                version_nums.add(int(match.group(1)))

    if not version_nums:
        # Default to version 1 if directory exists
        if paper_dir.exists():
            version_nums.add(1)

    return sorted(version_nums)


def parse_publication(
    arxiv_id: str,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    intermediate_dir: Path = DEFAULT_INTERMEDIATE_DIR,
    force: bool = False,
) -> Dict[str, Any]:
    """Execute parsing, standardization, and deduplication for one publication.

    Args:
        arxiv_id: Base arXiv identifier.
        corpus_dir: Directory containing raw paper files.
        intermediate_dir: Directory to save structured intermediate JSON.
        force: If True, re-parse even if intermediate output exists.

    Returns:
        Summary dict containing status, element counts, and version details.
    """
    out_json = intermediate_dir / arxiv_id / "parsed.json"
    if out_json.exists() and not force:
        try:
            with out_json.open("r", encoding="utf-8") as f:
                cached_data = json.load(f)
            return {
                "arxiv_id": arxiv_id,
                "status": "cached",
                "elements_count": len(cached_data.get("elements", [])),
                "hierarchy_count": len(cached_data.get("hierarchy", [])),
                "bibtex_count": len(cached_data.get("bibtex", [])),
            }
        except (OSError, json.JSONDecodeError):
            pass

    paper_dir = corpus_dir / arxiv_id
    if not paper_dir.exists():
        # Record empty publication per spec
        save_publication(
            arxiv_id=arxiv_id,
            elements_rows=[],
            hierarchy_rows=[],
            bibtex_rows=[],
            version_meta={1: {"parsed": False, "reason": "directory_not_found"}},
            output_dir=intermediate_dir,
        )
        return {
            "arxiv_id": arxiv_id,
            "status": "empty_no_dir",
            "elements_count": 0,
            "hierarchy_count": 0,
            "bibtex_count": 0,
        }

    version_nums = _discover_versions(paper_dir, arxiv_id)
    version_trees: Dict[int, Element] = {}
    version_bibtex: Dict[int, List[Dict[str, Any]]] = {}
    version_meta: Dict[int, Dict[str, Any]] = {}

    for v_num in version_nums:
        version_dir = paper_dir / f"v{v_num}"
        if not version_dir.exists():
            version_meta[v_num] = {
                "parsed": False,
                "reason": "version_dir_not_found",
                "element_count": 0,
                "bib_count": 0,
            }
            continue

        # Check for non-latex source and filter out binary/PDF files disguised as .tex
        non_binary_tex = []
        for tf in version_dir.glob("*.tex"):
            try:
                with tf.open("rb") as bf:
                    header = bf.read(1024)
                if not (header.startswith(b"%PDF-") or b"\x00" in header):
                    non_binary_tex.append(tf)
            except OSError:
                pass

        if (version_dir / ".no_latex").exists() or not non_binary_tex:
            version_meta[v_num] = {
                "parsed": False,
                "reason": "no_latex_sources",
                "element_count": 0,
                "bib_count": 0,
            }
            continue

        # Find compilation root
        root_tex = find_root_tex(version_dir)
        if root_tex is None:
            # Fallback to largest non-binary .tex file
            all_tex = sorted(
                non_binary_tex,
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            if all_tex:
                root_tex = all_tex[0]

        if root_tex is None:
            version_meta[v_num] = {
                "parsed": False,
                "reason": "no_root_tex",
                "element_count": 0,
                "bib_count": 0,
            }
            continue

        # Gather included files
        tex_files = gather_tex_files(root_tex, version_dir)

        # Build hierarchical document tree
        try:
            tree = build_hierarchy(tex_files, arxiv_id, v_num)
            version_trees[v_num] = tree
            elem_count = len(flatten_tree(tree))
        except Exception as exc:
            version_meta[v_num] = {
                "parsed": False,
                "reason": f"hierarchy_build_error: {exc}",
                "element_count": 0,
                "bib_count": 0,
            }
            continue

        # Collect cited keys across all .tex files in version_dir to filter huge .bib files
        cited_keys: Set[str] = set()
        for tf in version_dir.glob("*.tex"):
            try:
                tf_text = tf.read_text(encoding="utf-8", errors="replace")
                cited_keys.update(extract_citation_keys(tf_text))
            except OSError:
                pass

        # Extract references (from tex and .bib files)
        bib_entries: List[Dict[str, Any]] = []
        for tf in tex_files:
            try:
                tf_text = tf.read_text(encoding="utf-8", errors="replace")
                bib_entries.extend(extract_bibitems_from_tex(tf_text))
            except OSError:
                pass

        for bf in version_dir.glob("*.bib"):
            bib_entries.extend(parse_bib_file(bf, filter_keys=cited_keys))

                # Fallback: references that only exist as a compiled .bbl
        if not bib_entries:
            for bbl in version_dir.glob("*.bbl"):
                try:
                    bib_entries.extend(
                        extract_bibitems_from_tex(
                            bbl.read_text(encoding="utf-8", errors="replace")
                        )
                    )
                except OSError:
                    pass

        version_bibtex[v_num] = bib_entries
        version_meta[v_num] = {
            "parsed": True,
            "element_count": elem_count,
            "bib_count": len(bib_entries),
        }

    # Deduplicate across versions
    if version_trees:
        elements_rows, hierarchy_rows = deduplicate_elements(version_trees, arxiv_id)
        bibtex_rows, _ = deduplicate_bibtex(version_bibtex, arxiv_id)
    else:
        elements_rows = []
        hierarchy_rows = []
        bibtex_rows, _ = deduplicate_bibtex(version_bibtex, arxiv_id) if version_bibtex else ([], {})

    save_publication(
        arxiv_id=arxiv_id,
        elements_rows=elements_rows,
        hierarchy_rows=hierarchy_rows,
        bibtex_rows=bibtex_rows,
        version_meta=version_meta,
        output_dir=intermediate_dir,
    )

    return {
        "arxiv_id": arxiv_id,
        "status": "success",
        "elements_count": len(elements_rows),
        "hierarchy_count": len(hierarchy_rows),
        "bibtex_count": len(bibtex_rows),
    }


def parse_corpus(
    arxiv_ids: List[str],
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    intermediate_dir: Path = DEFAULT_INTERMEDIATE_DIR,
    force: bool = False,
) -> Dict[str, Any]:
    """Execute parsing across an entire corpus of papers.

    Args:
        arxiv_ids: List of base arXiv IDs.
        corpus_dir: Directory containing raw paper files.
        intermediate_dir: Directory to save intermediate stores.
        force: If True, re-parse existing entries.

    Returns:
        Summary dict containing execution statistics.
    """
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    total = len(arxiv_ids)
    completed = 0
    cached = 0
    failed = 0
    total_elements = 0
    total_hierarchy = 0
    total_bibtex = 0

    pbar = tqdm(arxiv_ids, desc="Parsing LaTeX Corpus", unit="paper")
    for aid in pbar:
        try:
            res = parse_publication(
                aid,
                corpus_dir=corpus_dir,
                intermediate_dir=intermediate_dir,
                force=force,
            )
            if res.get("status") == "cached":
                cached += 1
            else:
                completed += 1

            total_elements += res.get("elements_count", 0)
            total_hierarchy += res.get("hierarchy_count", 0)
            total_bibtex += res.get("bibtex_count", 0)

        except Exception:
            failed += 1

        pbar.set_postfix({
            "completed": completed,
            "cached": cached,
            "failed": failed,
        })

    return {
        "total_papers": total,
        "completed_papers": completed,
        "cached_papers": cached,
        "failed_papers": failed,
        "total_elements": total_elements,
        "total_hierarchy": total_hierarchy,
        "total_bibtex": total_bibtex,
    }
