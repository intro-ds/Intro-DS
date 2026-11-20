"""Deduplicator — cross-version deduplication of elements and BibTeX entries.

Element deduplication: elements with identical cleaned content share one element_id.
BibTeX deduplication: entries with the same underlying reference are merged, fields unioned.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from src.latexParser.hierarchy_builder import Element, KIND_DOCUMENT, flatten_tree, tree_to_hierarchy_rows


# ---------------------------------------------------------------------------
# Element deduplication
# ---------------------------------------------------------------------------


def _content_hash(kind: str, content: str) -> str:
    """Compute a stable hash for an element based on kind and content."""
    normalized = kind + "::" + re.sub(r"\s+", " ", content).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def deduplicate_elements(
    version_trees: Dict[int, Element],
    arxiv_id: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Deduplicate elements across versions of one publication.

    Algorithm:
    1. Flatten each version tree to a list of elements.
    2. For each element, compute a content hash (kind + content, normalized).
    3. If two elements from different versions share the same hash, assign the
       earlier version's element_id to both.
    4. Build hierarchy rows using the canonical element_ids.

    Args:
        version_trees: Mapping from version number to its root Element.
        arxiv_id: The arXiv ID of the publication.

    Returns:
        Tuple of:
            elements_rows: List of dicts for the elements table (deduplicated).
            hierarchy_rows: List of dicts for the hierarchy table (all versions).
    """
    # Map: content_hash → canonical element_id
    hash_to_id: Dict[str, str] = {}
    # Map: original element_id → canonical element_id
    id_remap: Dict[str, str] = {}
    # Collect unique elements (by hash)
    seen_hashes: Dict[str, Dict] = {}  # hash → element dict for the elements table

    # Process versions in order so lower version numbers get priority
    for version in sorted(version_trees.keys()):
        root = version_trees[version]
        all_elems = flatten_tree(root)
        for elem in all_elems:
            h = _content_hash(elem.kind, elem.content)
            if h in hash_to_id:
                # Duplicate — remap this element's id to the canonical one
                id_remap[elem.element_id] = hash_to_id[h]
            else:
                # New element
                hash_to_id[h] = elem.element_id
                id_remap[elem.element_id] = elem.element_id
                seen_hashes[h] = {
                    "element_id": elem.element_id,
                    "arxiv_id": arxiv_id,
                    "kind": elem.kind,
                    "content": elem.content,
                }

    elements_rows = list(seen_hashes.values())

    # Build hierarchy rows with remapped IDs
    hierarchy_rows: List[Dict] = []
    for version in sorted(version_trees.keys()):
        root = version_trees[version]
        raw_rows = tree_to_hierarchy_rows(root, arxiv_id, version, parent_id=None)
        for row in raw_rows:
            hierarchy_rows.append({
                "arxiv_id": row["arxiv_id"],
                "version": row["version"],
                "element_id": id_remap.get(row["element_id"], row["element_id"]),
                "parent_id": id_remap.get(row["parent_id"], row["parent_id"]),
            })

    return elements_rows, hierarchy_rows


# ---------------------------------------------------------------------------
# BibTeX deduplication
# ---------------------------------------------------------------------------


def _normalize_title(title: Optional[str]) -> str:
    """Normalize a title for comparison purposes."""
    if not title:
        return ""
    return re.sub(r"\s+", " ", title.lower().strip())


def _normalize_author(author: str) -> str:
    """Normalize an author name for comparison."""
    # Keep only letters and spaces, lowercase
    return re.sub(r"[^a-z ]", "", author.lower()).strip()


def _entries_are_duplicates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Return True if two BibTeX entries refer to the same underlying reference."""
    # Same title (normalized)
    title_a = _normalize_title(a.get("title"))
    title_b = _normalize_title(b.get("title"))
    if title_a and title_b and title_a == title_b:
        return True

    # Same first author + same year (if title is missing)
    authors_a = a.get("authors") or []
    authors_b = b.get("authors") or []
    if authors_a and authors_b:
        first_a = _normalize_author(authors_a[0])
        first_b = _normalize_author(authors_b[0])
        year_a = a.get("year")
        year_b = b.get("year")
        if first_a and first_b and first_a == first_b and year_a and year_b and year_a == year_b:
            # Also require at least partial title match if available
            if not title_a or not title_b:
                return True
            # Fuzzy: one title contains the other (truncated version)
            if title_a in title_b or title_b in title_a:
                return True

    return False


def _union_fields(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Union fields from multiple duplicate entries, taking first non-null value for each field."""
    result = dict(entries[0])  # start with the canonical entry

    for entry in entries[1:]:
        for key, val in entry.items():
            if key == "citation_key":
                continue  # keep canonical key
            current = result.get(key)
            if not current and val:
                result[key] = val
            elif isinstance(current, list) and isinstance(val, list) and not current and val:
                result[key] = val

    return result


def deduplicate_bibtex(
    version_bibtex: Dict[int, List[Dict[str, Any]]],
    arxiv_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Deduplicate BibTeX reference entries across versions.

    Algorithm:
    1. Collect all entries from all versions (in version order).
    2. Group entries that refer to the same underlying reference.
    3. Choose canonical key (from earliest version containing that entry).
    4. Union fields from all duplicate entries.
    5. Record key renames: old_key → canonical_key.

    Args:
        version_bibtex: Mapping from version number to list of bibtex dicts.
        arxiv_id: The arXiv ID.

    Returns:
        Tuple of:
            bibtex_rows: Deduplicated list of bibtex dicts for the bibtex table.
            key_renames: Dict mapping old citation keys to canonical keys.
    """
    # Collect all entries in version order
    all_entries: List[Dict[str, Any]] = []
    for version in sorted(version_bibtex.keys()):
        for entry in version_bibtex[version]:
            all_entries.append(dict(entry, arxiv_id=arxiv_id))

    # Cluster duplicates using union-find logic
    # cluster_id[i] = index of the canonical entry for entry i
    n = len(all_entries)
    cluster: List[int] = list(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if cluster[j] == j:  # j is not yet assigned
                if _entries_are_duplicates(all_entries[i], all_entries[j]):
                    cluster[j] = cluster[i]

    # Build groups: canonical_index → [entries]
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for i, canon_idx in enumerate(cluster):
        groups.setdefault(canon_idx, []).append(all_entries[i])

    # Build key rename map and deduplicated entries
    key_renames: Dict[str, str] = {}
    bibtex_rows: List[Dict[str, Any]] = []

    for canon_idx, group_entries in groups.items():
        canonical_key = group_entries[0]["citation_key"]
        merged = _union_fields(group_entries)
        merged["citation_key"] = canonical_key
        merged["arxiv_id"] = arxiv_id

        for entry in group_entries:
            old_key = entry["citation_key"]
            if old_key != canonical_key:
                key_renames[old_key] = canonical_key

        bibtex_rows.append(merged)

    return bibtex_rows, key_renames
