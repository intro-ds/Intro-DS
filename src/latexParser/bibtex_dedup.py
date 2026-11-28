import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Tuple


_CITE_COMMAND = re.compile(
    r"(?<!\\)\\cite[a-zA-Z]*\*?(?:\s*\[[^\]]*\])*\s*\{([^{}]*)\}"
)
_DOI_PREFIX = re.compile(r"^(?:(?:https?://)?(?:dx\.)?doi\.org/|doi:\s*)", re.I)


def _normalize_doi(value: object) -> str:
    doi = str(value or "").strip()
    doi = _DOI_PREFIX.sub("", doi)
    return doi.rstrip(".,; ").casefold()


def _normalize_title(value: object) -> str:
    title = unicodedata.normalize("NFKC", str(value or ""))
    title = re.sub(r"\\[a-zA-Z]+\*?", " ", title)
    title = re.sub(r"[{}]", " ", title)
    title = unicodedata.normalize("NFKD", title).casefold()
    return re.sub(r"[\W_]+", "", title, flags=re.UNICODE)


def _title_year(entry: Mapping[str, object]) -> Tuple[str, str]:
    title = _normalize_title(entry.get("title"))
    year = str(entry.get("year") or "").strip().casefold()
    return title, year


def dedup_bibtex(
    entries_by_version: Mapping[int, Iterable[Mapping[str, object]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Unify BibTeX entries across versions and return entries plus old-to-canonical keys.

    Versions are processed in ascending order, preserving entry order within each
    version. The first key in a merged group is canonical. DOI is the strongest
    match: equal non-empty DOIs always match, while different non-empty DOIs never
    match. If either entry has no DOI, normalized title plus year is a fallback;
    a DOI-less entry can join a DOI-bearing group only when that group is the
    unique compatible match.

    Fields are unioned. For conflicts, the non-empty value with the most
    non-whitespace characters wins; ties keep the value from the earliest
    version, then earliest entry order. `citation_key` is always set to the
    group's canonical key.

    A reused citation key always joins its existing group unless the DOIs conflict, 
    in which case the later entry gets a unique key such as key_2.
    """
    groups: List[Dict[str, Any]] = []
    key_map: Dict[str, str] = {}

    for version in sorted(entries_by_version):
        for entry in entries_by_version[version]:
            citation_key = str(entry.get("citation_key") or "").strip()
            if not citation_key:
                raise ValueError("Each BibTeX entry must have a non-empty citation_key")

            doi = _normalize_doi(entry.get("doi"))
            title_year = _title_year(entry)
            compatible: List[Dict[str, Any]] = []

            for candidate in groups:
                group_doi = candidate["doi"]
                if doi and group_doi:
                    if doi == group_doi:
                        compatible.append(candidate)
                    continue
                same_key = citation_key in candidate["keys"]
                same_title_year = bool(
                    title_year[0] and title_year[1] and title_year == candidate["title_year"]
                )
                if same_key or same_title_year:
                    compatible.append(candidate)

            if len(compatible) > 1:
                # Ambiguous: only trust a unique group that already used this key
                same_key_groups = [g for g in compatible if citation_key in g["keys"]]
                compatible = same_key_groups if len(same_key_groups) == 1 else []
            group = compatible[0] if compatible else None

            if group is None:
                used = {str(g["canonical_key"]) for g in groups}
                canonical_key, suffix = citation_key, 2
                while canonical_key in used:
                    canonical_key, suffix = f"{citation_key}_{suffix}", suffix + 1
                group = {
                    "canonical_key": canonical_key,
                    "keys": set(),
                    "doi": doi,
                    "title_year": title_year,
                    "fields": {},
                    "field_priority": {},
                }
                groups.append(group)

            group["keys"].add(citation_key)
            if doi and not group["doi"]:
                group["doi"] = doi

            fields, priorities = group["fields"], group["field_priority"]
            for field, raw_value in entry.items():
                if field == "citation_key" or raw_value is None:
                    continue
                if isinstance(raw_value, (list, tuple)):  # e.g. authors
                    value: Any = [str(v).strip() for v in raw_value if str(v).strip()]
                    size = sum(len(v) for v in value)
                else:
                    value = str(raw_value).strip()
                    size = len(value)
                if not value:
                    continue
                priority = (size, -version)
                if field not in fields or priority > priorities[field]:
                    fields[field] = value
                    priorities[field] = priority
            group["title_year"] = _title_year(fields)
            key_map.setdefault(citation_key, str(group["canonical_key"]))

    unified: List[Dict[str, Any]] = []
    for group in groups:
        unified_entry = dict(group["fields"])
        unified_entry["citation_key"] = str(group["canonical_key"])
        unified.append(unified_entry)

    return unified, key_map


def rewrite_cites(text: str, key_mapping: Mapping[str, str]) -> str:
    """Rewrite citation keys in `\\cite...{...}` commands using `key_mapping`.

    Comma-separated keys, optional citation arguments, and all cite command
    variants are retained. Chained mappings are resolved to their final key;
    mapping cycles raise `ValueError`.
    """

    def canonical_key(key: str) -> str:
        current = key
        seen = set()
        while current in key_mapping and key_mapping[current] != current:
            if current in seen:
                raise ValueError(f"Cyclic citation-key mapping involving {current!r}")
            seen.add(current)
            current = key_mapping[current]
        return current

    def replace_match(match: re.Match[str]) -> str:
        content = match.group(1)
        rewritten_keys = []
        for token in content.split(","):
            key = token.strip()
            if not key:
                rewritten_keys.append(token)
                continue
            leading = token[: len(token) - len(token.lstrip())]
            trailing = token[len(token.rstrip()) :]
            rewritten_keys.append(f"{leading}{canonical_key(key)}{trailing}")
        start = match.start(1) - match.start()
        end = match.end(1) - match.start()
        return match.group(0)[:start] + ",".join(rewritten_keys) + match.group(0)[end:]

    return _CITE_COMMAND.sub(replace_match, text)
