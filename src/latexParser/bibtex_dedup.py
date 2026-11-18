import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Dict, List, Tuple


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
) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
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
    """
    groups: List[Dict[str, object]] = []
    key_map: Dict[str, str] = {}

    for version in sorted(entries_by_version):
        for entry in entries_by_version[version]:
            citation_key = str(entry.get("citation_key") or "").strip()
            if not citation_key:
                raise ValueError("Each BibTeX entry must have a non-empty citation_key")

            doi = _normalize_doi(entry.get("doi"))
            title_year = _title_year(entry)
            compatible_groups: List[Dict[str, object]] = []

            for group in groups:
                group_doi = group["doi"]
                if doi and group_doi:
                    if doi == group_doi:
                        compatible_groups.append(group)
                    continue
                if title_year[0] and title_year[1] and title_year == group["title_year"]:
                    compatible_groups.append(group)

            if len(compatible_groups) > 1:
                # Ambiguous title/year matches must not collapse entries carrying
                # distinct DOI identities.
                group = None
            else:
                group = compatible_groups[0] if compatible_groups else None

            if group is None:
                if any(
                    existing["canonical_key"] == citation_key for existing in groups
                ):
                    raise ValueError(
                        f"Citation key {citation_key!r} identifies distinct references"
                    )
                group = {
                    "canonical_key": citation_key,
                    "doi": doi,
                    "title_year": title_year,
                    "fields": {},
                    "field_priority": {},
                }
                groups.append(group)

            mapped_key = key_map.get(citation_key)
            canonical_key = str(group["canonical_key"])
            if mapped_key is not None and mapped_key != canonical_key:
                raise ValueError(
                    f"Citation key {citation_key!r} identifies distinct references"
                )

            if doi and not group["doi"]:
                group["doi"] = doi
            fields = group["fields"]
            priorities = group["field_priority"]
            for field, raw_value in entry.items():
                if field == "citation_key" or raw_value is None:
                    continue
                value = str(raw_value).strip()
                if not value:
                    continue
                priority = (len(value), -version)
                if field not in fields or priority > priorities[field]:
                    fields[field] = value
                    priorities[field] = priority
            group["title_year"] = _title_year(fields)

            key_map[citation_key] = canonical_key

    unified: List[Dict[str, str]] = []
    for group in groups:
        canonical_key = str(group["canonical_key"])
        unified_entry = dict(group["fields"])
        unified_entry["citation_key"] = canonical_key
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
