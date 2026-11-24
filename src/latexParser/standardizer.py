"""Standardizer — cleans LaTeX markup and normalizes math within element content.

Also converts \\bibitem entries to structured BibTeX records.
"""

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Accent / special character expansion table
# ---------------------------------------------------------------------------
_ACCENT_MAP: Dict[str, str] = {
    r"\'a": "á", r"\'e": "é", r"\'i": "í", r"\'o": "ó", r"\'u": "ú",
    r"\'A": "Á", r"\'E": "É", r"\'I": "Í", r"\'O": "Ó", r"\'U": "Ú",
    r"\`a": "à", r"\`e": "è", r"\`i": "ì", r"\`o": "ò", r"\`u": "ù",
    r"\`A": "À", r"\`E": "È", r"\`I": "Ì", r"\`O": "Ò", r"\`U": "Ù",
    r"\^a": "â", r"\^e": "ê", r"\^i": "î", r"\^o": "ô", r"\^u": "û",
    r"\^A": "Â", r"\^E": "Ê", r"\^I": "Î", r"\^O": "Ô", r"\^U": "Û",
    r'\"a': "ä", r'\"e': "ë", r'\"i': "ï", r'\"o': "ö", r'\"u': "ü",
    r'\"A': "Ä", r'\"E': "Ë", r'\"I': "Ï", r'\"O': "Ö", r'\"U': "Ü",
    r"\~n": "ñ", r"\~N": "Ñ", r"\~a": "ã", r"\~o": "õ",
    r"\c{c}": "ç", r"\c{C}": "Ç",
    r"\ss": "ß",
    r"\ae": "æ", r"\AE": "Æ",
    r"\oe": "œ", r"\OE": "Œ",
    r"\o": "ø", r"\O": "Ø",
    r"\aa": "å", r"\AA": "Å",
    r"\l": "ł", r"\L": "Ł",
}

# Accent commands: \'e, \'{e}, \"o, \~n ...
_ACCENT_RE = re.compile(r"\\(['`^\"~])\s*\{?([A-Za-z])\}?")
_CEDILLA_RE = re.compile(r"\\c\s*\{([cC])\}")
# Letter-named commands (\ss, \ae, \o, \l ...) must not match a longer command
_LETTER_MAP: Dict[str, str] = {
    k[1:]: v for k, v in _ACCENT_MAP.items() if k[1:].isalpha()
}
_LETTER_RE = re.compile(
    r"\\(" + "|".join(sorted(_LETTER_MAP, key=len, reverse=True)) + r")(?![A-Za-z])(?:\{\})?"
)

# LaTeX formatting commands that wrap an argument: \cmd{arg} → arg
_FORMATTING_CMDS = (
    "textbf", "textit", "texttt", "textrm", "textsc", "textsf",
    "emph", "underline", "uline", "overline",
    "text", "mathrm", "mathbf", "mathit", "mathsf", "mathtt",
    "boldsymbol", "bm",
    "mbox", "hbox", "vbox",
    "tiny", "scriptsize", "footnotesize", "small", "normalsize",
    "large", "Large", "LARGE", "huge", "Huge",
)

# Commands that should be stripped entirely (command + optional argument)
_DROP_CMDS_NO_ARG = (
    "centering", "raggedright", "raggedleft", "noindent", "newline",
    "hfill", "vfill", "hspace", "vspace", "medskip", "bigskip", "smallskip",
    "midrule", "toprule", "bottomrule", "hline",
    "pagebreak", "newpage", "clearpage",
    "nonumber", "notag",
)

# ---------------------------------------------------------------------------
# Comment stripping
# ---------------------------------------------------------------------------

def strip_comments(source: str) -> str:
    """Remove LaTeX % comments, preserving newlines."""
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


# ---------------------------------------------------------------------------
# Accent expansion
# ---------------------------------------------------------------------------

def expand_accents(text: str) -> str:
    """Expand LaTeX accent commands to Unicode characters.

    Letter-named commands such as \\o or \\l are only expanded when they are
    not the prefix of a longer command (e.g. \\label, \\linewidth, \\LaTeX).
    """
    text = _ACCENT_RE.sub(
        lambda m: _ACCENT_MAP.get("\\" + m.group(1) + m.group(2), m.group(0)),
        text,
    )
    text = _CEDILLA_RE.sub(lambda m: "ç" if m.group(1) == "c" else "Ç", text)
    return _LETTER_RE.sub(lambda m: _LETTER_MAP[m.group(1)], text)


# ---------------------------------------------------------------------------
# Math normalization
# ---------------------------------------------------------------------------

def normalize_inline_math(text: str) -> str:
    r"""Convert \(...\) and \begin{math}...\end{math} to $...$."""
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    text = re.sub(
        r"\\begin\{math\}(.+?)\\end\{math\}",
        r"$\1$",
        text,
        flags=re.DOTALL,
    )
    return text


def normalize_block_math(text: str) -> str:
    r"""Convert \[...\], $$...$$, \begin{align}...\end{align}, etc. to \begin{equation}...\end{equation}."""
    # \[...\]
    text = re.sub(
        r"\\\[(.+?)\\\]",
        r"\\begin{equation}\1\\end{equation}",
        text,
        flags=re.DOTALL,
    )
    # $$...$$
    text = re.sub(
        r"\$\$(.+?)\$\$",
        r"\\begin{equation}\1\\end{equation}",
        text,
        flags=re.DOTALL,
    )
    # \begin{align/align*/eqnarray/multline/gather}
    for env in ("align", "align\\*", "eqnarray", "eqnarray\\*", "multline", "gather", "gather\\*"):
        text = re.sub(
            rf"\\begin\{{{env}\}}(.+?)\\end\{{{env}\}}",
            r"\\begin{equation}\1\\end{equation}",
            text,
            flags=re.DOTALL,
        )
    return text


# ---------------------------------------------------------------------------
# General LaTeX cleanup
# ---------------------------------------------------------------------------

def _strip_formatting_commands(text: str) -> str:
    r"""Replace \cmd{content} → content for known formatting commands."""
    for cmd in _FORMATTING_CMDS:
        # Match \cmd{...} with balanced braces (simple one-level)
        pattern = rf"\\{re.escape(cmd)}\{{([^{{}}]*)\}}"
        text = re.sub(pattern, r"\1", text)
    return text


def _drop_no_arg_commands(text: str) -> str:
    """Remove commands that don't contribute semantic meaning."""
    for cmd in _DROP_CMDS_NO_ARG:
        # With optional argument: \cmd[...]{...}
        text = re.sub(
            rf"\\{re.escape(cmd)}(?:\[[^\]]*\])?(?:\{{[^}}]*\}})?",
            " ",
            text,
        )
        text = re.sub(rf"\\{re.escape(cmd)}\b", " ", text)
    return text


def _remove_labels_refs(text: str) -> str:
    """Remove \\label{}, \\ref{}, \\eqref{}, \\autoref{}."""
    text = re.sub(r"\\(?:label|ref|eqref|autoref|pageref|cref|Cref)\{[^}]*\}", "", text)
    return text


def _remove_citations(text: str) -> str:
    """Remove \\cite{}, \\citep{}, \\citet{}, etc. — replace with empty string."""
    text = re.sub(r"\\cite[tp]?\*?\{[^}]*\}", "", text)
    return text


def _normalize_dashes_quotes(text: str) -> str:
    """Normalize LaTeX dashes and quotes to Unicode equivalents."""
    text = text.replace("---", "\u2014")   # em dash
    text = text.replace("--", "\u2013")    # en dash
    text = text.replace("``", "\u201c")    # left double quote
    text = text.replace("''", "\u201d")    # right double quote
    text = text.replace("`", "\u2018")     # left single quote
    return text


def _remove_remaining_commands(text: str) -> str:
    r"""Remove leftover LaTeX commands: \cmd{arg} → arg, \cmd → ''."""
    # \cmd{arg} → arg (keep content)
    text = re.sub(r"\\[a-zA-Z@]+\*?\{([^{}]*)\}", r"\1", text)
    # \cmd → '' (bare commands)
    text = re.sub(r"\\[a-zA-Z@]+\*?\b", "", text)
    # Remaining braces
    text = re.sub(r"[{}]", "", text)
    return text


def clean_element_content(text: str, kind: str = "sentence") -> str:
    """Clean LaTeX markup from element content.

    For sentence/heading/item/document kinds:
    - Strip comments, expand accents, normalize math, strip formatting, clean dashes.

    For formula kind:
    - Normalize block math environment form only; keep raw LaTeX content.

    For figure kind:
    - Return cleaned caption text.

    Args:
        text: Raw LaTeX text of the element.
        kind: Element kind string (sentence, heading, formula, figure, item, document).

    Returns:
        Cleaned string.
    """
    if kind == "formula":
        text = strip_comments(text)
        text = normalize_block_math(text)
        return text.strip()

    text = strip_comments(text)
    text = expand_accents(text)
    text = normalize_inline_math(text)
    text = normalize_block_math(text)
    text = _strip_formatting_commands(text)
    text = _drop_no_arg_commands(text)
    text = _remove_labels_refs(text)
    text = _remove_citations(text)
    text = _normalize_dashes_quotes(text)
    text = _remove_remaining_commands(text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Known abbreviations that contain periods but do not end a sentence
_ABBREVS = (
    r"i\.e", r"e\.g", r"et\s+al", r"Fig", r"Figs", r"Eq", r"Eqs",
    r"Sec", r"Secs", r"Tab", r"Tabs", r"Ref", r"Refs", r"cf", r"vs",
    r"approx", r"Dr", r"Prof", r"Mr", r"Mrs", r"Ms",
)
_ABBREV_RE = re.compile(
    r"\b(?:" + "|".join(_ABBREVS) + r")\.",
    re.IGNORECASE,
)
_INITIALS_RE = re.compile(r"\b[A-Z]\.")

# Decimal / version numbers: 3.14, 1.0.0
_DECIMAL_RE = re.compile(r"\d+\.\d")

# URLs
_URL_RE = re.compile(r"https?://\S+")

# Ellipsis
_ELLIPSIS_RE = re.compile(r"\.\.\.")


def split_sentences(text: str) -> List[str]:
    """Split prose text into sentences at sentence-ending periods.

    Does NOT split at:
    - Periods inside $...$ math spans.
    - Known abbreviations (i.e., et al., Fig., Eq., etc.).
    - Decimal numbers (3.14) or version numbers.
    - URLs.
    - Ellipsis (...).

    Args:
        text: Cleaned prose text (after LaTeX cleanup).

    Returns:
        List of non-empty sentence strings.
    """
    if not text.strip():
        return []

    # Protect math spans: replace $...$ with placeholder
    math_spans: List[str] = []

    def _protect_math(m: re.Match) -> str:  # type: ignore[type-arg]
        math_spans.append(m.group(0))
        return f"MATHSPAN{len(math_spans) - 1}END"

    protected = re.sub(r"\$[^$]*\$", _protect_math, text)

    # Protect URLs
    url_spans: List[str] = []

    def _protect_url(m: re.Match) -> str:  # type: ignore[type-arg]
        url_spans.append(m.group(0))
        return f"URLSPAN{len(url_spans) - 1}END"

    protected = _URL_RE.sub(_protect_url, protected)

    # Protect ellipsis
    protected = _ELLIPSIS_RE.sub("ELLIPSIS", protected)

    # Protect decimals
    protected = _DECIMAL_RE.sub(lambda m: m.group(0).replace(".", "DECIMALDOT"), protected)

    # Protect abbreviations
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", "ABBREVDOT"), protected)
    protected = _INITIALS_RE.sub(lambda m: m.group(0).replace(".", "ABBREVDOT"), protected)

    # Now split at ". " or ".\n" followed by an uppercase letter or end of string
    sentences_raw = re.split(r"\.(?:\s+(?=[A-Z])|\s*$)", protected)

    # Pre-compile placeholder restoration pattern
    restore_re = re.compile(r"(MATHSPAN|URLSPAN)(\d+)END")

    def _restore_match(m: re.Match) -> str:
        kind_tag, idx_str = m.group(1), m.group(2)
        idx = int(idx_str)
        if kind_tag == "MATHSPAN" and idx < len(math_spans):
            return math_spans[idx]
        if kind_tag == "URLSPAN" and idx < len(url_spans):
            return url_spans[idx]
        return m.group(0)

    # Restore protections
    result: List[str] = []
    for s in sentences_raw:
        s = s.replace("ELLIPSIS", "...")
        s = s.replace("DECIMALDOT", ".")
        s = s.replace("ABBREVDOT", ".")
        if "MATHSPAN" in s or "URLSPAN" in s:
            s = restore_re.sub(_restore_match, s)
        s = s.strip()
        if s:
            result.append(s)

    return result


# ---------------------------------------------------------------------------
# BibItem → BibTeX conversion
# ---------------------------------------------------------------------------

# revtex/apsrev .bbl entries wrap fields as \bibinfo {title} {...}
_BIBNAME_RE = re.compile(
    r"((?:\\bibfnamefont\s*\{[^{}]*\}(?:\s|~|\\ )*)+)\\bibnamefont\s*\{([^{}]*)\}"
)


def _parse_revtex_bibitem(raw_text: str) -> Dict[str, Any]:
    """Extract title, journal, year and authors from a revtex-style bibitem."""
    result: Dict[str, Any] = {}
    if "\\bibinfo" not in raw_text:
        return result

    for field in ("title", "journal", "year"):
        m = re.search(
            r"\\bibinfo\s*\{" + field + r"\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}", raw_text
        )
        if m:
            result[field] = clean_element_content(m.group(1), kind="sentence")

    if "year" in result:
        ym = re.search(r"\d{4}", result["year"])
        result["year"] = int(ym.group(0)) if ym else None

    authors: List[str] = []
    for firsts, last in _BIBNAME_RE.findall(raw_text):
        first_names = re.findall(r"\\bibfnamefont\s*\{([^{}]*)\}", firsts)
        authors.append(
            clean_element_content(" ".join(first_names) + " " + last, kind="sentence")
        )
    if authors:
        result["authors"] = authors
    return result

_NEWBLOCK_RE = re.compile(r"\\newblock\b")
# "A.", "A. B.", "F" ... (given-name initials only)
_INITIALS_ONLY_RE = re.compile(r"^(?:[A-Z][a-z]?\.?[\s\-~]*){1,4}$")
# Things that must not be mistaken for a publication year
_URL_DOI_ARXIV_RE = re.compile(
    r"https?://\S+|\b10\.\d{4,}/\S+|arxiv[:\s]*\d{4}\.\d{4,5}(?:v\d+)?",
    re.IGNORECASE,
)
_ARXIV_ID_RE = re.compile(
    r"(?:arxiv[:\s/]*(?:abs/)?|abs/)(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE
)


def _split_authors(text: str) -> List[str]:
    """Split an author string into individual names.

    Handles "A. B. Last, C. Last, and D. Last" and "Last, F., Last, G., and Last, H.".
    A comma-separated list is read as "Last, Initials" pairs only when every
    second token consists of initials; otherwise each token is one author.
    """
    text = re.sub(r"\bet\s+al\.?", "", text, flags=re.IGNORECASE)
    # Strip separators but keep the final period of an initial such as "C."
    text = text.strip(" ,;(")
    if text.endswith(".") and not re.search(r"(?:^|[\s\-~])[A-Z][a-z]?\.$", text):
        text = text[:-1]
    
    pieces = re.split(r"\s*,?\s+and\s+|\s*&\s*", text)
    authors: List[str] = []
    for piece in pieces:
        toks = [t.strip() for t in piece.split(",") if t.strip()]
        if (
            len(toks) >= 2
            and len(toks) % 2 == 0
            and all(_INITIALS_ONLY_RE.match(toks[i]) for i in range(1, len(toks), 2))
        ):
            authors += [f"{toks[i + 1]} {toks[i]}" for i in range(0, len(toks), 2)]
        else:
            authors += toks
    return [a for a in authors if len(a) > 2]


def _infer_entry_type(cleaned: str, venue: Optional[str]) -> str:
    """Guess the BibTeX entry type.

    Rules, first match wins:
    1. mentions arXiv or "preprint"                    -> misc
    2. venue starts with "In" or mentions proceedings,
       conference, workshop, symposium                 -> inproceedings
    3. mentions journal, transactions, Rev., Lett.     -> article
    4. mentions ISBN, press, publisher, Springer, Wiley -> book
    5. otherwise                                        -> misc
    """
    low = cleaned.lower()
    if re.search(r"arxiv|preprint", low):
        return "misc"
    if (venue and re.match(r"in\b", venue, re.IGNORECASE)) or re.search(
        r"proceedings|proc\.|conference|workshop|symposium", low
    ):
        return "inproceedings"
    if re.search(r"journal|transactions|trans\.|rev\.|lett\.|letters", low):
        return "article"
    if re.search(r"isbn|\bpress\b|publisher|springer|wiley", low):
        return "book"
    return "misc"

def bibitem_to_bibtex(key: str, raw_text: str) -> Dict[str, Any]:
    """Convert a \\bibitem{key} raw text to a structured BibTeX record.

    Strategy, first applicable wins:
    1. revtex/apsrev \\bibinfo fields.
    2. natbib/plain style split by \\newblock: authors / title / venue.
    3. Quoted title: authors before the quote, venue after it.
    4. APA style "Authors (2017). Title. Venue.".
    5. Fallback: authors are everything before the year.
    The entry type is guessed by _infer_entry_type.
    """
    revtex = _parse_revtex_bibitem(raw_text)
    cleaned = clean_element_content(raw_text, kind="sentence")

    # Split by \newblock BEFORE cleaning, because cleaning deletes \newblock
    blocks = [
        clean_element_content(b, kind="sentence").strip(" .,\n")
        for b in _NEWBLOCK_RE.split(raw_text)
    ]
    blocks = [b for b in blocks if b]

    scrub = _URL_DOI_ARXIV_RE.sub(" ", cleaned)
    year_match = re.search(r"\b(?:19|20)\d{2}\b", scrub)
    year: Optional[int] = int(year_match.group(0)) if year_match else None

    doi: Optional[str] = None
    doi_match = re.search(r"\b10\.\d{4,}/\S+", cleaned)
    if doi_match:
        doi = doi_match.group(0).rstrip(".,)")

    url: Optional[str] = None
    url_match = re.search(r"https?://\S+", raw_text)
    if url_match:
        url = url_match.group(0).rstrip(".,)}")
    if url is None:
        arxiv_match = _ARXIV_ID_RE.search(cleaned)
        if arxiv_match:
            url = f"https://arxiv.org/abs/{arxiv_match.group(1)}"

    title: Optional[str] = None
    authors_text = ""
    venue: Optional[str] = None
    title_match = re.search(r'["\u201c]([^"\u201d]{5,})["\u201d]', cleaned)

    if len(blocks) >= 2:
        authors_text, title = blocks[0], blocks[1]
        venue = blocks[2] if len(blocks) > 2 else None
    elif title_match:
        title = title_match.group(1).strip(" ,.")
        authors_text = cleaned[: title_match.start()]
        venue = cleaned[title_match.end():].lstrip(" ,.")
    elif year_match and scrub[max(0, year_match.start() - 1):year_match.start()] == "(":
        authors_text = scrub[: year_match.start()]
        sentences = [
            s.strip()
            for s in re.split(r"\.\s+|\.$", scrub[year_match.end():].lstrip(").,: "))
            if s.strip()
        ]
        title = sentences[0] if sentences else None
        venue = sentences[1] if len(sentences) > 1 else None
    elif year_match:
        authors_text = scrub[: year_match.start()]
        venue = scrub[year_match.end():].lstrip(",. ")

    journal: Optional[str] = None
    if venue:
        v = re.sub(r"^in\s*:?\s+", "", venue, flags=re.IGNORECASE)
        journal = re.split(r",|\(|\bvol", v)[0].strip(" .")
        if len(journal) < 5 or journal[0].isdigit():
            journal = None

    return {
        "citation_key": key,
        "entry_type": _infer_entry_type(cleaned, venue),
        "title": revtex.get("title") or title,
        "authors": revtex.get("authors") or _split_authors(authors_text),
        "year": revtex.get("year") or year,
        "doi": doi,
        "journal": revtex.get("journal") or journal,
        "url": url,
    }


def extract_bibitems_from_tex(tex_content: str) -> List[Dict[str, Any]]:
    """Extract all \\bibitem entries from a LaTeX document."""
    items: List[Dict[str, Any]] = []
    pattern = re.compile(r"\\bibitem\s*(?:\[(?:[^\[\]]|\[[^\]]*\])*\])?\s*\{([^}]+)\}")
    matches = list(pattern.finditer(tex_content))
    if not matches:
        return items

    for i, m in enumerate(matches):
        key = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(tex_content)
        end_bib = tex_content.find(r"\end{thebibliography}", start)
        if end_bib != -1 and end_bib < end:
            end = end_bib
        raw_text = tex_content[start:end].strip()
        record = bibitem_to_bibtex(key, raw_text)
        items.append(record)
    return items


def extract_citation_keys(tex_content: str) -> List[str]:
    """Extract all citation keys cited via \\cite, \\citep, \\citet, etc."""
    keys: List[str] = []
    pattern = re.compile(
        r"\\(?:cite|citep|citet|citeauthor|citeyear|citealp|citealt|citenum|parencite|textcite|autocite|footcite|nocite|fullcite|citeyearpar)\*?(?:\[[^\]]*\])*\{([^}]+)\}"
    )
    for m in pattern.finditer(tex_content):
        for raw_k in m.group(1).split(","):
            k = raw_k.strip()
            if k and k != "*":
                keys.append(k)
    return keys


def parse_bib_file(
    bib_path: Path,
    filter_keys: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Parse BibTeX entries from a .bib file using regular expressions.

    If filter_keys is provided, for large .bib files (> 500 entries or > 500 KB,
    such as IACR crypto.bib or ACL anthology.bib), only cited entries are extracted.
    This prevents database-sized bib files from freezing the parsing pipeline.
    """
    items: List[Dict[str, Any]] = []
    try:
        content = bib_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return items

    content = strip_comments(content)
    entry_re = re.compile(r"@([a-zA-Z]+)\s*\{\s*([^,]+),", re.MULTILINE)
    matches = list(entry_re.finditer(content))
    if not matches:
        return items

    # If the file is large (> 500 KB or > 500 entries) and filter_keys is provided,
    # filter to only parse cited entries to prevent massive database files from hanging.
    should_filter = bool(filter_keys) and (len(matches) > 500 or bib_path.stat().st_size > 500_000)
    filter_keys_lower = {k.lower() for k in filter_keys} if filter_keys else set()

    for i, m in enumerate(matches):
        entry_type = m.group(1).lower()
        if entry_type in ("string", "comment", "preamble", "control"):
            continue

        key = m.group(2).strip()
        if should_filter and key.lower() not in filter_keys_lower:
            continue

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].rstrip(" \t\n\r}")

        field_re = re.compile(
            r"([a-zA-Z_]+)\s*=\s*(?:\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}|\"([^\"]*)\"|([^\s,]+))",
            re.MULTILINE | re.DOTALL,
        )
        fields: Dict[str, str] = {}
        for fm in field_re.finditer(body):
            f_name = fm.group(1).lower().strip()
            val = fm.group(2) if fm.group(2) is not None else (
                fm.group(3) if fm.group(3) is not None else fm.group(4)
            )
            if val is not None:
                cleaned_val = clean_element_content(val, kind="sentence")
                fields[f_name] = cleaned_val

        year: Optional[int] = None
        if "year" in fields:
            ym = re.search(r"\b(19|20)\d{2}\b", fields["year"])
            if ym:
                year = int(ym.group(0))

        authors: List[str] = []
        if "author" in fields:
            raw_authors = re.split(r"\s+and\s+", fields["author"])
            authors = [
                clean_element_content(a, kind="sentence").strip()
                for a in raw_authors
                if a.strip()
            ]

        items.append({
            "citation_key": key,
            "entry_type": entry_type,
            "title": fields.get("title"),
            "authors": authors,
            "year": year,
            "doi": fields.get("doi"),
            "journal": fields.get("journal") or fields.get("booktitle"),
            "url": fields.get("url"),
        })

    return items
