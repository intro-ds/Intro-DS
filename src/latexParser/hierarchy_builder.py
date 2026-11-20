"""Hierarchy builder — builds the document tree from concatenated .tex source.

Parses the LaTeX content into a tree of Elements where:
- Document is the root
- Chapters/Sections/Subsections/etc. form intermediate nodes
- Sentences, block formulas, and figures/tables are leaf nodes
- itemize/enumerate blocks branch into item children
- References section is excluded; Acknowledgements and Appendices are included
"""

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.latexParser.standardizer import clean_element_content, split_sentences, strip_comments


# ---------------------------------------------------------------------------
# Element kinds
# ---------------------------------------------------------------------------

KIND_DOCUMENT = "document"
KIND_HEADING = "heading"
KIND_SENTENCE = "sentence"
KIND_FORMULA = "formula"
KIND_FIGURE = "figure"
KIND_ITEMIZE = "itemize"
KIND_ITEM = "item"


# ---------------------------------------------------------------------------
# Heading levels
# ---------------------------------------------------------------------------

HEADING_CMDS: List[Tuple[str, int]] = [
    ("chapter", 1),
    ("section", 2),
    ("subsection", 3),
    ("subsubsection", 4),
    ("paragraph", 5),
    ("subparagraph", 6),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Element:
    """A node in the document hierarchy."""

    element_id: str
    kind: str
    content: str
    children: List["Element"] = field(default_factory=list)

    def to_dict(self, arxiv_id: str) -> Dict:
        """Return the row dict for the elements table."""
        return {
            "element_id": self.element_id,
            "arxiv_id": arxiv_id,
            "kind": self.kind,
            "content": self.content,
        }


# ---------------------------------------------------------------------------
# ID generator
# ---------------------------------------------------------------------------

class _IdGen:
    """Generates sequential element IDs of the form '<arxiv_id>#<seq>'."""

    def __init__(self, arxiv_id: str) -> None:
        self._prefix = arxiv_id.replace(".", "-") + "#"
        self._counter = 0

    def next(self) -> str:
        self._counter += 1
        return f"{self._prefix}{self._counter}"


# ---------------------------------------------------------------------------
# LaTeX block detection helpers
# ---------------------------------------------------------------------------

def _extract_command_arg(text: str, start: int) -> Tuple[str, int]:
    """Extract the content of {...} starting at or after position start.

    Returns (content, end_pos) where end_pos is after the closing brace.
    """
    # skip optional whitespace and optional [...]
    pos = start
    while pos < len(text) and text[pos] in " \t\n":
        pos += 1
    if pos < len(text) and text[pos] == "[":
        depth = 1
        pos += 1
        while pos < len(text) and depth > 0:
            if text[pos] == "[":
                depth += 1
            elif text[pos] == "]":
                depth -= 1
            pos += 1
    if pos < len(text) and text[pos] == "{":
        depth = 1
        pos += 1
        inner_start = pos
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        return text[inner_start : pos - 1], pos
    return "", start


# Regex to find section-type commands
_HEADING_RE = re.compile(
    r"\\(chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\s*\{",
    re.IGNORECASE,
)

# Regex for block math environments
_BLOCK_MATH_BEGIN = re.compile(
    r"\\begin\{(equation|align|align\*|eqnarray|eqnarray\*|multline|gather|gather\*|displaymath)\}"
)
_BLOCK_MATH_END = {
    "equation": r"\end{equation}",
    "align": r"\end{align}",
    "align*": r"\end{align*}",
    "eqnarray": r"\end{eqnarray}",
    "eqnarray*": r"\end{eqnarray*}",
    "multline": r"\end{multline}",
    "gather": r"\end{gather}",
    "gather*": r"\end{gather*}",
    "displaymath": r"\end{displaymath}",
}

# Display math: \[...\] and $$...$$
_DISPLAY_MATH_BRACKET = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_DISPLAY_MATH_DOLLAR = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

# Figure / table environments
_FIGURE_BEGIN = re.compile(r"\\begin\{(figure|table)(?:\*|)\}")
_FIGURE_CAPTION = re.compile(r"\\caption\{")

# Itemize / enumerate / description
_LIST_BEGIN = re.compile(r"\\begin\{(itemize|enumerate|description)\}")
_LIST_END_RE = re.compile(r"\\end\{(?:itemize|enumerate|description)\}")
_ITEM_RE = re.compile(r"\\item\b")

# References section markers
_REFS_SECTION_RE = re.compile(
    r"\\(?:section|section\*)\{[^}]*(?:References?|Bibliography|Bibliograph)[^}]*\}",
    re.IGNORECASE,
)
_BIBLIO_CMDS = re.compile(r"\\bibliography\{|\\begin\{thebibliography\}")


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_hierarchy(
    tex_files: List[Path],
    arxiv_id: str,
    version: int,
) -> Element:
    """Parse concatenated TeX source into a document tree.

    Args:
        tex_files: Ordered list of .tex files in the compilation path.
        arxiv_id: The arXiv ID of the publication.
        version: Integer version number.

    Returns:
        Root Element with kind=document.
    """
    id_gen = _IdGen(arxiv_id)

    # Concatenate all source files
    parts: List[str] = []
    for tex_path in tex_files:
        try:
            parts.append(tex_path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    full_source = "\n".join(parts)

    # Strip comments for structural parsing
    source = strip_comments(full_source)

    # Find the body: content between \begin{document} and \end{document}
    body_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", source, re.DOTALL)
    body = body_match.group(1) if body_match else source

    # Truncate at References section
    refs_match = _REFS_SECTION_RE.search(body)
    biblio_match = _BIBLIO_CMDS.search(body)
    truncate_at = len(body)
    if refs_match:
        truncate_at = min(truncate_at, refs_match.start())
    if biblio_match:
        truncate_at = min(truncate_at, biblio_match.start())
    body = body[:truncate_at]

    # Create document root
    root_id = id_gen.next()
    root = Element(element_id=root_id, kind=KIND_DOCUMENT, content=arxiv_id, children=[])

    # Parse the body into a flat list of tokens, then build the tree
    _parse_body(body, root, id_gen, arxiv_id)

    return root


def _parse_body(source: str, parent: Element, id_gen: _IdGen, arxiv_id: str) -> None:
    """Parse body source and append children to parent.

    Processes headings, block math, figures, lists, and prose text in order.
    """
    pos = 0
    text_buffer = ""
    current_section: Optional[Element] = None  # the most recent heading element

    def _flush_text(buf: str, target: Element) -> None:
        """Split buffered prose into sentences and append to target."""
        cleaned = clean_element_content(buf, kind="sentence")
        sentences = split_sentences(cleaned)
        for s in sentences:
            s = s.strip()
            if s:
                elem = Element(
                    element_id=id_gen.next(),
                    kind=KIND_SENTENCE,
                    content=s,
                    children=[],
                )
                target.children.append(elem)

    def _current_target() -> Element:
        """Return the element that new children should be attached to."""
        return current_section if current_section is not None else parent

    n = len(source)
    while pos < n:
        # --- Check for heading ---
        h_match = _HEADING_RE.match(source, pos)
        if h_match:
            if text_buffer.strip():
                _flush_text(text_buffer, _current_target())
                text_buffer = ""
            # Extract heading title
            cmd_name = h_match.group(1).lower()
            is_star = "*" in source[h_match.start():h_match.end()]
            arg, end_pos = _extract_command_arg(source, h_match.end() - 1)
            heading_text = clean_element_content(arg, kind="heading")
            heading_elem = Element(
                element_id=id_gen.next(),
                kind=KIND_HEADING,
                content=heading_text,
                children=[],
            )
            parent.children.append(heading_elem)
            current_section = heading_elem
            pos = end_pos
            continue

        # --- Check for block math environment ---
        bm_match = _BLOCK_MATH_BEGIN.match(source, pos)
        if bm_match:
            env_name = bm_match.group(1)
            end_tag = rf"\end{{{env_name}}}"
            end_pos = source.find(end_tag, bm_match.end())
            if end_pos == -1:
                end_pos = n
                content = source[bm_match.start():end_pos]
            else:
                end_pos += len(end_tag)
                content = source[bm_match.start():end_pos]
            elem = Element(
                element_id=id_gen.next(),
                kind=KIND_FORMULA,
                content=clean_element_content(content, kind="formula"),
                children=[],
            )
            _current_target().children.append(elem)
            pos = end_pos
            continue

        # --- Check for display math \[...\] ---
        if source[pos:pos+2] == r"\[":
            end_pos = source.find(r"\]", pos + 2)
            if end_pos == -1:
                end_pos = n
                content = source[pos:end_pos]
            else:
                end_pos += 2
                content = source[pos:end_pos]
            elem = Element(
                element_id=id_gen.next(),
                kind=KIND_FORMULA,
                content=clean_element_content(content, kind="formula"),
                children=[],
            )
            _current_target().children.append(elem)
            pos = end_pos
            continue

        # --- Check for display math $$...$$ ---
        if source[pos:pos+2] == "$$":
            end_pos = source.find("$$", pos + 2)
            if end_pos == -1:
                end_pos = n
                content = source[pos:end_pos]
            else:
                end_pos += 2
                content = source[pos:end_pos]
            elem = Element(
                element_id=id_gen.next(),
                kind=KIND_FORMULA,
                content=clean_element_content(content, kind="formula"),
                children=[],
            )
            _current_target().children.append(elem)
            pos = end_pos
            continue

        # --- Check for figure/table environment ---
        fig_match = _FIGURE_BEGIN.match(source, pos)
        if fig_match:
            env_name = fig_match.group(1)
            end_tag = rf"\end{{{env_name}}}"
            # handle starred versions too
            end_tag_star = rf"\end{{{env_name}*}}"
            end_pos = source.find(end_tag, fig_match.end())
            end_pos_star = source.find(end_tag_star, fig_match.end())
            if end_pos == -1 and end_pos_star == -1:
                end_pos = n
            elif end_pos == -1:
                end_pos = end_pos_star + len(end_tag_star)
            elif end_pos_star == -1:
                end_pos = end_pos + len(end_tag)
            else:
                end_pos = min(end_pos, end_pos_star)
                end_pos += len(end_tag) if source[end_pos:end_pos+len(end_tag)] == end_tag else len(end_tag_star)

            block = source[fig_match.start():end_pos]
            # Extract caption
            cap_match = _FIGURE_CAPTION.search(block)
            caption = ""
            if cap_match:
                cap_arg, _ = _extract_command_arg(block, cap_match.end() - 1)
                caption = clean_element_content(cap_arg, kind="sentence")
            elem = Element(
                element_id=id_gen.next(),
                kind=KIND_FIGURE,
                content=caption,
                children=[],
            )
            _current_target().children.append(elem)
            pos = end_pos
            continue

        # --- Check for list environments ---
        list_match = _LIST_BEGIN.match(source, pos)
        if list_match:
            env_name = list_match.group(1)
            end_tag = rf"\end{{{env_name}}}"
            end_pos = source.find(end_tag, list_match.end())
            if end_pos == -1:
                end_pos = n
                block = source[list_match.start():end_pos]
            else:
                end_pos += len(end_tag)
                block = source[list_match.start():end_pos]

            list_elem = Element(
                element_id=id_gen.next(),
                kind=KIND_ITEMIZE,
                content="",
                children=[],
            )
            # Parse items within the list block
            inner = block[len(list_match.group(0)):-len(end_tag)] if end_pos < n else block[len(list_match.group(0)):]
            item_positions = [m.start() for m in _ITEM_RE.finditer(inner)]
            for idx, item_start in enumerate(item_positions):
                item_end = item_positions[idx + 1] if idx + 1 < len(item_positions) else len(inner)
                item_text = inner[item_start + len(r"\item"):item_end]
                cleaned = clean_element_content(item_text, kind="sentence")
                item_elem = Element(
                    element_id=id_gen.next(),
                    kind=KIND_ITEM,
                    content=cleaned.strip(),
                    children=[],
                )
                list_elem.children.append(item_elem)

            _current_target().children.append(list_elem)
            pos = end_pos
            continue

        # --- Skip other \begin{...}...\end{...} environments ---
        env_begin = re.match(r"\\begin\{([^}]+)\}", source[pos:])
        if env_begin:
            env_name = env_begin.group(1)
            end_tag = rf"\end{{{env_name}}}"
            end_pos = source.find(end_tag, pos + env_begin.end())
            if end_pos == -1:
                pos += env_begin.end()
            else:
                # Include environment content as text (may contain text)
                inner = source[pos + env_begin.end():end_pos]
                text_buffer += " " + inner + " "
                pos = end_pos + len(end_tag)
            continue

        # --- Accumulate plain text ---
        text_buffer += source[pos]
        pos += 1

    # Flush remaining text
    if text_buffer.strip():
        _flush_text(text_buffer, _current_target())


def flatten_tree(root: Element) -> List[Element]:
    """Return all elements in depth-first pre-order."""
    result: List[Element] = []

    def _walk(elem: Element) -> None:
        result.append(elem)
        for child in elem.children:
            _walk(child)

    _walk(root)
    return result


def tree_to_hierarchy_rows(
    root: Element,
    arxiv_id: str,
    version: int,
    parent_id: Optional[str] = None,
) -> List[Dict]:
    """Convert the tree to hierarchy table rows.

    Args:
        root: The root element.
        arxiv_id: arXiv ID string.
        version: Integer version number.
        parent_id: Parent element ID (None for the document root itself).

    Returns:
        List of dicts with keys: arxiv_id, version, element_id, parent_id.
    """
    rows: List[Dict] = []

    def _walk(elem: Element, pid: Optional[str]) -> None:
        if pid is not None:
            rows.append({
                "arxiv_id": arxiv_id,
                "version": version,
                "element_id": elem.element_id,
                "parent_id": pid,
            })
        for child in elem.children:
            _walk(child, elem.element_id)

    _walk(root, parent_id)
    return rows
