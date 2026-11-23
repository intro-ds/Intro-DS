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
from src.latexParser.tex_gatherer import expand_includes

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

HEADING_LEVELS: Dict[str, int] = dict(HEADING_CMDS)
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

    def __init__(self, arxiv_id: str, version: int) -> None:
        self._prefix = f"{arxiv_id.replace('.', '-')}#v{version}-"
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
# References section markers
# Matches e.g. \section{References}, \section*{Bibliography}, \chapter{References}
_REFS_SECTION_RE = re.compile(
    r"\\(?:chapter|section)\*?\s*\{[^}]*(?:References?|Bibliography)[^}]*\}",
    re.IGNORECASE,
)
# Next structural boundary after a references section
_SECTION_BOUNDARY_RE = re.compile(r"\\(?:chapter|section|appendix)\b")
_THEBIBLIOGRAPHY_RE = re.compile(
    r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL
)
_BIBLIOGRAPHY_CMD_RE = re.compile(r"\\bibliography(?:style)?\s*\{[^}]*\}")

def _remove_bibliography(body: str) -> str:
    """Remove bibliography content without dropping sections that follow it.

    Appendices frequently appear after the references, so the body must not be
    truncated at the bibliography.
    """
    # 1. Drop a References/Bibliography section up to the next section or \appendix
    while True:
        refs_match = _REFS_SECTION_RE.search(body)
        if refs_match is None:
            break
        boundary = _SECTION_BOUNDARY_RE.search(body, refs_match.end())
        end = boundary.start() if boundary else len(body)
        body = body[: refs_match.start()] + body[end:]

    # 2. Drop explicit thebibliography environments
    body = _THEBIBLIOGRAPHY_RE.sub("", body)

    # 3. Drop \bibliography{...} and \bibliographystyle{...}
    return _BIBLIOGRAPHY_CMD_RE.sub("", body)

# Commands that only define macros or load packages and carry no content.
# Value = number of mandatory arguments to drop.
_DEF_CMDS: Dict[str, int] = {
    "newcommand": 2, "renewcommand": 2, "providecommand": 2,
    "DeclareRobustCommand": 2, "DeclareMathOperator": 2,
    "newtheorem": 2, "newenvironment": 3, "renewenvironment": 3,
    "usepackage": 1, "RequirePackage": 1, "newcounter": 1,
    "setlength": 2, "setcounter": 2, "addtolength": 2,
    "graphicspath": 1, "hypersetup": 1,
    "pagestyle": 1, "thispagestyle": 1, "bibliographystyle": 1,
}
_DEF_RE = re.compile(
    r"\\(" + "|".join(sorted(_DEF_CMDS, key=len, reverse=True)) + r")\*?(?![A-Za-z@])"
)
_DEF_PRIMITIVE_RE = re.compile(r"\\(?:def|gdef|edef|xdef)\s*\\[A-Za-z@]+")
_CONTROL_SEQ_RE = re.compile(r"\\(?:[A-Za-z@]+|.)")


def _match_group(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Return the index after the group opening at start, or -1 if unbalanced."""
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _skip_args(text: str, pos: int, count: int) -> int:
    """Skip `count` mandatory arguments (plus optional [..] ones) from pos."""
    taken = 0
    while taken < count:
        p = pos
        while p < len(text) and text[p] in " \t\r\n":
            p += 1
        if p >= len(text):
            return pos
        ch = text[p]
        if ch == "[":
            end = _match_group(text, p, "[", "]")
        elif ch == "{":
            end = _match_group(text, p, "{", "}")
            taken += 1
        elif ch == "\\":
            m = _CONTROL_SEQ_RE.match(text, p)
            end = m.end() if m else -1
            taken += 1
        else:
            return pos
        if end < 0:
            return pos
        pos = end
    return pos


def _skip_optional(text: str, pos: int) -> int:
    """Skip one optional [..] group (e.g. the [section] of \\newtheorem)."""
    p = pos
    while p < len(text) and text[p] in " \t":
        p += 1
    if p < len(text) and text[p] == "[":
        end = _match_group(text, p, "[", "]")
        if end > 0:
            return end
    return pos


def _strip_definitions(source: str) -> str:
    """Remove macro definitions and package commands from the body source."""
    out: List[str] = []
    pos = 0
    while True:
        m1 = _DEF_RE.search(source, pos)
        m2 = _DEF_PRIMITIVE_RE.search(source, pos)
        found = [m for m in (m1, m2) if m]
        if not found:
            break
        m = min(found, key=lambda x: x.start())
        if m is m1:
            end = _skip_args(source, m.end(), _DEF_CMDS[m.group(1)])
            if m.group(1) == "newtheorem":
                end = _skip_optional(source, end)
        else:
            brace = source.find("{", m.end())
            end = m.end()
            if brace != -1 and re.fullmatch(r"[#\d\s]*", source[m.end():brace]):
                close = _match_group(source, brace, "{", "}")
                end = close if close > 0 else m.end()
        out.append(source[pos:m.start()])
        pos = end
    out.append(source[pos:])
    return "".join(out)


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
    id_gen = _IdGen(arxiv_id, version)

    # Inline \input/\include so included text sits where LaTeX would place it
    full_source = expand_includes(tex_files)

    # Strip comments for structural parsing
    source = strip_comments(full_source)

    # Find the body: content between \begin{document} and \end{document}
    body_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", source, re.DOTALL)
    body = body_match.group(1) if body_match else source

    
    body = _remove_bibliography(body)
    # Drop macro definitions / package commands that leaked into the body
    body = _strip_definitions(body)
    
    # Create document root
    root_id = id_gen.next()
    root = Element(
        element_id=root_id,
        kind=KIND_DOCUMENT,
        content=f"{arxiv_id}v{version}",
        children=[],
    )

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
    heading_stack: List[Tuple[int, Element]] = []

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
            # Nest the heading under the closest open heading of a higher level
            level = HEADING_LEVELS[cmd_name]
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_parent = heading_stack[-1][1] if heading_stack else parent
            heading_parent.children.append(heading_elem)
            heading_stack.append((level, heading_elem))
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
