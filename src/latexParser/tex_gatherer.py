"""Tex gatherer — finds the compilation root and resolves transitively included .tex files.

The compilation root is the .tex file containing the \\documentclass declaration.
\\input and \\include commands are followed transitively; commented-out commands are ignored.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set


_DOCUMENTCLASS_RE = re.compile(r"\\documentclass\b")
_BEGIN_DOC_RE = re.compile(r"\\begin\{document\}")
_END_DOC_RE = re.compile(r"\\end\{document\}")
_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\{([^}]+)\}")
# File stems that usually name the real entry point of a submission
_ROOT_NAME_HINTS = {"main", "ms", "paper", "article", "root"}


def _strip_comments(source: str) -> str:
    """Remove LaTeX comments from source text.

    Preserves newlines so line numbers stay consistent.
    """
    # Replace inline comments (not escaped %) with empty string, keep newline
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def _read_source(path: Path) -> Optional[str]:
    """Read a .tex file as text, returning None for unreadable or binary files."""
    try:
        with path.open("rb") as bf:
            header = bf.read(1024)
        if header.startswith(b"%PDF-") or b"\x00" in header:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def find_root_tex(version_dir: Path) -> Optional[Path]:
    """Return the .tex file that is the compilation root of a version.

    Candidates are files with an uncommented \\documentclass (and, when any
    exist, a \\begin{document}). Among candidates the best one is picked by:
    1. Not being \\input/\\include'd by another file.
    2. Hints that it is the real entry point: its stem matches a compiled
       .bbl file (arXiv names the .bbl after the main file) or is a typical
       name such as "main".
    3. Having \\end{document}.
    4. Being the largest file (templates and specimens are usually smaller).

    Args:
        version_dir: Path to the vN/ directory.

    Returns:
        Path to the root .tex file, or None if not found.
    """
    sources: Dict[Path, str] = {}
    for tex_path in sorted(version_dir.glob("*.tex")):
        text = _read_source(tex_path)
        if text is not None:
            sources[tex_path] = _strip_comments(text)

    with_class = [p for p, s in sources.items() if _DOCUMENTCLASS_RE.search(s)]
    if not with_class:
        return None
    with_body = [p for p in with_class if _BEGIN_DOC_RE.search(sources[p])]
    candidates = with_body or with_class

    included_refs: Set[str] = set()
    for text in sources.values():
        for m in _INPUT_RE.finditer(text):
            ref = Path(m.group(1).strip()).name
            included_refs.add(ref)
            included_refs.add(Path(ref).stem)

    bbl_stems = {p.stem for p in version_dir.glob("*.bbl")}

    def _score(path: Path) -> tuple:
        text = sources[path]
        is_included = path.name in included_refs or path.stem in included_refs
        hints = int(path.stem in bbl_stems) + int(path.stem.lower() in _ROOT_NAME_HINTS)
        return (not is_included, hints, bool(_END_DOC_RE.search(text)), len(text))

    return max(candidates, key=_score)


def _resolve_input(ref: str, version_dir: Path) -> Optional[Path]:
    """Resolve an \\input or \\include argument to a file path.

    Args:
        ref: The raw argument of \\input{ref} or \\include{ref}.
        version_dir: Directory to search in.

    Returns:
        Resolved Path if found, else None.
    """
    ref = ref.strip()
    # Try as-is, then with .tex extension
    for candidate in [Path(ref), Path(ref + ".tex")]:
        # Resolve relative to version_dir
        resolved = version_dir / candidate
        if resolved.exists() and resolved.is_file():
            return resolved
        # Also try just the filename part (some papers use subdirs)
        name_only = version_dir / candidate.name
        if name_only.exists() and name_only.is_file():
            return name_only
    return None


def gather_tex_files(root: Path, version_dir: Path) -> List[Path]:
    """Return ordered list of .tex files in the compilation path.

    Follows \\input{file} and \\include{file} commands transitively.
    Commented-out commands are ignored.
    Circular inclusions are detected and broken.

    Args:
        root: The root .tex file (containing \\documentclass).
        version_dir: The version directory to resolve includes from.

    Returns:
        List of Paths in traversal order (root first).
    """
    visited: Set[Path] = set()
    result: List[Path] = []

    def _walk(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        result.append(path)

        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        stripped = _strip_comments(source)
        for match in _INPUT_RE.finditer(stripped):
            ref = match.group(1)
            child = _resolve_input(ref, version_dir)
            if child is not None:
                _walk(child)

    _walk(root)
    return result


def expand_includes(tex_files: List[Path]) -> str:
    """Return the source of the root file with \\input/\\include inlined in place.

    The first item of tex_files is the root. Each \\input{x} / \\include{x}
    command is replaced by the (comment-free) content of the matching file in
    tex_files, so included text appears at the position LaTeX would place it.
    Missing or circular includes are dropped.

    Args:
        tex_files: Ordered files of the compilation path (root first).

    Returns:
        Single source string with comments removed.
    """
    if not tex_files:
        return ""

    by_name: Dict[str, Path] = {}
    for p in tex_files:
        by_name[p.name] = p
        by_name[p.stem] = p

    def _expand(path: Path, stack: Set[Path]) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        text = _strip_comments(text)

        def _sub(match: "re.Match[str]") -> str:
            child = by_name.get(Path(match.group(1).strip()).name)
            if child is None or child.resolve() in stack:
                return ""
            return "\n" + _expand(child, stack | {child.resolve()}) + "\n"

        return _INPUT_RE.sub(_sub, text)

    root = tex_files[0]
    return _expand(root, {root.resolve()})