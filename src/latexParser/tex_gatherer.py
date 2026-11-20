"""Tex gatherer — finds the compilation root and resolves transitively included .tex files.

The compilation root is the .tex file containing the \\documentclass declaration.
\\input and \\include commands are followed transitively; commented-out commands are ignored.
"""

import re
from pathlib import Path
from typing import List, Optional, Set


_DOCUMENTCLASS_RE = re.compile(r"\\documentclass\b")
_COMMENT_LINE_RE = re.compile(r"(?m)^[ \t]*%.*$")
_INLINE_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\{([^}]+)\}")


def _strip_comments(source: str) -> str:
    """Remove LaTeX comments from source text.

    Preserves newlines so line numbers stay consistent.
    """
    # Replace inline comments (not escaped %) with empty string, keep newline
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def find_root_tex(version_dir: Path) -> Optional[Path]:
    """Return the .tex file containing \\documentclass (the compilation root).

    Strategy:
    1. Any .tex file that contains \\documentclass not on a commented line.
    2. If multiple candidates, pick the one where \\documentclass appears earliest.
    3. If none, return None.

    Args:
        version_dir: Path to the vN/ directory.

    Returns:
        Path to the root .tex file, or None if not found.
    """
    tex_files = sorted(version_dir.glob("*.tex"))
    if not tex_files:
        return None

    best_path: Optional[Path] = None
    best_line = float("inf")

    for tex_path in tex_files:
        try:
            with tex_path.open("rb") as bf:
                header = bf.read(1024)
            if header.startswith(b"%PDF-") or b"\x00" in header:
                continue
            source = tex_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        stripped = _strip_comments(source)
        match = re.search(r"\\documentclass\b", stripped)
        if match:
            # Count how many newlines before the match = line number
            line_num = stripped[: match.start()].count("\n")
            if line_num < best_line:
                best_line = line_num
                best_path = tex_path

    return best_path


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
