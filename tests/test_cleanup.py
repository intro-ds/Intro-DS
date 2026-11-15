from src.latexParser import PublicationHierarchyBuilder

clean = PublicationHierarchyBuilder.clean_latex_text


def test_cleanup_removes_formatting_but_keeps_content():
    raw = (
        r"\begin{figure}[htpb] \centering \includegraphics{a.png} "
        r"\toprule A & B \\ \midrule 1 & 2 \\ \bottomrule "
        r"\caption{A \textbf{bold} and \textbf{\emph{nested}} result} \end{figure}"
    )
    out = clean(raw)

    for removed in (r"\centering", "[htpb]", r"\toprule", r"\midrule", r"\bottomrule",
                    r"\textbf", r"\emph"):
        assert removed not in out, f"{removed} should have been removed"

    # semantic content survives
    assert r"\caption{A bold and nested result}" in out
    assert r"\includegraphics{a.png}" in out


def test_math_normalization():
    builder_cls = PublicationHierarchyBuilder

    # inline: \( ... \) and \begin{math} ... \end{math} -> $...$
    assert builder_cls.normalize_inline_math(
        r"Let \(x+y\) and \begin{math}z\end{math} hold"
    ) == "Let $x+y$ and $z$ hold"

    # block: $$...$$, \[...\], align -> equation
    expected = r"\begin{equation} a+b \end{equation}"
    assert builder_cls.normalize_block_math_content("$$ a+b $$") == expected
    assert builder_cls.normalize_block_math_content(r"\[ a+b \]") == expected
    assert builder_cls.normalize_block_math_content(
        r"\begin{align} a &= b \\ c &= d \end{align}"
    ) == r"\begin{equation} a &= b \\ c &= d \end{equation}"
    assert builder_cls.normalize_block_math_content(
        r"\begin{align*} x \end{align*}"
    ) == r"\begin{equation} x \end{equation}"

    # end to end: $$...$$ in the body becomes a block_formula element, no `$$` left
    builder = PublicationHierarchyBuilder("2310.12345")
    builder.parse_version(
        1, r"\begin{document}\section{A} Text. $$x=1$$ More text.\end{document}"
    )
    formulas = [e["content"] for e in builder.elements_rows if e["kind"] == "block_formula"]
    assert formulas == [r"\begin{equation} x=1 \end{equation}"]
    assert all("$$" not in e["content"] for e in builder.elements_rows)