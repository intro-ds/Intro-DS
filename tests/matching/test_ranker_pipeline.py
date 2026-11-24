"""Unit tests for split_data, features and the ranking metrics."""

import pytest

from src.matching.features import FEATURE_COLUMNS, publication_features
from src.matching.split_data import make_split
from src.matching.train_ranker import reciprocal_rank, summarize


def _labels(n_manual: int, n_auto: int):
    labels, source = {}, {}
    for i in range(n_manual):
        aid = f"2511.1000{i}"
        labels[aid] = {f"k{j}": ("1706.03762" if j % 2 else "") for j in range(25)}
        source[aid] = "manual"
    for i in range(n_auto):
        aid = f"2511.2000{i}"
        labels[aid] = {f"k{j}": "1412.6980" for j in range(8)}
        source[aid] = "auto"
    return labels, source


def test_make_split_sizes_and_disjoint() -> None:
    labels, source = _labels(5, 4)
    split = make_split(labels, source, members=1, seed=1)
    for origin in ("manual", "auto"):
        test, valid, train = split["test"][origin], split["valid"][origin], split["train"][origin]
        assert len(test) == 1 and len(valid) == 1
        assert set(test).isdisjoint(valid)
        assert set(train).isdisjoint(test + valid)
    assert len(split["train"]["manual"]) == 3
    assert len(split["train"]["auto"]) == 2


def test_make_split_requires_enough_publications() -> None:
    labels, source = _labels(2, 4)
    with pytest.raises(ValueError):
        make_split(labels, source)


def test_reciprocal_rank_and_summary() -> None:
    assert reciprocal_rank(["a", "b", "c"], "b") == 0.5
    assert reciprocal_rank(["a", "b"], "z") == 0.0
    items = [("a", ["a", "b"]), ("b", ["x", "b"]), ("", ["a"])]
    s = summarize(items)
    assert s["n_labeled"] == 3 and s["n_positive"] == 2
    assert s["mrr5"] == pytest.approx(0.5)               # (1 + 0.5 + 0) / 3
    assert s["mrr5_positive_only"] == pytest.approx(0.75)
    assert s["hit@1"] == pytest.approx(0.5)


def test_publication_features_best_match(workspace) -> None:
    entries = [
        workspace.entry("a", "Attention Is All You Need", ["Vaswani"], 2017),
        workspace.entry("b", "Adam: A Method for Stochastic Optimization", ["Kingma"], 2014),
    ]
    refs = [
        workspace.ref("1706.03762", "Attention Is All You Need", ["Ashish Vaswani"], "2017-06-12"),
        workspace.ref("1412.6980", "Adam: A Method for Stochastic Optimization", ["Diederik Kingma"], "2014-12-22"),
    ]
    df = publication_features(entries, refs)
    assert len(df) == 4
    assert all(col in df.columns for col in FEATURE_COLUMNS)
    best = df.loc[df.groupby("entry_idx")["heuristic_score"].idxmax()]
    assert list(best["ref_arxiv_id"]) == ["1706.03762", "1412.6980"]


def test_publication_features_empty() -> None:
    assert publication_features([], []).empty