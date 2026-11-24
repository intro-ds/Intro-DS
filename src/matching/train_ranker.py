"""Train a LightGBM ranker for reference matching, evaluate MRR@5, export predictions.

Pipeline:
  1. read split.json and the labels (split_data.py)
  2. build pair features for every publication of the split (features.py)
  3. train LGBMRanker (lambdarank) on the train publications; hard + random negatives
     are sampled per labelled entry, valid publications drive early stopping
  4. rank ALL references of each entry, keep the top-5, compute MRR@5 on valid/test
     (model vs. rule-based baseline, manual vs. automatic publications)
  5. write predictions.parquet (one row per BibTeX entry of every split publication)

Usage:
    python -m src.matching.train_ranker --student-id 24127590
    python -m src.matching.train_ranker --drop-leaky   # ablation, writes metrics only
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.matching.features import FEATURE_COLUMNS, LEAKY_FEATURES, publication_features
from src.matching.select_publications import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_INTERMEDIATE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STUDENT_ID,
    load_entries,
    load_references,
)
from src.matching.split_data import ORIGINS, PARTITIONS, load_labels, load_split

TOP_K = 5
DEFAULT_PROCESSED_DIR = Path("data/processed")
METRICS_FILE = "metrics.json"
ABLATION_METRICS_FILE = "metrics_drop_leaky.json"

PREDICTIONS_SCHEMA = pa.schema(
    [
        ("arxiv_id", pa.string()),
        ("partition", pa.dictionary(pa.int32(), pa.string())),
        ("citation_key", pa.string()),
        ("groundtruth_arxiv_id", pa.string()),
    ]
    + [(f"cand_{k}", pa.string()) for k in range(1, TOP_K + 1)]
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def reciprocal_rank(candidates: Sequence[str], truth: str, k: int = TOP_K) -> float:
    """1 / rank of the true reference among the first k candidates, else 0."""
    for rank, cand in enumerate(candidates[:k], start=1):
        if cand == truth:
            return 1.0 / rank
    return 0.0


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize(items: List[Tuple[str, List[str]]]) -> Dict[str, float]:
    """Metrics of (truth, top-k candidates) pairs; truth == "" means no true reference.

    mrr5                literal reading of the statement: every labelled entry counts,
                        entries without a true reference score 0
    mrr5_positive_only  only entries that have a true reference
    """
    positives = [(t, c) for t, c in items if t]
    return {
        "n_labeled": len(items),
        "n_positive": len(positives),
        "mrr5": _mean([reciprocal_rank(c, t) if t else 0.0 for t, c in items]),
        "mrr5_positive_only": _mean([reciprocal_rank(c, t) for t, c in positives]),
        "hit@1": _mean([float(bool(c) and c[0] == t) for t, c in positives]),
        "hit@5": _mean([float(t in c[:TOP_K]) for t, c in positives]),
    }


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


def _top_k(df: pd.DataFrame, m: int, n: int, scores: np.ndarray, k: int = TOP_K) -> List[List[str]]:
    """Top-k reference IDs per entry (rows of df are entry-major, m x n)."""
    order = np.argsort(-scores.reshape(m, n), axis=1, kind="stable")[:, :k]
    ref_ids = df["ref_arxiv_id"].to_numpy().reshape(m, n)
    return [[str(x) for x in ref_ids[i, order[i]]] for i in range(m)]


def _sample_rows(
    df: pd.DataFrame,
    entries: List[Dict[str, Any]],
    refs: List[Dict[str, Any]],
    truth_map: Dict[str, str],
    n_hard: int,
    n_random: int,
    rng: random.Random,
) -> pd.DataFrame:
    """Rows of the labelled entries that have a true reference, with a `label` column.

    Per entry we keep the positive, the n_hard best-scoring negatives (hard negatives
    by the rule-based score) and n_random random negatives. n_hard = huge keeps everything.
    """
    m, n = len(entries), len(refs)
    empty = df.iloc[0:0].copy()
    empty["label"] = pd.Series(dtype=int)
    if m == 0 or n == 0:
        return empty

    ref_index = {r["ref_arxiv_id"]: j for j, r in enumerate(refs)}
    heuristic = df["heuristic_score"].to_numpy(dtype=float).reshape(m, n)
    labels = np.zeros(m * n, dtype=int)
    keep: List[int] = []
    for i, entry in enumerate(entries):
        truth = truth_map.get(entry.get("citation_key", ""))
        j_pos = ref_index.get(truth) if truth else None
        if j_pos is None:
            continue  # unlabelled or "no match" entries carry no ranking signal
        labels[i * n + j_pos] = 1
        others = [int(j) for j in np.argsort(-heuristic[i], kind="stable") if j != j_pos]
        rest = others[n_hard:]
        chosen = others[:n_hard] + rng.sample(rest, min(n_random, len(rest)))
        keep.extend(i * n + j for j in [j_pos] + chosen)

    if not keep:
        return empty
    keep.sort()  # keeps each entry's rows contiguous (LightGBM groups)
    sub = df.iloc[keep].copy()
    sub["label"] = labels[keep]
    return sub


def _build_matrix(
    part: str,
    split: Dict,
    data: Dict[str, Tuple[List, List, pd.DataFrame]],
    labels: Dict[str, Dict[str, str]],
    features: List[str],
    n_hard: int,
    n_random: int,
    rng: random.Random,
) -> Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    """Stack the sampled rows of one partition: (X, y, group sizes)."""
    frames = []
    for origin in ORIGINS:
        for aid in split[part][origin]:
            entries, refs, df = data[aid]
            sub = _sample_rows(df, entries, refs, labels[aid], n_hard, n_random, rng)
            if len(sub):
                sub["group"] = aid + "|" + sub["entry_idx"].astype(str)
                frames.append(sub)
    if not frames:
        return None
    full = pd.concat(frames, ignore_index=True)
    sizes = full.groupby("group", sort=False).size().to_numpy()
    return full[features], full["label"].to_numpy(), sizes


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _write_predictions(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {f.name: pa.array([r[f.name] for r in rows], type=f.type) for f in PREDICTIONS_SCHEMA}
    pq.write_table(pa.table(columns, schema=PREDICTIONS_SCHEMA), path, compression="zstd")


def run_training(
    student_id: str = DEFAULT_STUDENT_ID,
    intermediate_dir: Path = DEFAULT_INTERMEDIATE_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    drop_leaky: bool = False,
    seed: int = 42,
    n_hard: int = 30,
    n_random: int = 10,
    n_estimators: int = 400,
) -> Dict[str, Any]:
    """Train, evaluate and (unless drop_leaky) write predictions.parquet."""
    import lightgbm as lgb  # imported here so that the metric helpers work without it

    split = load_split(output_dir)
    labels, _ = load_labels(output_dir)
    features = [f for f in FEATURE_COLUMNS if not (drop_leaky and f in LEAKY_FEATURES)]

    partition_of: Dict[str, str] = {}
    origin_of: Dict[str, str] = {}
    for part in PARTITIONS:
        for origin in ORIGINS:
            for aid in split[part][origin]:
                partition_of[aid] = part
                origin_of[aid] = origin

    data: Dict[str, Tuple[List, List, pd.DataFrame]] = {}
    for aid in tqdm(partition_of, desc="Building pair features", unit="paper"):
        entries = load_entries(aid, intermediate_dir)
        refs = load_references(aid, corpus_dir)
        data[aid] = (entries, refs, publication_features(entries, refs))

    rng = random.Random(seed)
    train = _build_matrix("train", split, data, labels, features, n_hard, n_random, rng)
    valid = _build_matrix("valid", split, data, labels, features, 10**9, 0, rng)
    if train is None:
        raise RuntimeError("No positive training pairs: check the labels and split.json.")

    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    fit_kwargs: Dict[str, Any] = {}
    if valid is not None:
        fit_kwargs = {
            "eval_set": [(valid[0], valid[1])],
            "eval_group": [valid[2]],
            "eval_at": [TOP_K],
            "callbacks": [lgb.early_stopping(30, verbose=False)],
        }
    model.fit(train[0], train[1], group=train[2], **fit_kwargs)
    print(f"Trained on {len(train[1])} pairs / {len(train[2])} entries, best iteration: {model.best_iteration_}")

    prediction_rows: List[Dict[str, Any]] = []
    model_items = {p: {o: [] for o in ORIGINS} for p in ("valid", "test")}
    base_items = {p: {o: [] for o in ORIGINS} for p in ("valid", "test")}

    for aid, (entries, refs, df) in data.items():
        part, origin = partition_of[aid], origin_of[aid]
        m, n = len(entries), len(refs)
        if m and n:
            scores = model.predict(df[features])
            model_top = _top_k(df, m, n, np.asarray(scores))
            base_top = _top_k(df, m, n, df["heuristic_score"].to_numpy(dtype=float))
        else:
            model_top = base_top = [[] for _ in entries]

        for i, entry in enumerate(entries):
            key = entry.get("citation_key", "")
            truth = labels[aid].get(key)  # None = not labelled, "" = no true reference
            row: Dict[str, Any] = {
                "arxiv_id": aid,
                "partition": part,
                "citation_key": key,
                "groundtruth_arxiv_id": truth or None,
            }
            for k, cand in enumerate((model_top[i] + [None] * TOP_K)[:TOP_K], start=1):
                row[f"cand_{k}"] = cand
            prediction_rows.append(row)

            if part in ("valid", "test") and truth is not None:
                model_items[part][origin].append((truth, model_top[i]))
                base_items[part][origin].append((truth, base_top[i]))

    def _report(items: Dict[str, Dict[str, List]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for part, by_origin in items.items():
            merged = [x for o in ORIGINS for x in by_origin[o]]
            out[part] = {o: summarize(by_origin[o]) for o in ORIGINS}
            out[part]["all"] = summarize(merged)
        return out

    gains = model.booster_.feature_importance(importance_type="gain")
    metrics = {
        "model": _report(model_items),
        "baseline_heuristic": _report(base_items),
        "feature_importance_gain": {f: float(g) for f, g in sorted(zip(features, gains), key=lambda x: -x[1])},
        "config": {
            "drop_leaky": drop_leaky,
            "seed": seed,
            "n_hard": n_hard,
            "n_random": n_random,
            "best_iteration": int(model.best_iteration_ or n_estimators),
            "split": split,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / (ABLATION_METRICS_FILE if drop_leaky else METRICS_FILE)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_path: Optional[Path] = None
    if not drop_leaky:  # the ablation model must never overwrite the submitted predictions
        pred_path = processed_dir / student_id / "predictions.parquet"
        _write_predictions(prediction_rows, pred_path)

    return {"metrics": metrics, "metrics_path": str(metrics_path), "predictions_path": str(pred_path) if pred_path else None, "n_prediction_rows": len(prediction_rows)}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the reference ranker and export predictions")
    parser.add_argument("--student-id", default=DEFAULT_STUDENT_ID)
    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--drop-leaky", action="store_true", help="Ablation: drop rule-based score features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-hard", type=int, default=30)
    parser.add_argument("--neg-random", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=400)
    args = parser.parse_args()

    res = run_training(
        args.student_id, args.intermediate_dir, args.corpus_dir, args.output_dir, args.processed_dir,
        args.drop_leaky, args.seed, args.neg_hard, args.neg_random, args.n_estimators,
    )
    for name in ("model", "baseline_heuristic"):
        print(f"\n== {name} ==")
        for part in ("valid", "test"):
            for origin in (*ORIGINS, "all"):
                s = res["metrics"][name][part][origin]
                print(
                    f"  {part:5s} {origin:6s} n={s['n_labeled']:4d} MRR@5={s['mrr5']:.4f} "
                    f"(positives only {s['mrr5_positive_only']:.4f}) hit@1={s['hit@1']:.3f} hit@5={s['hit@5']:.3f}"
                )
    print(f"\nMetrics: {res['metrics_path']}")
    if res["predictions_path"]:
        print(f"Predictions: {res['predictions_path']} ({res['n_prediction_rows']} rows)")


if __name__ == "__main__":
    main()