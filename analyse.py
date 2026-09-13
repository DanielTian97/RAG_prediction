#!/usr/bin/env python3
"""Evaluate single features and feature-family combinations against RAG targets.

This stage consumes feature matrices produced by ``compute_features.py``. It does
not compute any feature itself. The only outputs are numerical correlation
result tables.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

PAPER_QPP_FEATURES = ["nqc", "spatial", "maxScore", "a_ratio", "bertQPP"]
LOG1P_GROUPS = {"qpp", "qualt5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--dev-zero-eval", type=Path, nargs="+", required=True)
    parser.add_argument("--dev-k-eval", type=Path, nargs="+", required=True)
    parser.add_argument("--test-zero-eval", type=Path, nargs="+", required=True)
    parser.add_argument("--test-k-eval", type=Path, nargs="+", required=True)
    parser.add_argument("--task", choices=["nq", "dl"], required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/correlations")
    )
    parser.add_argument(
        "--include-qv-qpp",
        action="store_true",
        help="Include bertQPP(QV) if present. The paper analysis excluded it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace canonical result CSVs instead of appending/deduplicating this condition.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"{label} does not exist or is not a file: {path}")
    return path


def load_feature_csv(path: Path, label: str) -> pd.DataFrame:
    path = require_file(path, label)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Could not read {label} as CSV ({path}): {exc}") from exc
    if "qid" not in frame.columns:
        raise SystemExit(f"{label} must contain a qid column: {path}")
    frame = frame.copy()
    frame["qid"] = frame["qid"].astype(str)
    if frame["qid"].duplicated().any():
        sample = frame.loc[frame["qid"].duplicated(), "qid"].head(5).tolist()
        raise SystemExit(f"{label} contains duplicate qid rows: {sample}")
    return frame


def load_jsons(paths: list[Path], label: str) -> dict:
    merged = {}
    for path in paths:
        path = require_file(path, label)
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except Exception as exc:
            raise SystemExit(f"Could not read {label} as JSON ({path}): {exc}") from exc
        if not isinstance(value, dict):
            raise SystemExit(f"{label} must contain a top-level JSON object: {path}")
        overlap = set(merged) & set(value)
        if overlap:
            sample = ", ".join(sorted(map(str, overlap))[:5])
            raise SystemExit(
                f"Duplicate qid(s) while merging {label} files: {sample}"
            )
        merged.update(value)
    return merged


def extract_f1(evaluations: dict, task: str) -> dict[str, float]:
    scores = {}
    for qid, item in evaluations.items():
        try:
            if task == "nq":
                score = item["0"]["0"]["F1"]
            else:
                score = item["0"]["0"]["qrel_1"]["f1"]["max"]
        except (KeyError, TypeError):
            continue
        scores[str(qid)] = float(score)
    return scores


def attach_targets(
    features: pd.DataFrame,
    zero_eval_paths: list[Path],
    k_eval_paths: list[Path],
    task: str,
    split_label: str = "dataset",
) -> pd.DataFrame:
    base = extract_f1(load_jsons(zero_eval_paths, f"{split_label} zero-context evaluation"), task)
    with_context = extract_f1(
        load_jsons(k_eval_paths, f"{split_label} retrieved-context evaluation"), task
    )
    common = sorted(set(base) & set(with_context))
    if not common:
        raise SystemExit(
            f"No qids overlap between {split_label} zero-context and retrieved-context evaluations."
        )

    target = pd.DataFrame(
        {
            "qid": common,
            "f1": [with_context[qid] for qid in common],
            "utility": [with_context[qid] - base[qid] for qid in common],
        }
    )
    frame = features.copy()
    frame["qid"] = frame["qid"].astype(str)
    merged = frame.merge(target, on="qid", how="inner", validate="one_to_one")
    if merged.empty:
        raise SystemExit(
            f"No qids overlap between {split_label} feature matrix and evaluation targets."
        )
    return merged


def load_manifest(feature_path: Path) -> dict[str, list[str]] | None:
    manifest = feature_path.parent / "feature_groups.json"
    if not manifest.is_file():
        return None
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not read feature manifest {manifest}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Feature manifest must contain a JSON object: {manifest}")
    cleaned = {}
    for name, columns in value.items():
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
            raise SystemExit(
                f"Feature manifest entry '{name}' must be a list of column names: {manifest}"
            )
        cleaned[str(name)] = columns
    return cleaned


def load_feature_groups(
    dev_feature_path: Path,
    test_feature_path: Path,
    columns: list[str],
) -> dict[str, list[str]]:
    dev_manifest = load_manifest(dev_feature_path)
    test_manifest = load_manifest(test_feature_path)

    if dev_manifest is not None and test_manifest is not None:
        if dev_manifest != test_manifest:
            raise SystemExit(
                "Development and test feature_groups.json files differ. Recompute both "
                "splits with the same feature families/settings."
            )
        return {
            name: [column for column in values if column in columns]
            for name, values in test_manifest.items()
        }

    if (dev_manifest is None) != (test_manifest is None):
        raise SystemExit(
            "Only one feature matrix has a feature_groups.json manifest. Either provide "
            "both manifests or neither."
        )

    return {
        "qpp": [c for c in PAPER_QPP_FEATURES if c in columns],
        "context_perplexity": [c for c in columns if "perpC" in c],
        "qualt5": [c for c in columns if "docQual" in c],
        "posteriors": [c for c in ["prob(k)"] if c in columns],
        "readability": [
            c
            for c in columns
            if c.startswith(("max(", "min(", "avg(", "itg("))
            and "perpC" not in c
            and "docQual" not in c
        ],
    }


def valid_common_features(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    groups: dict[str, list[str]],
    include_qv_qpp: bool,
) -> dict[str, list[str]]:
    result = {}
    for group, features in groups.items():
        clean = []
        for feature in features:
            if feature not in dev.columns or feature not in test.columns:
                continue
            if feature == "bertQPP(QV)" and not include_qv_qpp:
                continue
            dev_values = pd.to_numeric(dev[feature], errors="coerce")
            test_values = pd.to_numeric(test[feature], errors="coerce")
            if not (np.isfinite(dev_values).all() and np.isfinite(test_values).all()):
                continue
            if group == "readability" and (
                (dev_values == -1).any() or (test_values == -1).any()
            ):
                continue
            clean.append(feature)
        result[group] = clean

    if "qpp" in result:
        paper = [c for c in PAPER_QPP_FEATURES if c in result["qpp"]]
        if paper:
            result["qpp"] = paper
    return result


def single_correlations(test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for target, prediction_name in [("f1", "GPP"), ("utility", "RPP")]:
        for feature in feature_columns:
            values = pd.to_numeric(test[feature], errors="coerce")
            mask = np.isfinite(values) & np.isfinite(test[target])
            if mask.sum() < 2:
                continue
            rho = stats.spearmanr(values[mask], test.loc[mask, target]).statistic
            tau = stats.kendalltau(values[mask], test.loc[mask, target]).statistic
            rows.append(
                {
                    "Prediction Name": prediction_name,
                    "Target": target,
                    "Feature": feature,
                    "Spearman": float(rho),
                    "Kendall": float(tau),
                    "Number of Queries": int(mask.sum()),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "Prediction Name",
            "Target",
            "Feature",
            "Spearman",
            "Kendall",
            "Number of Queries",
        ],
    )


def transform_group(frame: pd.DataFrame, group: str, features: list[str]) -> pd.DataFrame:
    values = frame[features].apply(pd.to_numeric, errors="coerce").copy()
    if group in LOG1P_GROUPS:
        values = np.log1p(values)
    return values


def combination_correlations(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    groups: dict[str, list[str]],
    singles: pd.DataFrame,
    task: str,
    retriever: str,
    k: int,
) -> pd.DataFrame:
    pregen = [
        group
        for group in ["qpp", "context_perplexity", "qualt5", "readability"]
        if groups.get(group)
    ]
    has_posteriors = bool(groups.get("posteriors"))
    rows = []

    combinations = []
    for size in range(1, len(pregen) + 1):
        combinations.extend(itertools.combinations(pregen, size))

    for target, prediction_name in [("f1", "GPP"), ("utility", "RPP")]:
        for combo in combinations:
            for use_posteriors in ([False, True] if has_posteriors else [False]):
                selected_groups = list(combo) + (["posteriors"] if use_posteriors else [])
                pieces_dev = []
                pieces_test = []
                feature_names = []
                for group in selected_groups:
                    features = groups[group]
                    pieces_dev.append(transform_group(dev, group, features))
                    pieces_test.append(transform_group(test, group, features))
                    feature_names.extend(features)

                x_dev = pd.concat(pieces_dev, axis=1)
                x_test = pd.concat(pieces_test, axis=1)
                dev_mask = np.isfinite(x_dev.to_numpy()).all(axis=1) & np.isfinite(
                    dev[target].to_numpy()
                )
                test_mask = np.isfinite(x_test.to_numpy()).all(axis=1) & np.isfinite(
                    test[target].to_numpy()
                )
                if dev_mask.sum() < 2 or test_mask.sum() < 2:
                    continue

                model = LinearRegression()
                model.fit(x_dev.loc[dev_mask], dev.loc[dev_mask, target])
                predictions = model.predict(x_test.loc[test_mask])
                truth = test.loc[test_mask, target]
                rho = stats.spearmanr(predictions, truth).statistic
                tau = stats.kendalltau(predictions, truth).statistic

                eligible_singles = singles[
                    (singles["Prediction Name"] == prediction_name)
                    & singles["Feature"].isin(feature_names)
                ].copy()
                eligible_singles["Spearman"] = pd.to_numeric(
                    eligible_singles["Spearman"], errors="coerce"
                )
                eligible_singles = eligible_singles[
                    np.isfinite(eligible_singles["Spearman"])
                ]
                if eligible_singles.empty:
                    best_feature = None
                    best_rho = float("nan")
                    best_tau = float("nan")
                else:
                    best = eligible_singles.loc[eligible_singles["Spearman"].idxmax()]
                    best_feature = best["Feature"]
                    best_rho = best["Spearman"]
                    best_tau = best["Kendall"]

                rows.append(
                    {
                        "Prediction Name": prediction_name,
                        "QA Task": task,
                        "Retriever": retriever,
                        "Top-Retrieved Docs": k,
                        "Feature Groups": "+".join(combo),
                        "Use Postgen": use_posteriors,
                        "Features": ",".join(feature_names),
                        "Rho": float(rho),
                        "Tau": float(tau),
                        "Best Single Signal": best_feature,
                        "Best Single Rho": best_rho,
                        "Best Single Tau": best_tau,
                        "Number of Queries": int(test_mask.sum()),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "Prediction Name",
            "QA Task",
            "Retriever",
            "Top-Retrieved Docs",
            "Feature Groups",
            "Use Postgen",
            "Features",
            "Rho",
            "Tau",
            "Best Single Signal",
            "Best Single Rho",
            "Best Single Tau",
            "Number of Queries",
        ],
    )


def write_results(
    frame: pd.DataFrame,
    path: Path,
    key_columns: list[str],
    overwrite: bool,
) -> None:
    if path.is_file() and not overwrite:
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=key_columns, keep="last")
    frame.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        raise SystemExit("--k must be a positive integer.")

    dev_features = load_feature_csv(args.dev_features, "Development feature CSV")
    test_features = load_feature_csv(args.test_features, "Test feature CSV")

    dev = attach_targets(
        dev_features,
        args.dev_zero_eval,
        args.dev_k_eval,
        args.task,
        split_label="development",
    )
    test = attach_targets(
        test_features,
        args.test_zero_eval,
        args.test_k_eval,
        args.task,
        split_label="test",
    )

    group_candidates = load_feature_groups(
        args.dev_features, args.test_features, list(test.columns)
    )
    groups = valid_common_features(
        dev, test, group_candidates, include_qv_qpp=args.include_qv_qpp
    )
    feature_columns = list(
        dict.fromkeys(feature for values in groups.values() for feature in values)
    )
    if not feature_columns:
        raise SystemExit("No valid common feature columns remain after input validation.")

    singles = single_correlations(test, feature_columns)
    singles.insert(0, "k", args.k)
    singles.insert(1, "retriever", args.retriever)
    singles.insert(2, "task", args.task)

    combinations = combination_correlations(
        dev, test, groups, singles, args.task, args.retriever, args.k
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    single_path = args.output_dir / "single_feature_correlations.csv"
    group_path = args.output_dir / "feature_group_correlations.csv"
    write_results(
        singles,
        single_path,
        ["k", "retriever", "task", "Prediction Name", "Feature"],
        args.overwrite,
    )
    write_results(
        combinations,
        group_path,
        [
            "Prediction Name",
            "QA Task",
            "Retriever",
            "Top-Retrieved Docs",
            "Feature Groups",
            "Use Postgen",
        ],
        args.overwrite,
    )
    print(single_path)
    print(group_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())