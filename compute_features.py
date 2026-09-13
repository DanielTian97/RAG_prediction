#!/usr/bin/env python3
"""Compute all RAG prediction features for one dataset/retriever/context setting.

This is the single public entry point for feature construction. It owns input
loading and delegates feature calculation to one helper class per feature
family. Family-level CSVs and one merged feature matrix are written under the
requested output directory.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

FAMILIES = ["qpp", "context_perplexity", "qualt5", "posteriors", "readability"]
RETRIEVAL_COLUMNS = {"qid", "query", "docno", "rank", "score"}
QPP_COLUMNS = {"qid", "qpp_method", "qpp_estimate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-csv", type=Path, required=True)
    parser.add_argument(
        "--generation-json",
        type=Path,
        help="Generation JSON required when the posteriors family is enabled.",
    )
    parser.add_argument(
        "--doc-dict",
        type=Path,
        help=(
            "Optional pickle mapping docno -> passage text. When supplied, this "
            "text source is used for PerpC, QualT5 and readability."
        ),
    )
    parser.add_argument("--dataset", required=True, help="e.g. nq_dev, nq_test, dev_small, dl")
    parser.add_argument("--retriever", required=True, help="e.g. bm25, mt5, e5")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/features"),
        help="Parent directory for feature outputs.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=FAMILIES,
        help="Feature families to compute (default: all).",
    )
    parser.add_argument(
        "--qpp-precomputed",
        type=Path,
        help=(
            "Optional paper-era QPP CSV, primarily for learned signals such as "
            "bertQPP. Values in this file override directly computed features "
            "with the same name."
        ),
    )
    parser.add_argument(
        "--compute-dense-qpp",
        action="store_true",
        help=(
            "Compute A-Pair-Ratio and spatial/DenseQPP using the RagWiki E5 dense "
            "index and its matching PyTerrier-DR E5 query encoder. A local index "
            "can be supplied with --nq-dense-index; otherwise the public Hugging "
            "Face artifact is used."
        ),
    )
    parser.add_argument(
        "--msmarco-index",
        type=Path,
        default=Path("/mnt/indices/msmarco-passage.terrier/"),
        help="Local Terrier index used for non-NQ/MS MARCO experiments.",
    )
    parser.add_argument(
        "--nq-sparse-artifact",
        default="pyterrier/ragwiki-terrier",
        help="Public PyTerrier sparse RagWiki artifact used for NQ text and NQC.",
    )
    parser.add_argument(
        "--nq-dense-index",
        type=Path,
        help=(
            "Optional local RagWiki E5 FlexIndex directory for dense QPP. "
            "When supplied, it takes precedence over --nq-dense-artifact."
        ),
    )
    parser.add_argument(
        "--nq-dense-artifact",
        default="pyterrier/ragwiki-e5.flex",
        help="Public PyTerrier E5 RagWiki artifact used when no local dense index is supplied.",
    )
    parser.add_argument("--qualt5-model", default="pyterrier-quality/qt5-small")
    parser.add_argument("--perpc-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", help="Device for the PerpC causal LM, e.g. cuda:0 or cpu")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"{label} does not exist or is not a file: {path}")
    return path


def load_csv(path: Path, label: str, required_columns: set[str]) -> pd.DataFrame:
    path = require_file(path, label)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Could not read {label} as CSV ({path}): {exc}") from exc
    missing = required_columns - set(frame.columns)
    if missing:
        raise SystemExit(
            f"{label} is missing required columns {sorted(missing)}: {path}"
        )
    return frame


def load_json(path: Path, label: str) -> dict:
    path = require_file(path, label)
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise SystemExit(f"Could not read {label} as JSON ({path}): {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must contain a top-level JSON object: {path}")
    return value


def init_pyterrier():
    import pyterrier as pt

    if not pt.java.started():
        pt.java.init()
    return pt


def is_nq(args: argparse.Namespace) -> bool:
    return "nq" in args.dataset.lower()


def load_text_index(args: argparse.Namespace, pt):
    if is_nq(args):
        return pt.Artifact.from_hf(args.nq_sparse_artifact)
    return pt.IndexFactory.of(pt.IndexRef.of(str(args.msmarco_index)))


def load_qpp_sparse_index(args: argparse.Namespace, pt):
    if is_nq(args):
        return pt.Artifact.from_hf(args.nq_sparse_artifact)
    return pt.IndexFactory.of(pt.IndexRef.of(str(args.msmarco_index)))


def load_qpp_dense_resources(args: argparse.Namespace, pt):
    if not is_nq(args):
        raise SystemExit(
            "--compute-dense-qpp currently uses the RagWiki E5 index and "
            "is therefore supported only for NQ datasets."
        )
    import pyterrier_dr

    local_path = getattr(args, "nq_dense_index", None)
    if local_path is not None:
        local_path = Path(local_path)
        if not local_path.is_dir():
            raise SystemExit(f"Local NQ dense FlexIndex does not exist: {local_path}")
        dense_index = pyterrier_dr.FlexIndex(str(local_path))
    else:
        dense_index = pt.Artifact.from_hf(args.nq_dense_artifact)
    query_encoder = pyterrier_dr.E5()
    return dense_index, query_encoder


def load_doc_dict(path: Path) -> dict[str, str]:
    """Load and validate a pickled mapping from document id to passage text."""
    path = require_file(path, "Document-text pickle")
    try:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    except Exception as exc:
        raise SystemExit(f"Could not read document-text pickle ({path}): {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("--doc-dict must contain a pickled Python dictionary.")

    result = {str(docno): text for docno, text in raw.items()}
    invalid = [docno for docno, text in result.items() if not isinstance(text, str)]
    if invalid:
        sample = ", ".join(invalid[:5])
        raise SystemExit(
            "--doc-dict values must be passage-text strings; invalid docno(s): " + sample
        )
    return result


def make_doc_dict_text_loader(doc_dict: dict[str, str]):
    """Create the small text-loader interface expected by feature helpers."""

    def load(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        docnos = output["docno"].astype(str)
        missing = list(dict.fromkeys(docno for docno in docnos if docno not in doc_dict))
        if missing:
            sample = ", ".join(missing[:5])
            raise KeyError(
                "Document-text dictionary is missing retrieved docno(s): " + sample
            )
        output["text"] = docnos.map(doc_dict)
        return output

    return load


def merge_feature_frames(frames: dict[str, pd.DataFrame], retrieval: pd.DataFrame) -> pd.DataFrame:
    merged = retrieval[["qid", "query"]].drop_duplicates().copy()
    merged["qid"] = merged["qid"].astype(str)

    for family, frame in frames.items():
        if frame.empty:
            continue
        frame = frame.copy()
        frame["qid"] = frame["qid"].astype(str)
        if "query" in frame.columns:
            frame = frame.drop(columns=["query"])
        duplicates = (set(merged.columns) & set(frame.columns)) - {"qid"}
        if duplicates:
            raise ValueError(
                f"Feature-name collision while merging {family}: {sorted(duplicates)}"
            )
        if frame["qid"].duplicated().any():
            examples = frame.loc[frame["qid"].duplicated(), "qid"].astype(str).head(5).tolist()
            raise ValueError(f"Feature family {family} has duplicate qid rows: {examples}")
        merged = merged.merge(frame, on="qid", how="left", validate="one_to_one")
    return merged


def main() -> int:
    args = parse_args()
    families = list(dict.fromkeys(args.families))

    if args.k <= 0:
        raise SystemExit("--k must be a positive integer.")
    if "posteriors" in families and args.generation_json is None:
        raise SystemExit("--generation-json is required when computing posteriors.")
    if args.compute_dense_qpp and "qpp" not in families:
        raise SystemExit("--compute-dense-qpp requires the qpp feature family.")

    retrieval = load_csv(args.retrieval_csv, "Retrieval CSV", RETRIEVAL_COLUMNS).copy()
    retrieval["qid"] = retrieval["qid"].astype(str)
    retrieval["docno"] = retrieval["docno"].astype(str)
    retrieval["rank"] = pd.to_numeric(retrieval["rank"], errors="raise")
    retrieval["score"] = pd.to_numeric(retrieval["score"], errors="raise")
    if retrieval[["qid", "query"]].drop_duplicates()["qid"].duplicated().any():
        raise SystemExit("Retrieval CSV contains more than one query string for a qid.")

    generations = None
    if args.generation_json is not None:
        generations = load_json(args.generation_json, "Generation JSON")

    qpp_precomputed = None
    if args.qpp_precomputed is not None:
        qpp_precomputed = load_csv(args.qpp_precomputed, "Precomputed QPP CSV", QPP_COLUMNS)
        qpp_precomputed["qid"] = qpp_precomputed["qid"].astype(str)

    condition = f"{args.dataset}_{args.retriever}_k{args.k}"
    condition_dir = args.output_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    pt = None
    text_loader = None

    needs_text = any(
        family in families for family in ["context_perplexity", "qualt5", "readability"]
    )

    # NQC requires the sparse index. If it is already present in a precomputed
    # QPP table, a user can reproduce the QPP family without loading that index.
    precomputed_methods = set()
    if qpp_precomputed is not None:
        precomputed_methods = set(qpp_precomputed["qpp_method"].astype(str).unique())
    qpp_needs_sparse = "qpp" in families and "nqc" not in precomputed_methods
    qpp_needs_dense = "qpp" in families and args.compute_dense_qpp

    if qpp_needs_sparse or qpp_needs_dense or (needs_text and args.doc_dict is None):
        pt = init_pyterrier()

    if needs_text:
        if args.doc_dict is not None:
            text_loader = make_doc_dict_text_loader(load_doc_dict(args.doc_dict))
        else:
            text_loader = load_text_index(args, pt).text_loader(["text"])

    if "qpp" in families:
        from qpp import QPPCalculator

        sparse_index = load_qpp_sparse_index(args, pt) if qpp_needs_sparse else None
        dense_index = None
        query_encoder = None
        if qpp_needs_dense:
            dense_index, query_encoder = load_qpp_dense_resources(args, pt)
        calculator = QPPCalculator(
            index=sparse_index,
            dense_index=dense_index,
            query_encoder=query_encoder,
        )
        frames["qpp"] = calculator.compute(
            retrieval,
            args.k,
            precomputed=qpp_precomputed,
            include_dense=qpp_needs_dense,
        )

    if "context_perplexity" in families:
        from context_perplexity import ContextPerplexityCalculator

        calculator = ContextPerplexityCalculator(
            text_loader,
            model_name=args.perpc_model,
            device=args.device,
        )
        frames["context_perplexity"] = calculator.compute(retrieval, args.k)

    if "qualt5" in families:
        from qualt5 import QualT5Calculator

        calculator = QualT5Calculator(text_loader, model_name=args.qualt5_model)
        frames["qualt5"] = calculator.compute(retrieval, args.k)

    if "posteriors" in families:
        from posteriors import PosteriorCalculator

        calculator = PosteriorCalculator()
        frames["posteriors"] = calculator.compute(generations)

    if "readability" in families:
        from readability import ReadabilityCalculator

        calculator = ReadabilityCalculator(text_loader)
        frames["readability"] = calculator.compute(retrieval, args.k)

    group_manifest = {}
    for family, frame in frames.items():
        frame.to_csv(condition_dir / f"{family}.csv", index=False)
        group_manifest[family] = [
            col for col in frame.columns if col not in {"qid", "query"}
        ]

    merged = merge_feature_frames(frames, retrieval)
    merged.to_csv(condition_dir / "features.csv", index=False)
    (condition_dir / "feature_groups.json").write_text(
        json.dumps(group_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    metadata = {
        "dataset": args.dataset,
        "retriever": args.retriever,
        "k": args.k,
        "retrieval_csv": str(args.retrieval_csv),
        "generation_json": str(args.generation_json) if args.generation_json else None,
        "doc_dict": str(args.doc_dict) if args.doc_dict else None,
        "families": families,
        "qpp_precomputed": str(args.qpp_precomputed) if args.qpp_precomputed else None,
        "compute_dense_qpp": args.compute_dense_qpp,
        "nq_sparse_artifact": args.nq_sparse_artifact if is_nq(args) else None,
        "nq_dense_index": str(args.nq_dense_index) if is_nq(args) and args.nq_dense_index else None,
        "nq_dense_artifact": args.nq_dense_artifact if is_nq(args) and args.nq_dense_index is None else None,
    }
    (condition_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(condition_dir / "features.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())