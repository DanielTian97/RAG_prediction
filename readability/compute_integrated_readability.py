#!/usr/bin/env python3
"""Compute readability features over concatenated top-k retrieved context."""

import argparse
from pathlib import Path

import nltk
import pandas as pd
import pyterrier as pt
from readability import Readability
from tqdm import tqdm

METRICS = {
    "Dale Chall": "dale_chall",
    "Spache": "spache",
    "Flesch-Kincaid": "flesch_kincaid",
    "Flesch": "flesch",
    "Gunning Fog": "gunning_fog",
    "Coleman Liau": "coleman_liau",
    "ARI": "ari",
    "Linsear Write": "linsear_write",
    "SMOG": "smog",
}


def load_index(dataset: str, msmarco_index: Path):
    if "nq" in dataset:
        return pt.Artifact.from_hf("pyterrier/ragwiki-terrier")
    return pt.IndexFactory.of(pt.IndexRef.of(str(msmarco_index)))


def calculate_readability(text: str, qid, docno: str):
    readability = Readability(text)
    rows = []

    for name, method_name in METRICS.items():
        try:
            result = getattr(readability, method_name)()
            score = result.score
            grade_level = getattr(result, "grade_levels", [])
        except Exception:
            score = -1
            grade_level = []
        rows.append([qid, docno, name, score, grade_level])

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--retrieval-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("readability/readability_res"))
    parser.add_argument(
        "--msmarco-index",
        type=Path,
        default=Path("/mnt/indices/msmarco-passage.terrier/"),
    )
    args = parser.parse_args()

    nltk.download("punkt_tab", quiet=True)
    if not pt.java.started():
        pt.java.init()

    retrieval_csv = args.retrieval_csv or Path(
        f"../../rag_utility/res/{args.retriever}_{args.dataset}.csv"
    )
    retrieval = pd.read_csv(retrieval_csv)
    index = load_index(args.dataset, args.msmarco_index)
    text_loader = index.text_loader(["text"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        f"integrated_readability_{args.dataset}_{args.retriever}_top_{args.top_k}.csv"
    )

    try:
        existing_qids = set(pd.read_csv(output).qid.unique())
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing_qids = set()

    for qid in tqdm(retrieval.qid.unique()):
        if qid in existing_qids:
            continue

        selected = retrieval[(retrieval.qid == qid) & (retrieval["rank"] < args.top_k)]
        loaded = text_loader(selected)
        context = "".join(loaded.text.values)
        docno = f"{qid}_integrated_{args.top_k}"
        rows = calculate_readability(context, qid, docno)

        frame = pd.DataFrame(
            rows,
            columns=["qid", "docno", "readability_metric", "score", "grade"],
        )
        frame.to_csv(output, mode="a", index=False, header=not output.exists())


if __name__ == "__main__":
    main()
