#!/usr/bin/env python3
"""Merge DL19 and DL20 QualT5 outputs into combined DL result files."""

import argparse
from pathlib import Path

import pandas as pd


def merge_results(result_dir: Path, retrievers: list[str], cutoffs: list[int]) -> None:
    suffixes = [""] + [f"_integrated_{k}" for k in cutoffs]
    for retriever in retrievers:
        for suffix in suffixes:
            dl19 = pd.read_csv(result_dir / f"{retriever}_dl_19{suffix}.csv")
            dl20 = pd.read_csv(result_dir / f"{retriever}_dl_20{suffix}.csv")
            output = result_dir / f"{retriever}_dl{suffix}.csv"
            pd.concat([dl19, dl20], axis=0, ignore_index=True).to_csv(output, index=False)
            print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("qualt5/quality_res"),
        help="Directory containing QualT5 CSV result files",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["bm25", "mt5", "e5"],
        help="Retriever result prefixes to merge",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=[2, 3, 5, 7, 10],
        help="Integrated-context top-k cutoffs",
    )
    args = parser.parse_args()
    merge_results(args.result_dir, args.retrievers, args.cutoffs)


if __name__ == "__main__":
    main()
