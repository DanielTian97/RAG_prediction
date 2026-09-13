#!/usr/bin/env python3
"""Compute mean answer-token posterior scores from zero-shot generations."""

import argparse
import ast
import json
import pickle
from pathlib import Path

import numpy as np


def load_answers(paths: list[Path]) -> dict:
    answers = {}
    for path in paths:
        with path.open() as f:
            answers.update(json.load(f))
    return answers


def parse_probabilities(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return value


def compute_mean_posteriors(answers: dict) -> dict[str, float]:
    posterior_dict = {}
    for qid, record in answers.items():
        answer_samples = record["0"]
        mean_probs = [
            float(np.mean(parse_probabilities(sample["probs"])))
            for sample in answer_samples.values()
        ]
        posterior_dict[str(qid)] = float(np.mean(mean_probs))
    return posterior_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Generation JSON files to merge")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("posteriors/res/dl_mean_posteriors.pkl"),
        help="Output pickle path",
    )
    args = parser.parse_args()

    answers = load_answers(args.inputs)
    posteriors = compute_mean_posteriors(answers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(posteriors, f)


if __name__ == "__main__":
    main()
