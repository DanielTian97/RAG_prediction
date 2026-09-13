"""Post-generation posterior feature helper."""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd


class PosteriorCalculator:
    """Extract mean answer-token posterior confidence from loaded generations."""

    def __init__(self, answer_key: str = "0", call_key: str = "0"):
        self.answer_key = answer_key
        self.call_key = call_key

    @staticmethod
    def _mean_probs(value) -> float:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return float(np.mean(value))

    def compute(self, generations: dict) -> pd.DataFrame:
        rows = []
        for qid, item in generations.items():
            try:
                answer = item[self.answer_key][self.call_key]
                score = self._mean_probs(answer["probs"])
            except (KeyError, TypeError, ValueError, SyntaxError):
                continue
            rows.append([str(qid), score])
        return pd.DataFrame(rows, columns=["qid", "prob(k)"])
