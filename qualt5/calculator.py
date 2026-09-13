"""QualT5 document-quality feature helper."""
from __future__ import annotations

import numpy as np
import pandas as pd


class QualT5Calculator:
    """Compute individual and integrated-context QualT5 signals."""

    def __init__(self, text_loader, model_name: str = "pyterrier-quality/qt5-small", model=None):
        self.text_loader = text_loader
        if model is None:
            from pyterrier_quality import QualT5

            model = QualT5(model_name)
        self.model = model

    @staticmethod
    def _quality_column(frame: pd.DataFrame) -> pd.Series:
        if "quality" not in frame.columns:
            raise ValueError("QualT5 output does not contain a 'quality' column.")
        # The paper pipeline exponentiated the model's log-quality output.
        return np.exp(pd.to_numeric(frame["quality"], errors="coerce"))

    def _score_documents(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score text with QualT5 using a strict PyTerrier document frame.

        Newer pyterrier-alpha validation rejects query-side columns such as
        ``qid`` and ``query`` when a transformer declares a document frame.
        QualT5 itself only consumes ``text`` and preserves ``docno``, so keep
        its input intentionally minimal and independent of the retrieval frame.
        """
        if "text" not in frame.columns:
            raise ValueError("QualT5 input does not contain a 'text' column.")

        if "docno" in frame.columns:
            docnos = frame["docno"].astype(str).values
        else:
            docnos = [f"doc_{i}" for i in range(len(frame))]

        model_input = pd.DataFrame({
            "docno": docnos,
            "text": frame["text"].astype(str).values,
        })
        return self.model(model_input)

    def compute(self, retrieval: pd.DataFrame, k: int, batch_size: int = 16) -> pd.DataFrame:
        rows = []
        integrated_inputs = []

        for qid, group in retrieval.groupby("qid", sort=False):
            selected = group[group["rank"] < k].copy()
            loaded = self.text_loader(selected)
            scored = self._score_documents(loaded)
            qualities = self._quality_column(scored)

            row = {
                "qid": str(qid),
                "max(docQual)": float(qualities.max()),
                "min(docQual)": float(qualities.min()),
                "avg(docQual)": float(qualities.mean()),
            }
            rows.append(row)

            texts = list(loaded["text"].values)
            integrated_text = "".join(
                f'Context {i + 1}: "{text}";\n' for i, text in enumerate(texts)
            )
            integrated_inputs.append(
                (str(qid), f"{qid}_itg_{k}", integrated_text)
            )

        integrated_values = {}
        for start in range(0, len(integrated_inputs), batch_size):
            current = integrated_inputs[start : start + batch_size]
            batch = pd.DataFrame(
                [(docno, text) for _, docno, text in current],
                columns=["docno", "text"],
            )
            scored = self._score_documents(batch)
            values = self._quality_column(scored)
            for (qid, _, _), value in zip(current, values):
                integrated_values[qid] = float(value)

        result = pd.DataFrame(rows)
        result["itg(docQual)"] = result["qid"].map(integrated_values)
        return result
