"""QPP feature helper used by the unified feature pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd


class QPPCalculator:
    """Compute query-performance-prediction features from loaded inputs.

    Classical score-based features are computed directly from a retrieval
    dataframe. Sparse and dense PyTerrier artifacts can be supplied for NQC
    and A-Pair-Ratio respectively. Paper-specific learned QPP signals can be
    supplied as an already-loaded long-form dataframe and are merged into the
    same wide table.
    """

    def __init__(self, index=None, dense_index=None, query_encoder=None):
        self.index = index
        self.dense_index = dense_index
        self.query_encoder = query_encoder

    @staticmethod
    def _max_score(group: pd.DataFrame) -> float:
        row = group.loc[group["rank"].idxmin()]
        return float(row["score"])

    def _terrier_index_obj(self):
        """Return the underlying Java Terrier index for sparse-index statistics.

        ``pt.Artifact.from_hf('pyterrier/ragwiki-terrier')`` returns the modern
        PyTerrier ``TerrierIndex`` wrapper, while older local experiments passed
        the Java index object returned by ``IndexFactory.of`` directly. Supporting
        both keeps the helper compatible with the public artifact and legacy local
        MS MARCO indices.
        """
        if self.index is None:
            raise ValueError("NQC requires a PyTerrier sparse index.")
        if hasattr(self.index, "index_obj"):
            return self.index.index_obj()
        return self.index

    def _max_idf(self, query: str) -> float:
        import pyterrier as pt

        index = self._terrier_index_obj()
        stats = index.getCollectionStatistics()
        doc_num = stats.getNumberOfDocuments()
        stemmer = pt.TerrierStemmer.porter
        lexicon = index.getLexicon()
        dfs = []
        for token in str(query).split():
            term = stemmer.stem(token)
            try:
                dfs.append(lexicon[term].getDocumentFrequency())
            except Exception:
                dfs.append(1)
        dfs = np.asarray(dfs, dtype=float)
        return float(np.max(np.log((doc_num - dfs + 0.5) / (dfs + 0.5))))

    def _nqc(self, group: pd.DataFrame, depth: int = 100) -> float:
        group = group.sort_values("score", ascending=False).head(depth)
        variance = float(np.var(group["score"].to_numpy()))
        return variance * self._max_idf(group["query"].iloc[0])

    def _a_ratio(self, group: pd.DataFrame, depth: int = 50) -> float:
        if self.dense_index is None:
            raise ValueError("a_ratio requires a dense PyTerrier index with vec_loader().")
        from sklearn.metrics.pairwise import cosine_similarity

        group = group[group["rank"] < min(depth, len(group))].copy()
        doc_vecs = self.dense_index.vec_loader()(group)["doc_vec"].values
        emb = np.vstack(doc_vecs)
        similarities = cosine_similarity(emb)
        scores = group["score"].to_numpy()[:, None]
        weighted = similarities @ (scores @ scores.T)
        n = len(group)
        top_n = max(1, int(np.ceil(0.1 * n)))
        top = weighted[:top_n, :top_n]
        tail_start = int(0.2 * n)
        tail = weighted[tail_start:, tail_start:]
        return float(np.mean(top) / np.mean(tail))

    def _spatial(self, group: pd.DataFrame, k: int) -> float:
        if self.dense_index is None or self.query_encoder is None:
            raise ValueError("spatial QPP requires dense_index and query_encoder.")
        selected = group[group["rank"] < k]
        q = self.query_encoder(selected[["qid", "query"]].iloc[:1])["query_vec"].values[0]
        docs = self.dense_index.vec_loader()(selected)["doc_vec"].values
        emb = np.vstack([q, *docs])
        edge = np.max(emb, axis=0) - np.min(emb, axis=0)
        return float(-np.sum(np.log(edge)))

    @staticmethod
    def prepare_precomputed(frame: pd.DataFrame) -> pd.DataFrame:
        required = {"qid", "qpp_method", "qpp_estimate"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Precomputed QPP dataframe is missing columns: {sorted(missing)}")

        frame = frame.copy()
        frame["qid"] = frame["qid"].astype(str)
        frame["qpp_method"] = frame["qpp_method"].astype(str)
        frame["qpp_estimate"] = pd.to_numeric(frame["qpp_estimate"], errors="raise")

        duplicate = frame.duplicated(subset=["qid", "qpp_method"], keep=False)
        if duplicate.any():
            examples = (
                frame.loc[duplicate, ["qid", "qpp_method"]]
                .drop_duplicates()
                .head(5)
                .astype(str)
                .agg(":".join, axis=1)
                .tolist()
            )
            raise ValueError(
                "Precomputed QPP must contain one value per (qid, qpp_method); "
                f"duplicates include {examples}"
            )

        # qid is the stable experiment identifier. Query text is metadata and is
        # intentionally not part of the join key, because harmless text-formatting
        # differences should not discard otherwise valid historical QPP values.
        wide = frame.pivot(index="qid", columns="qpp_method", values="qpp_estimate")
        return wide.reset_index().rename_axis(None, axis=1)

    def compute(
        self,
        retrieval: pd.DataFrame,
        k: int,
        precomputed: pd.DataFrame | None = None,
        include_dense: bool = False,
    ) -> pd.DataFrame:
        retrieval = retrieval.copy()
        retrieval["qid"] = retrieval["qid"].astype(str)
        base = retrieval[["qid", "query"]].drop_duplicates().copy()
        grouped = retrieval.groupby("qid", sort=False)
        max_scores = grouped.apply(self._max_score).rename("maxScore")
        base = base.merge(max_scores, left_on="qid", right_index=True, how="left")

        if self.index is not None:
            nqc = grouped.apply(lambda g: self._nqc(g, 100)).rename("nqc")
            base = base.merge(nqc, left_on="qid", right_index=True, how="left")

        if include_dense and self.dense_index is not None:
            a_ratio = grouped.apply(self._a_ratio).rename("a_ratio")
            base = base.merge(a_ratio, left_on="qid", right_index=True, how="left")
            if self.query_encoder is not None:
                spatial = grouped.apply(lambda g: self._spatial(g, k)).rename("spatial")
                base = base.merge(spatial, left_on="qid", right_index=True, how="left")

        if precomputed is not None:
            loaded = self.prepare_precomputed(precomputed)
            duplicate_features = [
                c for c in loaded.columns if c in base.columns and c != "qid"
            ]
            if duplicate_features:
                base = base.drop(columns=duplicate_features)
            base = base.merge(loaded, on="qid", how="left", validate="one_to_one")

        return base
