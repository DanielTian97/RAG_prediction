import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from analyse import attach_targets, load_feature_csv, single_correlations
from compute_features import (
    load_doc_dict,
    load_qpp_dense_resources,
    load_qpp_sparse_index,
    make_doc_dict_text_loader,
    merge_feature_frames,
)
from posteriors import PosteriorCalculator
from qpp import QPPCalculator


class PipelineTests(unittest.TestCase):
    def test_qpp_max_score_without_index(self):
        retrieval = pd.DataFrame(
            {
                "qid": ["1", "1", "2"],
                "query": ["q1", "q1", "q2"],
                "docno": ["d1", "d2", "d3"],
                "rank": [0, 1, 0],
                "score": [2.0, 1.0, 3.5],
            }
        )
        result = QPPCalculator().compute(retrieval, k=2)
        values = dict(zip(result.qid.astype(str), result.maxScore))
        self.assertEqual(values, {"1": 2.0, "2": 3.5})

    def test_qpp_precomputed_without_index(self):
        retrieval = pd.DataFrame(
            {
                "qid": ["1", "1"],
                "query": ["q1", "q1"],
                "docno": ["d1", "d2"],
                "rank": [0, 1],
                "score": [2.0, 1.0],
            }
        )
        precomputed = pd.DataFrame(
            {
                "qid": ["1", "1"],
                "qpp_method": ["nqc", "bertQPP"],
                "qpp_estimate": [0.25, 0.4],
            }
        )
        result = QPPCalculator().compute(retrieval, k=2, precomputed=precomputed)
        self.assertAlmostEqual(result.loc[0, "nqc"], 0.25)
        self.assertAlmostEqual(result.loc[0, "bertQPP"], 0.4)
        self.assertAlmostEqual(result.loc[0, "maxScore"], 2.0)

    def test_qpp_dense_features_with_vector_loader_and_query_encoder(self):
        retrieval = pd.DataFrame(
            {
                "qid": ["1", "1", "1"],
                "query": ["example query"] * 3,
                "docno": ["d1", "d2", "d3"],
                "rank": [0, 1, 2],
                "score": [3.0, 2.0, 1.0],
            }
        )
        vectors = {
            "d1": np.asarray([1.0, 0.2, 0.4]),
            "d2": np.asarray([0.1, 1.0, 0.3]),
            "d3": np.asarray([0.6, 0.4, 1.0]),
        }

        class FakeDenseIndex:
            def vec_loader(self):
                def load(frame):
                    output = frame.copy()
                    output["doc_vec"] = [vectors[str(docno)] for docno in output["docno"]]
                    return output

                return load

        def fake_query_encoder(frame):
            output = frame.copy()
            output["query_vec"] = [np.asarray([0.3, 0.5, 0.8])] * len(output)
            return output

        result = QPPCalculator(
            dense_index=FakeDenseIndex(), query_encoder=fake_query_encoder
        ).compute(retrieval, k=3, include_dense=True)
        self.assertIn("a_ratio", result.columns)
        self.assertIn("spatial", result.columns)
        self.assertTrue(np.isfinite(result.loc[0, "a_ratio"]))
        self.assertTrue(np.isfinite(result.loc[0, "spatial"]))

    def test_nq_qpp_loaders_use_public_ragwiki_artifacts(self):
        requested = []

        class FakeArtifact:
            @staticmethod
            def from_hf(name):
                requested.append(name)
                return f"artifact:{name}"

        fake_pt = SimpleNamespace(Artifact=FakeArtifact)
        args = SimpleNamespace(
            dataset="nq_dev",
            nq_sparse_artifact="pyterrier/ragwiki-terrier",
            nq_dense_artifact="pyterrier/ragwiki-e5.flex",
        )

        sparse = load_qpp_sparse_index(args, fake_pt)
        self.assertEqual(sparse, "artifact:pyterrier/ragwiki-terrier")

        fake_encoder = object()
        fake_pyterrier_dr = types.SimpleNamespace(E5=lambda: fake_encoder)
        with patch.dict(sys.modules, {"pyterrier_dr": fake_pyterrier_dr}):
            dense, encoder = load_qpp_dense_resources(args, fake_pt)

        self.assertEqual(dense, "artifact:pyterrier/ragwiki-e5.flex")
        self.assertIs(encoder, fake_encoder)
        self.assertEqual(
            requested,
            ["pyterrier/ragwiki-terrier", "pyterrier/ragwiki-e5.flex"],
        )

    def test_posterior_calculator(self):
        generations = {
            "1": {"0": {"0": {"probs": "[0.2, 0.4]"}}},
            "2": {"0": {"0": {"probs": [0.5, 0.7]}}},
        }
        result = PosteriorCalculator().compute(generations)
        values = dict(zip(result.qid, result["prob(k)"]))
        self.assertAlmostEqual(values["1"], 0.3)
        self.assertAlmostEqual(values["2"], 0.6)

    def test_document_pickle_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.pkl"
            with path.open("wb") as handle:
                pickle.dump({1: "first passage", "2": "second passage"}, handle)

            doc_dict = load_doc_dict(path)
            self.assertEqual(doc_dict, {"1": "first passage", "2": "second passage"})

            loader = make_doc_dict_text_loader(doc_dict)
            frame = pd.DataFrame({"docno": [1, "2"], "rank": [0, 1]})
            loaded = loader(frame)
            self.assertEqual(list(loaded["text"]), ["first passage", "second passage"])

    def test_document_pickle_loader_rejects_missing_docno(self):
        loader = make_doc_dict_text_loader({"1": "first passage"})
        with self.assertRaises(KeyError):
            loader(pd.DataFrame({"docno": ["2"]}))

    def test_merge_feature_frames(self):
        retrieval = pd.DataFrame(
            {
                "qid": ["1", "2"],
                "query": ["q1", "q2"],
                "docno": ["d1", "d2"],
                "rank": [0, 0],
                "score": [1.0, 1.0],
            }
        )
        frames = {
            "qpp": pd.DataFrame({"qid": ["1", "2"], "nqc": [0.1, 0.2]}),
            "posteriors": pd.DataFrame({"qid": ["1", "2"], "prob(k)": [0.3, 0.4]}),
        }
        result = merge_feature_frames(frames, retrieval)
        self.assertEqual(list(result.columns), ["qid", "query", "nqc", "prob(k)"])
        self.assertEqual(len(result), 2)

    def test_feature_csv_rejects_duplicate_qids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            pd.DataFrame({"qid": ["1", "1"], "signal": [0.1, 0.2]}).to_csv(
                path, index=False
            )
            with self.assertRaises(SystemExit):
                load_feature_csv(path, "Feature CSV")

    def test_attach_targets_and_correlations(self):
        features = pd.DataFrame(
            {
                "qid": ["1", "2", "3"],
                "query": ["q1", "q2", "q3"],
                "signal": [0.1, 0.5, 0.9],
            }
        )
        zero = {
            "1": {"0": {"0": {"F1": 0.0}}},
            "2": {"0": {"0": {"F1": 0.1}}},
            "3": {"0": {"0": {"F1": 0.2}}},
        }
        with_context = {
            "1": {"0": {"0": {"F1": 0.1}}},
            "2": {"0": {"0": {"F1": 0.5}}},
            "3": {"0": {"0": {"F1": 0.9}}},
        }

        with tempfile.TemporaryDirectory() as tmp:
            zero_path = Path(tmp) / "zero.json"
            k_path = Path(tmp) / "k.json"
            zero_path.write_text(json.dumps(zero), encoding="utf-8")
            k_path.write_text(json.dumps(with_context), encoding="utf-8")
            combined = attach_targets(features, [zero_path], [k_path], "nq")

        self.assertEqual(list(combined["f1"]), [0.1, 0.5, 0.9])
        self.assertEqual(list(combined["utility"]), [0.1, 0.4, 0.7])
        correlations = single_correlations(combined, ["signal"])
        gpp = correlations[correlations["Prediction Name"] == "GPP"].iloc[0]
        self.assertAlmostEqual(gpp["Spearman"], 1.0)


if __name__ == "__main__":
    unittest.main()
