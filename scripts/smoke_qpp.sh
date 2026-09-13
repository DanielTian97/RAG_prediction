#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

SPARSE_ARTIFACT="${NQ_SPARSE_ARTIFACT:-pyterrier/ragwiki-terrier}"
DENSE_ARTIFACT="${NQ_DENSE_ARTIFACT:-pyterrier/ragwiki-e5.flex}"
RUN_DENSE="${RUN_DENSE_QPP:-0}"

echo "QPP artifact smoke test"
echo "Sparse artifact: $SPARSE_ARTIFACT"
echo "Dense artifact:  $DENSE_ARTIFACT"
echo "Dense enabled:   $RUN_DENSE"
echo

python - <<PY
import numpy as np
import pandas as pd
import pyterrier as pt
from qpp import QPPCalculator

artifact_name = "${SPARSE_ARTIFACT}"
print(f"Loading sparse artifact: {artifact_name}")
index = pt.Artifact.from_hf(artifact_name)
retrieval = pd.DataFrame({
    "qid": ["1", "1", "1"],
    "query": ["capital france"] * 3,
    "docno": ["synthetic-1", "synthetic-2", "synthetic-3"],
    "rank": [0, 1, 2],
    "score": [3.0, 2.0, 1.0],
})
out = QPPCalculator(index=index).compute(retrieval, k=3)
assert {"maxScore", "nqc"}.issubset(out.columns)
assert np.isfinite(out[["maxScore", "nqc"]].to_numpy()).all()
print(out.to_string(index=False))
print("[PASS] sparse RagWiki artifact + NQC")
PY

if [ "$RUN_DENSE" = "1" ]; then
  python - <<PY
import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_dr
from qpp import QPPCalculator

artifact_name = "${DENSE_ARTIFACT}"
print(f"Loading dense artifact: {artifact_name}")
dense = pt.Artifact.from_hf(artifact_name)
encoder = pyterrier_dr.E5()

# Use actual document ids from a tiny retrieval against the dense artifact so
# vec_loader() receives valid docnos. This may download/model-cache resources.
query = pd.DataFrame({"qid": ["1"], "query": ["capital of france"]})
retriever = encoder >> dense
retrieval = retriever(query)
retrieval = retrieval.sort_values(["qid", "rank"]).head(10).copy()
assert len(retrieval) >= 3, "Dense artifact returned fewer than three documents"

out = QPPCalculator(dense_index=dense, query_encoder=encoder).compute(
    retrieval,
    k=min(3, len(retrieval)),
    include_dense=True,
)
assert {"maxScore", "a_ratio", "spatial"}.issubset(out.columns)
assert np.isfinite(out[["maxScore", "a_ratio", "spatial"]].to_numpy()).all()
print(out.to_string(index=False))
print("[PASS] dense RagWiki artifact + E5 + a_ratio/spatial")
PY
else
  echo "[SKIP] dense RagWiki artifact (~64.5 GB). Run with RUN_DENSE_QPP=1 when desired."
fi
