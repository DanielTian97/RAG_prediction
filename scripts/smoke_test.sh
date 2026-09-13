#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PASS=0
FAIL=0
SKIP=0

run_step() {
  local name="$1"
  shift
  echo
  echo "================================================================"
  echo "SMOKE: $name"
  echo "================================================================"
  if "$@"; then
    echo "[PASS] $name"
    PASS=$((PASS + 1))
  else
    rc=$?
    echo "[FAIL] $name (exit $rc)"
    FAIL=$((FAIL + 1))
  fi
}

run_python() {
  local name="$1"
  local code="$2"
  echo
  echo "================================================================"
  echo "SMOKE: $name"
  echo "================================================================"
  if python -c "$code"; then
    echo "[PASS] $name"
    PASS=$((PASS + 1))
  else
    rc=$?
    echo "[FAIL] $name (exit $rc)"
    FAIL=$((FAIL + 1))
  fi
}

echo "Repository: $ROOT_DIR"
echo "Python:     $(python --version 2>&1)"
echo "Executable: $(command -v python)"
echo "Commit:     $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

run_step "compileall" python -m compileall -q analyse.py compute_features.py qpp context_perplexity qualt5 posteriors readability
run_step "repository unit tests" python -m pytest -q

run_python "PosteriorCalculator" '
from posteriors import PosteriorCalculator
import numpy as np

generations = {
    "1": {"0": {"0": {"probs": "[0.2, 0.4, 0.6]"}}},
    "2": {"0": {"0": {"probs": [0.5, 0.7]}}},
}
out = PosteriorCalculator().compute(generations)
assert list(out["qid"].astype(str)) == ["1", "2"]
assert np.isfinite(out["prob(k)"].to_numpy()).all()
print(out.to_string(index=False))
'

run_python "ReadabilityCalculator with py-readability-metrics" '
import numpy as np
import pandas as pd
from readability import ReadabilityCalculator

base = ("Retrieval augmented generation combines search with language model generation. "
        "This synthetic passage exists only to exercise the readability feature calculator. ")
text = base * 18
texts = {"d1": text, "d2": text + " Additional context improves the length of this passage."}

def loader(frame):
    out = frame.copy()
    out["text"] = [texts[str(x)] for x in out["docno"]]
    return out

retrieval = pd.DataFrame({
    "qid": ["1", "1"],
    "query": ["example query", "example query"],
    "docno": ["d1", "d2"],
    "rank": [0, 1],
    "score": [2.0, 1.0],
})
out = ReadabilityCalculator(loader).compute(retrieval, k=2)
assert out.shape[0] == 1
assert out.shape[1] == 37, out.columns.tolist()
assert np.isfinite(out.drop(columns=["qid"]).to_numpy()).any()
print(out.iloc[:, :9].to_string(index=False))
print(f"columns={out.shape[1]}")
'

# The smoke models below are public. Do not let an expired cluster/user token
# turn anonymous public downloads into 401 errors.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

run_python "QualT5Calculator with qt5-tiny" '
import numpy as np
import pandas as pd
from qualt5 import QualT5Calculator

texts = {
    "d1": "Paris is the capital of France.",
    "d2": "France is a country in Western Europe whose capital is Paris.",
}

def loader(frame):
    out = frame.copy()
    out["text"] = [texts[str(x)] for x in out["docno"]]
    return out

retrieval = pd.DataFrame({
    "qid": ["1", "1"],
    "query": ["What is the capital of France?", "What is the capital of France?"],
    "docno": ["d1", "d2"],
    "rank": [0, 1],
    "score": [2.0, 1.0],
})
out = QualT5Calculator(loader, model_name="pyterrier-quality/qt5-tiny").compute(retrieval, k=2, batch_size=2)
expected = {"max(docQual)", "min(docQual)", "avg(docQual)", "itg(docQual)"}
assert expected.issubset(out.columns)
assert np.isfinite(out[list(expected)].to_numpy()).all()
print(out.to_string(index=False))
'

run_python "ContextPerplexityCalculator with tiny-gpt2 on CPU" '
import numpy as np
import pandas as pd
from context_perplexity import ContextPerplexityCalculator

texts = {
    "d1": "Paris is the capital of France.",
    "d2": "France is in Western Europe.",
}

def loader(frame):
    out = frame.copy()
    out["text"] = [texts[str(x)] for x in out["docno"]]
    return out

retrieval = pd.DataFrame({
    "qid": ["1", "1"],
    "query": ["What is the capital of France?", "What is the capital of France?"],
    "docno": ["d1", "d2"],
    "rank": [0, 1],
    "score": [2.0, 1.0],
})
out = ContextPerplexityCalculator(
    loader,
    model_name="sshleifer/tiny-gpt2",
    device="cpu",
).compute(retrieval, k=2)
cols = ["max(perpC)", "min(perpC)", "avg(perpC)", "itg(perpC)"]
assert np.isfinite(out[cols].to_numpy()).all()
print(out.to_string(index=False))
'

run_python "QPP synthetic dense/sparse-free path" '
import numpy as np
import pandas as pd
from qpp import QPPCalculator

retrieval = pd.DataFrame({
    "qid": ["1", "1", "1"],
    "query": ["example query"] * 3,
    "docno": ["d1", "d2", "d3"],
    "rank": [0, 1, 2],
    "score": [3.0, 2.0, 1.0],
})
vectors = {
    "d1": np.asarray([1.0, 0.2, 0.4]),
    "d2": np.asarray([0.1, 1.0, 0.3]),
    "d3": np.asarray([0.6, 0.4, 1.0]),
}
class Dense:
    def vec_loader(self):
        def load(frame):
            out = frame.copy()
            out["doc_vec"] = [vectors[str(x)] for x in out["docno"]]
            return out
        return load

def encoder(frame):
    out = frame.copy()
    out["query_vec"] = [np.asarray([0.3, 0.5, 0.8])] * len(out)
    return out

out = QPPCalculator(dense_index=Dense(), query_encoder=encoder).compute(retrieval, k=3, include_dense=True)
assert np.isfinite(out[["maxScore", "a_ratio", "spatial"]].to_numpy()).all()
print(out.to_string(index=False))
'

echo
echo "================================================================"
echo "SMOKE SUMMARY"
echo "================================================================"
echo "PASS: $PASS"
echo "FAIL: $FAIL"
echo "SKIP: $SKIP"

if [ "$FAIL" -ne 0 ]; then
  exit 1
fi
