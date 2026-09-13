#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# The tiny public models used here do not require authentication. Avoid stale
# cluster credentials breaking public Hugging Face downloads.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

FIXTURE="$ROOT_DIR/data/examples/e5_k3_qpp_dl_smoke.csv"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rag-pipeline-smoke.XXXXXX")"
if [ "${KEEP_SMOKE_OUTPUT:-0}" != "1" ]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

FEATURE_ROOT="$WORK_DIR/features"
CONDITION_DIR="$FEATURE_ROOT/dl_smoke_e5_k3"
CORR_DIR="$WORK_DIR/correlations"

echo "Repository: $ROOT_DIR"
echo "Python:     $(python --version 2>&1)"
echo "Executable: $(command -v python)"
echo "Commit:     $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Fixture:    $FIXTURE"
echo "Work dir:   $WORK_DIR"

echo
echo "================================================================"
echo "PIPELINE SMOKE: prepare compact inputs"
echo "================================================================"

python - "$FIXTURE" "$WORK_DIR" <<'PY'
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

fixture = Path(sys.argv[1])
work = Path(sys.argv[2])
qpp = pd.read_csv(fixture)
required = {"qid", "query", "qpp_estimate", "qpp_method"}
assert required.issubset(qpp.columns), qpp.columns.tolist()
assert set(qpp["qpp_method"]) == {"nqc", "maxScore"}
assert not qpp.duplicated(["qid", "qpp_method"]).any()

queries = qpp[["qid", "query"]].drop_duplicates().reset_index(drop=True)
assert len(queries) == 10

retrieval_rows = []
doc_dict = {}
generations = {}
zero_eval = {}
k_eval = {}

simple = (
    "This passage gives clear factual information in short sentences. "
    "It uses common words and describes the subject in a direct way. "
)
complex_text = (
    "The retrieved evidence additionally presents contextual qualifications, "
    "historical relationships, explanatory details, and several relevant observations. "
)

for i, row in queries.iterrows():
    qid = str(row["qid"])
    query = str(row["query"])
    for rank in range(3):
        docno = f"{qid}_d{rank}"
        retrieval_rows.append(
            {
                "qid": qid,
                "query": query,
                "docno": docno,
                "rank": rank,
                "score": 1.0 - 0.08 * rank + 0.003 * i,
            }
        )
        # >100 words per passage so all py-readability-metrics scorers can run.
        # The mixture changes across qids/ranks to avoid constant smoke features.
        repetitions = 9 + ((i + rank) % 5)
        doc_dict[docno] = (
            (simple * repetitions)
            + (complex_text * (4 + ((2 * i + rank) % 6)))
            + f" Query identifier {qid}, document rank {rank}."
        )

    p = 0.35 + 0.035 * i
    generations[qid] = {
        "0": {"0": {"probs": [p, min(p + 0.08, 0.95), min(p + 0.14, 0.97)]}}
    }

    p0 = 0.08 + 0.025 * (i % 5) + 0.006 * i
    pk = min(0.93, p0 + 0.08 + 0.035 * i)
    zero_eval[qid] = {"0": {"0": {"qrel_1": {"f1": {"max": p0}}}}}
    k_eval[qid] = {"0": {"0": {"qrel_1": {"f1": {"max": pk}}}}}

pd.DataFrame(retrieval_rows).to_csv(work / "retrieval.csv", index=False)
with (work / "docs.pkl").open("wb") as handle:
    pickle.dump(doc_dict, handle)
(work / "generations.json").write_text(json.dumps(generations, indent=2), encoding="utf-8")
(work / "zero.json").write_text(json.dumps(zero_eval, indent=2), encoding="utf-8")
(work / "k.json").write_text(json.dumps(k_eval, indent=2), encoding="utf-8")

print(f"queries={len(queries)}, retrieval_rows={len(retrieval_rows)}, documents={len(doc_dict)}")
PY

echo
echo "================================================================"
echo "PIPELINE SMOKE: compute_features.py"
echo "================================================================"

python compute_features.py \
  --retrieval-csv "$WORK_DIR/retrieval.csv" \
  --generation-json "$WORK_DIR/generations.json" \
  --doc-dict "$WORK_DIR/docs.pkl" \
  --qpp-precomputed "$FIXTURE" \
  --dataset dl_smoke \
  --retriever e5 \
  --k 3 \
  --output-dir "$FEATURE_ROOT" \
  --families qpp context_perplexity qualt5 posteriors readability \
  --qualt5-model pyterrier-quality/qt5-tiny \
  --perpc-model sshleifer/tiny-gpt2 \
  --device cpu

python - "$CONDITION_DIR" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

condition = Path(sys.argv[1])
expected_files = {
    "qpp.csv",
    "context_perplexity.csv",
    "qualt5.csv",
    "posteriors.csv",
    "readability.csv",
    "features.csv",
    "feature_groups.json",
    "metadata.json",
}
missing = sorted(name for name in expected_files if not (condition / name).is_file())
assert not missing, f"missing outputs: {missing}"

features = pd.read_csv(condition / "features.csv")
assert len(features) == 10, len(features)
assert features["qid"].astype(str).is_unique
assert not features.columns.duplicated().any()
required = {
    "nqc",
    "maxScore",
    "prob(k)",
    "max(perpC)",
    "max(docQual)",
    "max(Flesch)",
}
missing_columns = sorted(required - set(features.columns))
assert not missing_columns, f"missing feature columns: {missing_columns}"
for column in ["nqc", "maxScore", "prob(k)", "max(perpC)", "max(docQual)"]:
    assert np.isfinite(pd.to_numeric(features[column], errors="coerce")).all(), column

manifest = json.loads((condition / "feature_groups.json").read_text(encoding="utf-8"))
assert set(manifest) == {
    "qpp",
    "context_perplexity",
    "qualt5",
    "posteriors",
    "readability",
}
print(f"features.csv: rows={len(features)}, columns={len(features.columns)}")
print("families:", ", ".join(manifest))
PY

echo "[PASS] compute_features.py end-to-end"

echo
echo "================================================================"
echo "PIPELINE SMOKE: analyse.py"
echo "================================================================"

python analyse.py \
  --dev-features "$CONDITION_DIR/features.csv" \
  --test-features "$CONDITION_DIR/features.csv" \
  --dev-zero-eval "$WORK_DIR/zero.json" \
  --dev-k-eval "$WORK_DIR/k.json" \
  --test-zero-eval "$WORK_DIR/zero.json" \
  --test-k-eval "$WORK_DIR/k.json" \
  --task dl \
  --retriever e5 \
  --k 3 \
  --output-dir "$CORR_DIR" \
  --overwrite

python - "$CORR_DIR" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
single = pd.read_csv(root / "single_feature_correlations.csv")
groups = pd.read_csv(root / "feature_group_correlations.csv")
assert not single.empty
assert not groups.empty
assert {"GPP", "RPP"}.issubset(set(single["Prediction Name"]))
assert {"GPP", "RPP"}.issubset(set(groups["Prediction Name"]))
assert (single["retriever"] == "e5").all()
assert (single["k"] == 3).all()
print(f"single correlations: {len(single)} rows")
print(f"group correlations:  {len(groups)} rows")
PY

echo "[PASS] analyse.py end-to-end"
echo
echo "================================================================"
echo "PIPELINE SMOKE PASSED"
echo "================================================================"
if [ "${KEEP_SMOKE_OUTPUT:-0}" = "1" ]; then
  echo "Outputs retained at: $WORK_DIR"
else
  echo "Temporary outputs will be removed. Set KEEP_SMOKE_OUTPUT=1 to retain them."
fi
