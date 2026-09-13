#!/usr/bin/env bash
set -euo pipefail

python -m pip install py-readability-metrics
python -m pip install "git+https://github.com/terrierteam/pyterrier-quality.git"

echo
echo "Installed smoke-test-only missing dependencies:"
echo "  - py-readability-metrics"
echo "  - pyterrier-quality"
