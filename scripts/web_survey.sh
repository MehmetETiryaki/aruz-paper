#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY -I scripts/web_survey.py extract
$PY -I scripts/web_survey.py match
$PY -I scripts/web_survey.py catalogue
echo "websurvey COMPLETED"
