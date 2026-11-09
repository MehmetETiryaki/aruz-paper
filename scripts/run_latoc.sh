#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")/.."
${LATOC_PYTHON:-python} scripts/run_latoc.py --device cuda "$@"
