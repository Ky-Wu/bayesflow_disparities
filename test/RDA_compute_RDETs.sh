#!/usr/bin/env bash

# -e: Exit on error
# -u: Exit if a variable is unset
# -o pipefail: Exit if any command in a pipeline fails
set -euo pipefail

# 1. Setup Directories
# Get the directory of the current script, then move up to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Running from: $(pwd)"

echo "----------------------------------------------------------------"
echo "Computing RDETs for Reported Disparities"
echo "----------------------------------------------------------------"
echo "Running src/RDA_compute_RDETs.py from: $(pwd)"

python -u -m src.RDA_compute_RDETs
