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

# 2. Clean and aggregate data

echo "----------------------------------------------------------------"
echo "Cleaning and setting up Lung Cancer Dataset"
echo "----------------------------------------------------------------"

python -u -m src.RDA_data_setup

# 3. Fit chained posterior approximators

echo "----------------------------------------------------------------"
echo "Fitting Posterior Approximator to Lung Cancer Data"
echo "----------------------------------------------------------------"

python -u -m src.RDA_fit_joint_network.py