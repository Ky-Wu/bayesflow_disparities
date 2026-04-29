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

# 2. Run baseline MCMC analyses

echo "----------------------------------------------------------------"
echo "Running baseline MCMC analyses"
echo "----------------------------------------------------------------"
cd src 
echo "Running src/RDA_rstan.py from: $(pwd)"

Rscript RDA_rstan.R
cd "$PROJECT_ROOT"

# 3. Analyze Dataset (assuming neural posterior estimator is already trained)

echo "----------------------------------------------------------------"
echo "Applying ABI Network to Boundary Detection for Lung Cancer Data"
echo "----------------------------------------------------------------"
echo "Running src/RDA_analyze_data.py from: $(pwd)"

python -u -m src.RDA_analyze_data

# 4. Compute RDETs

echo "----------------------------------------------------------------"
echo "Computing RDETs for Reported Disparities"
echo "----------------------------------------------------------------"
echo "Running src/RDA_compute_RDETs.py from: $(pwd)"

python -u -m src.RDA_compute_RDETs

# 5. Draw figures for manuscript

echo "----------------------------------------------------------------"
echo "Drawing Manuscript Figures"
echo "----------------------------------------------------------------"
cd src
echo "Running src/RDA_draw_figures.R from: $(pwd)"

Rscript RDA_draw_figures.R