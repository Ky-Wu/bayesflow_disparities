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

# 2. Configuration Parameters
# Using descriptive names makes the Python call easier to read
SHP_PATH="data/cb_2017_us_county_500k/cb_2017_us_county_500k.shp"
REGION="CA"
P_VALUE="7"
FIX_X="True"
MODEL_NAME="ca_fixedXp7"
LAMBDA_RHO="0.03"
CORRUPT_RESIDUAL="True"
THETA_ISOTROPIC="True"
OUTPUT_DIR="output/CA_chained_sim/"

# 3. Safety Check
if [[ ! -f "$SHP_PATH" ]]; then
    echo "Error: Shapefile not found at $SHP_PATH"
    exit 1
fi

# 4. Execution
echo "----------------------------------------------------------------"
echo "Fitting Model: $MODEL_NAME"
echo "Region:         $REGION"
echo "----------------------------------------------------------------"

# We use -u to ensure the python output isn't buffered in logs
python -u -m src.sim_chained_networks \
    "$SHP_PATH" \
    "$REGION" \
    "$P_VALUE" \
    "$FIX_X" \
    "$MODEL_NAME" \
    "$LAMBDA_RHO" \
    "$CORRUPT_RESIDUAL" \
    "$THETA_ISOTROPIC" \
    "$OUTPUT_DIR"