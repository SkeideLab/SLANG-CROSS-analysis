#!/bin/bash -l

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
deriv_dir=$1
# analy_dir=$2
analy_dir="/ptmp/kazma/SLANG-CROSS-analysis"

# Load Apptainer for running containerized commands
module load apptainer

# Activate conda environment
module load anaconda/3/2023.03
conda activate SLANG

# Go into the BIDS dataset
bids_dir="$deriv_dir/.."
cd "$bids_dir"

neurodesk_container="$analy_dir/containers/neurodesk.simg"

apptainer exec --cleanenv \
  -B "$bids_dir":"$bids_dir" \
  -B "$analy_dir":"$analy_dir" \
  "$neurodesk_container" \
  python "$analy_dir/scripts/univariate.py"

# And we're done
echo SUCCESS
