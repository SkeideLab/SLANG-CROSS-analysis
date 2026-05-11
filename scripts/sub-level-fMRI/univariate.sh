#!/bin/bash -l

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
bids_dir=$1
analy_dir=$2
conda_env=$3
conda_env_name=$4

# Activate conda environment
source $conda_env # /viper/ptmp/kazma/miniforge3/etc/profile.d/conda.sh
conda activate $conda_env_name #nilearn_glm

# Go into the BIDS dataset
cd "$bids_dir"
  
python "$analy_dir/scripts/sub-level-fMRI/univariate.py"

# And we're done
echo SUCCESS
