#!/bin/bash -l

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
analy_dir=$1
contrast=$2
betapath=$3
afnipath=$4

# Enable use of Singularity containers
module load apptainer

# Activate conda environment
module load anaconda/3/2023.03
conda activate SLANG

# Paths for outputs
outdir="$analy_dir/outputs/anova/$contrast"
mkdir -p "$outdir"

# Path to your local afni container
afni_container="$analy_dir/containers/afni-25.3.02.sif"

# convert nii.gz to afni format
apptainer exec \
  --bind "$analy_dir":"$analy_dir" \
  "$afni_container" \
    3dcopy -overwrite "$betapath" "$afnipath"

# # And we're done
echo SUCCESS
