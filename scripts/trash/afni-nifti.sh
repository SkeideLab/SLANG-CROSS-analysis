#!/bin/bash -l

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
analy_dir=$1
contrast=$2
out_dir=$3
f_name=$4

# Enable use of Singularity containers
module load apptainer

# Activate conda environment
module load anaconda/3/2023.03
conda activate SLANG

# Path to your local afni container
afni_container="$analy_dir/containers/afni-25.3.02.sif"

# convert nii.gz to afni format
apptainer exec \
  --bind "$analy_dir":"$analy_dir" \
  "$afni_container" \
    3dAFNItoNIFTI \
        -prefix ${out_dir}/${f_name} \
        ${out_dir}/onewayANOVA+orig'[1]'

# # And we're done
echo SUCCESS
