#!/bin/bash -l

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
analy_dir=$1
contrast=$2
out_dir=$3
deriv_dir=$4
space=$5
smoothing=$6

# specify the output filename
outfile=${out_dir}/onewayANOVA
# remove if it already exists
rm -f ${outfile}+orig.*

# Enable use of Singularity containers
module load apptainer

# Activate conda environment
module load anaconda/3/2023.03
conda activate SLANG

# Path to your local afni container
afni_container="$analy_dir/containers/afni-25.3.02.sif"

apptainer exec \
  --bind "$analy_dir":"$analy_dir" \
  "$afni_container" \
    3dANOVA \
        -levels 3 \
        -dset 1 ${deriv_dir}/sub-101/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-102/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-103/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-104/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-105/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-106/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-107/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-109/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-110/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-112/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-114/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-115/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-117/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-119/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-123/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-127/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-129/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 1 ${deriv_dir}/sub-130/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
                                                                                     \
        -dset 2 ${deriv_dir}/sub-202/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-203/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-204/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-207/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-209/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-210/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-211/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-212/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-213/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-214/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-215/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-216/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-217/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-218/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-219/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-221/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-222/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-223/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 2 ${deriv_dir}/sub-224/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
                                                                                     \
        -dset 3 ${deriv_dir}/sub-401/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-402/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-403/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-404/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-407/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-411/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-412/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-413/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-414/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-415/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-416/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-417/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-418/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-419/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-420/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-425/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-426/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-428/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-429/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-431/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-432/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-433/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-435/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
        -dset 3 ${deriv_dir}/sub-437/glm/${space}/${smoothing}/${contrast}_beta_afni+orig \
                                                                                     \
        -ftr Grades \
        -bucket "$outfile"

# # And we're done
echo SUCCESS