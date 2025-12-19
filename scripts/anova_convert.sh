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

# Bind mounts: make your job_dir and heuristic visible inside the container
# apptainer exec \
#   --bind "$analy_dir":"$analy_dir" \
#   "$afni_container" \
#     3dANOVA \
#     -levels 3 \
#     -dset 1 dataset1_1+orig \
#     -dset 1 dataset1_2+orig \                    
#     -dset 2 dataset2_1+orig \
#     -dset 2 dataset2_2+orig \  
#     -dset 3 dataset2_1+orig \
#     -dset 3 dataset2_2+orig \              
#     -voxel 1234567 \
#     -ftr $outdir/anova_ftest \
#     -mean 1 $outdir/onesample_ttest_grade1 \
#     -mean 2 $outdir/onesample_ttest_grade2 \ 
#     -mean 3 $outdir/onesample_ttest_grade3 \                    
#     -diff 1 2 $outdir/twosample_ttest_grade1vs2 \
#     -diff 1 3 $outdir/twosample_ttest_grade1vs4 \
#     -diff 2 3 $outdir/twosample_ttest_grade2vs4 \
#     -contr 0.5 0.5 -1.0 $outdir/contrast_12avg_minus_4 \
#     -bucket $outdir/anova_bucket


# # Clean up everything
# chmod -R +wrx "$job_dir"
# rm -rf "$job_dir"

# # And we're done
echo SUCCESS
