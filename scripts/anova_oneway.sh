#!/bin/bash -l

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
analy_dir=$1
contrast=$2
space=$3
smooth_size=$4
grade1=($5)
grade2=($6)
grade4=($7)
factor_level=$8

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

# file name
file="${contrast}_beta_afni+orig"

for s in "${grade1[@]}"; do
    clean_s="${s#\{}"   # remove leading {
    clean_s="${clean_s%\}}" # remove trailing }
    echo "-dset 1 $analy_dir/derivatives/$clean_s/glm/$space/FWHM_${smooth_size}/$file"
done

# Initialize empty strings for each level
dset_grade1=""
dset_grade2=""
dset_grade4=""

# Grade 1 subjects
for s in "${grade1[@]}"; do
    clean_s="${s#\{}"       # remove leading {
    clean_s="${clean_s%\}}"  # remove trailing }
    dset_grade1+=" -dset 1 $analy_dir/derivatives/$clean_s/glm/$space/FWHM_${smooth_size}/$file"
done

# Grade 2 subjects
for s in "${grade2[@]}"; do
    clean_s="${s#\{}"
    clean_s="${clean_s%\}}"
    dset_grade2+=" -dset 2 $analy_dir/derivatives/$clean_s/glm/$space/FWHM_${smooth_size}/$file"
done

# Grade 4 subjects
for s in "${grade4[@]}"; do
    clean_s="${s#\{}"
    clean_s="${clean_s%\}}"
    dset_grade4+=" -dset 3 $analy_dir/derivatives/$clean_s/glm/$space/FWHM_${smooth_size}/$file"
done

# Bind mounts: make your job_dir and heuristic visible inside the container
apptainer exec \
  --bind "$analy_dir":"$analy_dir" \
  "$afni_container" \
    3dANOVA2 \
    -levels $factor_level \
    $dset_grade1 \
    $dset_grade2 \                    
    $dset_grade4 \            
    -voxel 12 34 56 \
    -ftr $outdir/anova_ftest \
    -mean 1 $outdir/onesample_ttest_grade1 \
    -mean 2 $outdir/onesample_ttest_grade2 \ 
    -mean 3 $outdir/onesample_ttest_grade3 \                    
    -diff 1 2 $outdir/twosample_ttest_grade1vs2 \
    -diff 1 3 $outdir/twosample_ttest_grade1vs4 \
    -diff 2 3 $outdir/twosample_ttest_grade2vs4 \
    -contr 0.5 0.5 -1.0 $outdir/contrast_grade1and2vs4 \
    -bucket $outdir/anova_bucket


# # Clean up everything
# chmod -R +wrx "$job_dir"
# rm -rf "$job_dir"

# # And we're done
echo SUCCESS
