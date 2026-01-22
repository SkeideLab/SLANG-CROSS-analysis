#!/bin/bash -l

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
analy_dir=$1
subj=$2
top_dir=$3
top_wri=$4

# Enable use of Singularity containers
module load apptainer

# Activate conda environment
module load anaconda/3/2023.03
conda activate SLANG

# Paths for outputs
outdir="$top_wri/$subj"
mkdir -p "$outdir"

SSW_DIR="$outdir/sswarper_flipcheck"
mkdir -p "$SSW_DIR"


# Path to your local afni container
afni_container="$analy_dir/containers/afni-25.3.02.sif"

# Path to your MNI template
MNItemplate="$analy_dir/templates/adult_MNI_2mm/MNI152_2mm.nii.gz"

FUNCFILES=$(echo /ptmp/kazma/SLANG-CROSS-conversion/$subj/ses-01/func/*.nii.gz)
# ANATFILES=$(echo /ptmp/kazma/SLANG-CROSS-conversion/$subj/ses-01/anat/*.nii.gz)
# FUNCFILES=(/ptmp/kazma/SLANG-CROSS-conversion/$subj/ses-01/func/*.nii.gz)
# FUNCFILE="${FUNCFILES[5]}"
ANATFILES=(/ptmp/kazma/SLANG-CROSS-conversion/$subj/ses-01/anat/*.nii.gz)
ANATFILE="${ANATFILES[@]: -1}"

apptainer exec \
  --bind "$analy_dir":"$analy_dir" \
  --bind "$top_dir":"$top_dir" \
  "$afni_container" \
    align_epi_anat.py \
      -anat "$MNItemplate" \
      -epi "$ANATFILE" \
      -epi_base 0 \
      -epi2anat \
      -cost lpc+ZZ \
      -giant_move \
      -check_flip \
      -suffix _flipcheck
    
# convert nii.gz to afni format
apptainer exec \
  --bind "$analy_dir":"$analy_dir" \
  --bind "$top_dir":"$top_dir" \
  "$afni_container" \
    afni_proc.py \
        -subj_id "$subj" \
        -script "$outdir/proc.$subj" \
        -out_dir "$outdir/$subj.results" \
        -dsets $FUNCFILES \
        -copy_anat "$ANATFILE" \
        -anat_has_skull yes \
        -blocks align volreg regress \
        -tcat_remove_first_trs 0 \
        -align_opts_aea -check_flip -giant_move -cost lpc+ZZ \
        -volreg_align_to MIN_OUTLIER \
        -volreg_align_e2a \
        -regress_run_clustsim no \
        -html_review_style pythonic \
        -execute

# # And we're done
echo SUCCESS