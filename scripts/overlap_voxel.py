# %% ===  Load modules ===
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import label, find_objects
# %% ===  FIXED: Parameters ===
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
DEMO_DIR     = ANALY_DIR / 'demographics'

# === Parameters ===
MODEL          = 'glm'
METHOD         = 'parametric' # 'parametric' 'non-parametric'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_words-images_pseudo'
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
P_CORRECTION   = 0.001 # 0.001, 0.05
CLUSTER_SIZE   = 50

grade_four = OUT_DIR / MODEL / SPACE / CONTRASTS / f'GRADE-all_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map_acc.nii.gz'
grade_all = OUT_DIR / MODEL / SPACE / CONTRASTS / f'GRADE-4_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz'

# %%
data_img_all  = nib.load(grade_all)
data_img_four = nib.load(grade_four)
data_all      = data_img_all.get_fdata()
data_four     = data_img_four.get_fdata()

# find the MNI coordinates of a cluster
cluster_mask_all  = data_all > 0
cluster_mask_four = data_four > 0

# Label connected clusters
labeled_array_all, num_clusters_all = label(cluster_mask_all)
print(f"Found {num_clusters_all} clusters")

for i in range(1, num_clusters_all + 1):
    cluster_mask_i = (labeled_array_all == i)
    # count total voxels in this cluster
    total_voxels = cluster_mask_i.sum()

    # overlapping voxels with cluster_mask_four
    overlap_voxels = (cluster_mask_i & cluster_mask_four).sum()

    print(f"Cluster {i}: total={total_voxels}, overlap={overlap_voxels}")
# %%
