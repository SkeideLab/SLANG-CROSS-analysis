# -------------------------------
# %% === Libraries ===
# -------------------------------
import numpy as np
import nibabel as nib
from pathlib import Path
import templateflow.api as tflow
from nilearn.image import resample_to_img
from matplotlib import colors
from nilearn.plotting import plot_glass_brain, plot_stat_map, show
from nilearn.plotting import plot_stat_map
from nilearn import plotting, image, datasets



# -------------------------------
# %% === Paths & parameters ===
# -------------------------------
ANALY_DIR = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
DERIV_DIR = ANALY_DIR / 'derivatives'
FIG_DIR   = ANALY_DIR / 'figures'
OUT_DIR   = ANALY_DIR / 'outputs'
TMPL_DIR  = ANALY_DIR / 'templates'

SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_words' # 'images_words', 'images_pseudo'
GRADES         = '1' # 1, 2, 4
MASK           = 'VWFA' # ventral, VWFA
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
CORRECTION     = 'fdr' # fdr, fpr, rft
P_CORRECTION   = 0.05 # 0.001, 0.05
CLUSTER_SIZE   = 100



# -------------------------------
# %% === Load the Mask ===
# -------------------------------

# file name
f_name_left    = f"left_{MASK}_pediatric_MNI.nii.gz"
f_name_right   = f"right_{MASK}_pediatric_MNI.nii.gz"

# path name
f_path_left    = TMPL_DIR / 'mask' / f_name_left
f_path_right   = TMPL_DIR / 'mask' / f_name_right

# load the mask
mask_left_img  = nib.load(f_path_left)
mask_right_img = nib.load(f_path_right)

# convert the mask to numpy
mask_left      = mask_left_img.get_fdata().astype(np.uint8)
mask_right     = mask_right_img.get_fdata().astype(np.uint8)



# -------------------------------
# %% === Load the statistical map ===
# -------------------------------

# file name
f_name  = f"GRADE-{GRADES}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"

# path name
f_path  = OUT_DIR / 'glm' / SPACE / CONTRASTS / f_name

# load the map
sig_map_img = nib.load(f_path)

# convert the map to numpy
sig_map     = sig_map_img.get_fdata().astype(np.uint8)



# -------------------------------
# %% === Count the overlap ===
# -------------------------------

# count the mask size
n_left_mask  = np.sum(mask_left > 0)
n_right_mask = np.sum(mask_right > 0)

# Overlap counts
n_left_overlap   = np.sum((sig_map > 0) & (mask_left > 0))
n_right_overlap  = np.sum((sig_map > 0) & (mask_right > 0))

# percentage 
pct_left  = 100.0 * (n_left_overlap  / n_left_mask)
pct_right = 100.0 * (n_right_overlap / n_right_mask)

print(f"Significant voxels in LEFT {MASK} mask : {n_left_overlap} ({pct_left:.2f}%)")
print(f"Significant voxels in RIGHT {MASK} mask: {n_right_overlap} ({pct_right:.2f}%)")

# %%
