# %% [markdown]
# ## fMRI Group-level analysis (2nd-level)
#
# Group-level analysis using Nilearn
# with 4 conditions: visual word, visual pseudoword, audio word, audio pseudoword.

# %%
# ===  Load modules ===
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_glass_brain, plot_stat_map, show
from nilearn.plotting import plot_stat_map
from nilearn import plotting, image
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.datasets import fetch_atlas_aal
import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
import templateflow.api as tflow
from nilearn.glm import cluster_level_inference, threshold_stats_img
from nilearn.mass_univariate import permuted_ols
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_multi_epi_mask

# %%
# ===  FIXED: Parameters ===
CONV_DIR     = Path('/ptmp/kazma/SLANG-CROSS-conversion')
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'

# === Parameters ===
MODEL          = 'glm'
METHOD         = 'parametric' # 'parametric' 'non-parametric'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_words-images_pseudo'
GRADE          = 'all' # 1, 2, 4 or all
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
CORRECTION     = 'fpr' # fdr, fpr 
P_CORRECTION   = 0.001 # 0.001, 0.05
CLUSTER_SIZE   = 10
EXC_SUBJECTS   = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434']
# Retrieve the T1-weighted template for cohort 4 (7.5-13.5yrs) at 2mm resolution
TEMPLATE       =  tflow.get(
    "MNIPediatricAsym",
    cohort="4",       # cohort 4 = age 7.5-13.5
    resolution=2,     # 2mm isotropic
    suffix="T1w"      # T1-weighted image
)
# List of the contrasts
"""             'images_words-images_pseudo' 
                'audios_words-audios_pseudo' 
                'images_words' 
                'audios_words'
                'images_pseudo'
                'audios_pseudo'
                'images_words+audios_words-images_pseudo+audios_pseudo' """

# %% =====================
# list the beta nii.files
if GRADE == 'all':
    subjects      = sorted(DERIV_DIR.glob(f"sub-*"))
else:
    subjects      = sorted(DERIV_DIR.glob(f"sub-{GRADE}*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]
subject_names = [p.name.replace('sub-', '') for p in subjects]

# get the beta paths
beta_lists = []
for sub in subjects:
    path = sub / 'glm' / SPACE / f"FWHM_{int(FWHM_SMOOTHING)}" 
    beta_path = next(path.glob(f"{CONTRASTS}_beta.nii.gz"), None)
    beta_lists.append(beta_path)
beta_paths = [str(p) for p in beta_lists]

# get the behavioral results
acc_lists = []
rt_lists  = []
for sub in subjects:
    path     = sub / 'behavior' / 'excluded' / 'accuracy_summary.csv'
    df       = pd.read_csv(path)
    acc_mean = float(round(df['accuracy_visual'].mean(), 1))
    rt_mean  = float(round(df['RT_visual'].mean(), 1))
    acc_lists.append(acc_mean)
    rt_lists.append(rt_mean)

# get the sex info
sex_lists = []
for sub in subjects:
    path = 

if GRADE=='all':
    # model design
    design_matrix = pd.DataFrame({
        "intercept": [1] * len(beta_paths),
        "accuracy": acc_lists,
        "sex": sex_lists
    })
else:
    # model design
    design_matrix = pd.DataFrame(
        [1] * len(beta_paths),
        columns=["intercept"],
    )

# specify the model
second_level_model = SecondLevelModel(
    smoothing_fwhm=None, 
    n_jobs=2
    )


# % =====================
# Parametric: One-sample t-test
if METHOD == 'parametric':
    second_level_model = second_level_model.fit(
        beta_paths,
        design_matrix=design_matrix,
    )
    z_map = second_level_model.compute_contrast(
        second_level_contrast="intercept",
        output_type="z_score",
    )

    # correction methods
    thresholded_map, threshold = threshold_stats_img(
    z_map,
    two_sided=False,
    alpha=P_CORRECTION,
    cluster_threshold=CLUSTER_SIZE,
    height_control=CORRECTION,
    )

# % =====================
# Non-Parametric: Permutation test
elif METHOD == 'non-parametric':
    # compute group-level mask
    mask_img = compute_multi_epi_mask(beta_paths)

    # Build masker
    nifti_masker = NiftiMasker(
        mask_img=mask_img,
        smoothing_fwhm=None,
        standardize=False,
    )
    nifti_masker.fit(beta_paths)

    # shape: subjects × voxels
    Y = nifti_masker.transform(beta_paths)
    X = np.ones((len(beta_paths), 1))   # intercept-only model

    ols_outputs = permuted_ols(
        X,  # this is equivalent to the design matrix, in array form
        Y,
        model_intercept=False,
        masker=nifti_masker,
        n_perm=10000,  # 100 for the sake of time. Ideally, this should be 10000.
        verbose=1,   # display progress bar
        n_jobs=2,
        output_type='dict',    # can be changed to use more CPUs
    )

    # extract FWE corrected p-values
    FWE_corr_pval     = ols_outputs['logp_max_t']
    FWE_corr_pval_img = nifti_masker.inverse_transform(FWE_corr_pval.reshape(1,-1)) # p-values


# %% =====================  Visualize

p_val = P_CORRECTION
p001_unc = norm.isf(p_val)

# Z-coordinates you want to visualize
if 'images' in CONTRASTS and 'audios' in CONTRASTS:
    z_coords = np.arange(0, 20, 2)
elif 'images' in CONTRASTS:
    z_coords = np.arange(-15, -4, 1)  # from -15 to -5 inclusive
elif 'audios' in CONTRASTS:
    z_coords = np.arange(-5, 10, 2) # from -5 to 5

# prepare the title
if '-' in CONTRASTS:
    contrast = CONTRASTS.replace('-', ' > ')
else:
    contrast = CONTRASTS

# create output dir
dir = f"{FIG_DIR}/{MODEL}"
os.makedirs(dir, exist_ok=True)

if METHOD == 'parametric':
    # Parametric: uncorrected p-value 
    # compute threshold for z-value

    # configure the figure
    plotting_config = {
        "display_mode": "z",
        "cut_coords": z_coords,
        "draw_cross": False,
        "vmax": 5,
        "vmin": 0,
        "cmap": "cold_hot",
    }

    # plot the uncorrrected p-value figure 
    display_unc = plotting.plot_stat_map(
        z_map,
        title=f"Grade-{GRADE} {contrast} (unc p<{P_CORRECTION})",
        bg_img=TEMPLATE,
        threshold=p001_unc,
        **plotting_config,
    )
    display_unc.savefig(f"{dir}/Grade_{GRADE}_z-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}.png", dpi=300)

    # plot the p-value with cluster size figure 
    display_FPR = plotting.plot_stat_map(
        thresholded_map,
        title=f"Grade-{GRADE} {contrast} (cluster-thr < {CLUSTER_SIZE}, p<{P_CORRECTION})",
        bg_img=TEMPLATE,
        threshold=threshold,
        **plotting_config,
    )
    display_FPR.savefig(f"{dir}/Grade_{GRADE}_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_cluster-thr<{CLUSTER_SIZE}_p-val<{P_CORRECTION}_{CORRECTION}.png", dpi=300)


elif METHOD == 'non-parametric':
    
    # non-parametric: permutation test
    display = plotting.plot_glass_brain(
        FWE_corr_pval_img,
        title=f"Grade-{GRADE} {contrast} (FWE p<0.1)",
        threshold=-np.log10(P_CORRECTION),
        colorbar=True,
        display_mode='lyrz',  # show left, sagittal, coronal, axial views
        plot_abs=False,     # keep sign info if you have positive/negative values
        black_bg=False,
    )
    # display.savefig(f"{dir}/Grade_{GRADE}_z-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}_glass.png", dpi=300)
    plotting.show()



# %% ================= supplement: 
# glass brain
display = plotting.plot_glass_brain(
    z_map,
    title=f"Grade-{GRADE} {contrast} (unc p<{P_CORRECTION})",
    threshold=p001_unc,
    colorbar=True,
    display_mode='lyrz',  # show left, sagittal, coronal, axial views
    plot_abs=False,     # keep sign info if you have positive/negative values
    black_bg=False,
)
display.savefig(f"{dir}/Grade_{GRADE}_z-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}_glass.png", dpi=300)
plotting.show()

# %%
# ====== peak coordinates
# parameters
MAP = thresholded_map

# load data
data = MAP.get_fdata()

# find peak voxel value and its index
peak_value = np.nanmax(data)
peak_index = np.unravel_index(np.nanargmax(data), data.shape)

# convert it into mni coordinates
affine = MAP.affine
peak_mni = image.coord_transform(
    peak_index[0], peak_index[1], peak_index[2], affine
)

print(f"Peak voxel intensity: {peak_value}")
print(f"Peak MNI coordinates: {peak_mni}")
# %%
