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
from nilearn import plotting, image, datasets
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.datasets import fetch_atlas_aal
import matplotlib.pyplot as plt
import os
import ants
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.ndimage import label, find_objects
import templateflow.api as tflow
from nilearn.glm import cluster_level_inference, threshold_stats_img
from nilearn.mass_univariate import permuted_ols
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_multi_epi_mask
from matplotlib import colors
from nilearn.datasets import fetch_atlas_harvard_oxford
from scipy.ndimage import distance_transform_edt
from collections import Counter

# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
DEMO_DIR     = ANALY_DIR / 'demographics'
TEMP_DIR     = ANALY_DIR / 'templates'
MASK_DIR     = TEMP_DIR / 'mask'

# === Parameters ===
MODEL          = 'glm'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_words'
MASK           = 'none' # none, ventral, WVFA, MTG, STG, IFG, STG-MTG, AG-SMG
GRADE          = '1' # 1, 2, 4 or all
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
CORRECTION     = 'fpr' # fdr, fpr, bonferoni 
P_CORRECTION   = 0.001 # 0.001, 0.05
CLUSTER_SIZE   = 50
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

# %% ===================== Compute thresholded map

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


# model design
# one sample t-test (one-sided)
design_matrix = pd.DataFrame({
    "intercept": [1] * len(beta_paths),
})



if GRADE == 'all':
    n1 = sum(s.startswith("1") for s in subject_names)
    n2 = sum(s.startswith("2") for s in subject_names)
    n4 = sum(s.startswith("4") for s in subject_names)

    # two-sample t-test (two-sided)
    group_indicator = [0]*(n1+n2) + [1]*n4
    design_matrix_twosample = pd.DataFrame({
        "intercept": [1] * len(beta_paths),
        "group": group_indicator,
    })

    # pairwise two-sample t-test (two-sided)
    beta_path_1vs2 = beta_paths[:(n1+n2)]
    group_indicator = [0]*n1 + [1]*n2
    design_matrix_1vs2 = pd.DataFrame({
        "intercept": [1] * len(beta_path_1vs2),
        "group": group_indicator,
    })

    beta_path_1 = beta_paths[:n1]
    beta_path_4 = beta_paths[-n4:]
    beta_path_1vs4 = beta_path_1 + beta_path_4
    group_indicator = [0]*n1 + [1]*n4
    design_matrix_1vs4 = pd.DataFrame({
        "intercept": [1] * len(beta_path_1vs4),
        "group": group_indicator,
    })

    beta_path_2vs4 = beta_paths[-(n4+n2):]
    group_indicator = [0]*n2 + [1]*n4
    design_matrix_2vs4 = pd.DataFrame({
        "intercept": [1] * len(beta_path_2vs4),
        "group": group_indicator,
    })

# specify the model
second_level_model = SecondLevelModel(
    smoothing_fwhm=None, 
    n_jobs=2
    )
second_level_model_acc = SecondLevelModel(
    smoothing_fwhm=None, 
    n_jobs=2
    )
if GRADE == 'all':
    second_level_model_twosample = SecondLevelModel(
        smoothing_fwhm=None, 
        n_jobs=2
        )
    second_level_model_1vs2 = SecondLevelModel(
        smoothing_fwhm=None, 
        n_jobs=2
        )
    second_level_model_1vs4 = SecondLevelModel(
        smoothing_fwhm=None, 
        n_jobs=2
        )
    second_level_model_2vs4 = SecondLevelModel(
        smoothing_fwhm=None, 
        n_jobs=2
        )


# % =====================
# Parametric: One-sample t-test
second_level_model = second_level_model.fit(
    beta_paths,
    design_matrix=design_matrix,
)

if GRADE == 'all':
    second_level_model_twosample = second_level_model_twosample.fit(
        beta_paths,
        design_matrix=design_matrix_twosample,
    )
    second_level_model_1vs2 = second_level_model_1vs2.fit(
        beta_path_1vs2,
        design_matrix=design_matrix_1vs2,
    )
    second_level_model_1vs4 = second_level_model_1vs4.fit(
        beta_path_1vs4,
        design_matrix=design_matrix_1vs4,
    )
    second_level_model_2vs4 = second_level_model_2vs4.fit(
        beta_path_2vs4,
        design_matrix=design_matrix_2vs4,
    )


# one-sample test  
z_map = second_level_model.compute_contrast(
    second_level_contrast="intercept",
    output_type="z_score",
)


if MASK == 'none':
    brain_mask_fn  = MASK_DIR / 'brain_pediatric_MNI.nii.gz'
    brain_mask_img = nib.load(brain_mask_fn)
    # Combine masks (union)
    brain_mask_data = (
            (brain_mask_img.get_fdata() > 0) 
        ).astype(np.uint8)
    
    brain_mask_img = nib.Nifti1Image(
            brain_mask_data,
            brain_mask_img.affine
        )
    z_map = image.math_img(
            "stat * mask",
            stat=z_map,
            mask=brain_mask_img
        )

else:
    left_mask_fn  = MASK_DIR / f'left_{MASK}_pediatric_MNI.nii.gz'
    right_mask_fn = MASK_DIR / f'right_{MASK}_pediatric_MNI.nii.gz'

    left_mask_img = nib.load(left_mask_fn)
    right_mask_img = nib.load(right_mask_fn)

    # Combine masks (union)
    ventral_mask_data = (
            (left_mask_img.get_fdata() > 0) |
            (right_mask_img.get_fdata() > 0)
        ).astype(np.uint8)
        
    ventral_mask_img = nib.Nifti1Image(
            ventral_mask_data,
            left_mask_img.affine
        )
        
    z_map = image.math_img(
            "stat * mask",
            stat=z_map,
            mask=ventral_mask_img
        )



# save the z-map before applying threshold
path        = OUT_DIR / MODEL / SPACE / CONTRASTS / MASK
path.mkdir(parents=True, exist_ok=True)
file_name   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_{MASK}_z-map.nii.gz"
output_path =  path / file_name
z_map.to_filename(output_path)

""" z_map_acc = image.math_img(
        "stat * mask",
        stat=z_map_acc,
        mask=ventral_mask_img
    ) """
    
thresholded_map, threshold = threshold_stats_img(
    z_map,
    two_sided=True,
    alpha=P_CORRECTION,
    cluster_threshold=CLUSTER_SIZE,
    height_control=CORRECTION,
)

""" thresholded_map_acc, threshold_acc = threshold_stats_img(
    z_map_acc,
    two_sided=True,
    alpha=P_CORRECTION,
    # cluster_threshold=CLUSTER_SIZE,
    cluster_threshold=0,
    height_control=CORRECTION,
) """


if GRADE == 'all':
    # pairwise two-sample ttest (two sided)
    z_map_1vs2 = second_level_model_1vs2.compute_contrast(
        second_level_contrast="group",
        output_type="z_score",
    )
    thresholded_map_1vs2, threshold_1vs2 = threshold_stats_img(
        z_map_1vs2,
        two_sided=True,
        alpha=P_CORRECTION,
        cluster_threshold=CLUSTER_SIZE,
        height_control=CORRECTION,
    )

    # pairwise two-sample ttest (two sided)
    z_map_1vs4 = second_level_model_1vs4.compute_contrast(
        second_level_contrast="group",
        output_type="z_score",
    )
    thresholded_map_1vs4, threshold_1vs4 = threshold_stats_img(
        z_map_1vs4,
        two_sided=True,
        alpha=P_CORRECTION,
        cluster_threshold=CLUSTER_SIZE,
        height_control=CORRECTION,
    )

    # pairwise two-sample ttest (two sided)
    z_map_2vs4 = second_level_model_2vs4.compute_contrast(
        second_level_contrast="group",
        output_type="z_score",
    )
    thresholded_map_2vs4, threshold_2vs4 = threshold_stats_img(
        z_map_2vs4,
        two_sided=True,
        alpha=P_CORRECTION,
        cluster_threshold=CLUSTER_SIZE,
        height_control=CORRECTION,
    )

    # two-sample ttest (two sided)
    z_map_twosample = second_level_model_twosample.compute_contrast(
        second_level_contrast="group",
        output_type="z_score",
    )
    thresholded_map_twosample, threshold_twosample = threshold_stats_img(
        z_map_twosample,
        two_sided=True,
        alpha=P_CORRECTION,
        cluster_threshold=CLUSTER_SIZE,
        height_control=CORRECTION,
    )



# save the z-map
path = OUT_DIR / MODEL / SPACE / CONTRASTS / MASK
path.mkdir(parents=True, exist_ok=True)

if GRADE == 'all':
    # pairwise two-sample 
    file_name   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map_1vs2.nii.gz"
    output_path = path / file_name
    thresholded_map_1vs2.to_filename(output_path)

    # pairwise two-sample 
    file_name   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map_1vs4.nii.gz"
    output_path = path / file_name
    thresholded_map_1vs4.to_filename(output_path)

    # pairwise two-sample 
    file_name   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map_2vs4.nii.gz"
    output_path = path / file_name
    thresholded_map_2vs4.to_filename(output_path)

    # two-sample 
    file_name   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map_twosample.nii.gz"
    output_path = path / file_name
    thresholded_map_twosample.to_filename(output_path)

# brain-behavior 
""" file_name   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map_acc.nii.gz"
output_path = path / file_name
thresholded_map_acc.to_filename(output_path) """

# one-sample test 
file_name   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
output_path = path / file_name
thresholded_map.to_filename(output_path)


# %% =====================  Visualize

p_val = P_CORRECTION
p001_unc = norm.isf(p_val)

# Z-coordinates you want to visualize
if 'ventral' in MASK:
    z_coords = np.arange(-16, -7, 1)  # from -16 to -6 inclusive
elif MASK in ['STG', 'AG-SMG']:
    x_coords = np.arange(-68, -56, 2) # from -5 to 5
elif 'none' in MASK:
    z_coords = np.arange(-16, -7, 1)  # from -16 to -6 inclusive


# prepare the title
if '-' in CONTRASTS:
    contrast = CONTRASTS.replace('-', ' > ')
else:
    contrast = CONTRASTS

# create output dir
dir = f"{FIG_DIR}/{MODEL}/{MASK}"
os.makedirs(dir, exist_ok=True)


# Parametric: uncorrected p-value 
# compute threshold for z-value

# configure the figure
if MASK in ['STG', 'AG-SMG']:
    plotting_config = {
        "display_mode": "x",
        "cut_coords": x_coords,
        "draw_cross": False,
        "vmax": 5,
        "vmin": 0,
        "cmap": "hot",
    }
else:
    plotting_config = {
        "display_mode": "z",
        "cut_coords": z_coords,
        "draw_cross": False,
        "vmax": 5,
        "vmin": 0,
        "cmap": "hot",
    }

# plot the uncorrrected p-value figure 
"""     display_unc = plotting.plot_stat_map(
    z_map,
    title=f"Grade-{GRADE} {contrast} (unc p<{P_CORRECTION})",
    bg_img=TEMPLATE,
    threshold=p001_unc,
    **plotting_config,
)
display_unc.savefig(f"{dir}/Grade_{GRADE}_z-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}.png", dpi=300)
"""

# plot the p-value with cluster size figure
thresholded_map = image.math_img(
    "img * (img > 0)",
    img=thresholded_map
)
if  CORRECTION == 'fdr':
    title=f"Grade-{GRADE} {contrast} (cluster-thr > {CLUSTER_SIZE}, pFDR < {P_CORRECTION})"
elif CORRECTION == 'fpr':
    title=f"Grade-{GRADE} {contrast} (cluster-thr > {CLUSTER_SIZE}, p-unc < {P_CORRECTION})"

display_FPR = plotting.plot_stat_map(
    thresholded_map,
    title=title,
    bg_img=TEMPLATE,
    threshold=threshold,
    **plotting_config,
)
# Add contours for the clusters
display_FPR.add_contours(
    thresholded_map, 
    levels=[threshold],       # level(s) to contour
    colors='black',             # color of the outline
    linewidths=1
)
# add pitative VWFA as reference
vwfa_lh_path = TEMP_DIR / 'mask' / 'left_VWFA_pediatric_MNI.nii.gz'
vwfa_rh_path = TEMP_DIR / 'mask' / 'right_VWFA_pediatric_MNI.nii.gz'
vwfa_lh = image.load_img(vwfa_lh_path)
vwfa_rh = image.load_img(vwfa_rh_path)
vwfa_lh_data = vwfa_lh.get_fdata()
vwfa_rh_data = vwfa_rh.get_fdata()
n_lh = np.count_nonzero(vwfa_lh_data)
n_rh = np.count_nonzero(vwfa_rh_data) 

green_cmap = colors.ListedColormap(['springgreen'])
blue_cmap = colors.ListedColormap(['dodgerblue'])
# Add LH VWFA contour
display_FPR.add_contours(
    vwfa_lh,
    levels=[0.5],
    colors='springgreen',
    linewidths=2
)

# Add RH VWFA contour
display_FPR.add_contours(
    vwfa_rh,
    levels=[0.5],
    colors='dodgerblue',
    linewidths=2
)

# count the overlapping significant voxels
overlap_lh = image.math_img(
    "(img1 > 0) & (img2 > 0)",
    img1=thresholded_map,
    img2=vwfa_lh
)

overlap_rh = image.math_img(
    "(img1 > 0) & (img2 > 0)",
    img1=thresholded_map,
    img2=vwfa_rh
)
n_vox_lh = int(np.sum(overlap_lh.get_fdata()))
n_vox_rh = int(np.sum(overlap_rh.get_fdata()))

print(f"-------{CONTRASTS}-------")
print(f"VWFA LH overlapping significant voxels: {n_vox_lh}")
print(f"VWFA LH overlapping volume %: {(n_vox_lh/n_lh)*100:.2f}%")
print(f"VWFA RH overlapping significant voxels: {n_vox_rh}")
print(f"VWFA RH overlapping volume %: {(n_vox_rh/n_rh)*100:.2f}%")

# save the figure
display_FPR.savefig(f"{dir}/Grade_{GRADE}_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_cluster-thr<{CLUSTER_SIZE}_p-val<{P_CORRECTION}_{CORRECTION}_{MASK}.png", dpi=300)


if GRADE == 'all':
    # plot the p-value with cluster size figure 
    display_1vs2 = plotting.plot_stat_map(
        thresholded_map_1vs2,
        title=f"Grade-1 vs. 2 {contrast} (cluster-thr > {CLUSTER_SIZE}, p < {P_CORRECTION})",
        bg_img=TEMPLATE,
        threshold=threshold_1vs2,
        **plotting_config,
    )
    # Add contours for the clusters
    display_1vs2.add_contours(
        thresholded_map_1vs2, 
        levels=[threshold_1vs2],       # level(s) to contour
        colors='black',             # color of the outline
        linewidths=1
    )
    display_1vs2.savefig(f"{dir}/Grade_1vs2_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_cluster-thr<{CLUSTER_SIZE}_p-val<{P_CORRECTION}_{CORRECTION}.png", dpi=300)

    # plot the p-value with cluster size figure 
    display_1vs4 = plotting.plot_stat_map(
        thresholded_map_1vs4,
        title=f"Grade-1 vs. 4 {contrast} (cluster-thr > {CLUSTER_SIZE}, p < {P_CORRECTION})",
        bg_img=TEMPLATE,
        threshold=threshold_1vs4,
        **plotting_config,
    )
    # Add contours for the clusters
    display_1vs4.add_contours(
        thresholded_map_1vs4, 
        levels=[threshold_1vs4],    # level(s) to contour
        colors='black',             # color of the outline
        linewidths=1
    )
    display_1vs4.savefig(f"{dir}/Grade_1vs4_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_cluster-thr<{CLUSTER_SIZE}_p-val<{P_CORRECTION}_{CORRECTION}.png", dpi=300)

    # plot the p-value with cluster size figure 
    display_2vs4 = plotting.plot_stat_map(
        thresholded_map_2vs4,
        title=f"Grade-2 vs. 4 {contrast} (cluster-thr > {CLUSTER_SIZE}, p < {P_CORRECTION})",
        bg_img=TEMPLATE,
        threshold=threshold_2vs4,
        **plotting_config,
    )
    # Add contours for the clusters
    display_2vs4.add_contours(
        thresholded_map_2vs4, 
        levels=[threshold_2vs4],       # level(s) to contour
        colors='black',             # color of the outline
        linewidths=1
    )
    display_2vs4.savefig(f"{dir}/Grade_2vs4_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_cluster-thr<{CLUSTER_SIZE}_p-val<{P_CORRECTION}_{CORRECTION}.png", dpi=300)




# glass brain:: one-sample test
display = plotting.plot_glass_brain(
    thresholded_map,
    title=title,
    threshold=threshold,
    colorbar=True,
    display_mode='lyrz',  # show left, sagittal, coronal, axial views
    plot_abs=False,     # keep sign info if you have positive/negative values
    black_bg=False,
    vmin=0,
)
display.add_contours(
    thresholded_map, 
    levels=[threshold],       # level(s) to contour
    colors='black',             # color of the outline
    linewidths=0.5
)
display.savefig(f"{dir}/Grade_{GRADE}_z-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}_glass.png", dpi=300)
plotting.show()


# glass brain:: brain-behavior
""" display = plotting.plot_glass_brain(
    thresholded_map_acc,
    title=f"Grade-{GRADE} {contrast} (cluster-thr > 0, pFDR < {P_CORRECTION})",
    threshold=threshold_acc,
    colorbar=True,
    display_mode='lyrz',  # show left, sagittal, coronal, axial views
    plot_abs=False,     # keep sign info if you have positive/negative values
    black_bg=False,
    vmin=0,
)
display.add_contours(
    thresholded_map_acc, 
    levels=[threshold_acc],       # level(s) to contour
    colors='black',             # color of the outline
    linewidths=0.5
)
display.savefig(f"{dir}/Grade_{GRADE}_z-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}_glass_acc.png", dpi=300)
plotting.show() """


if GRADE == 'all':
    # glass brain:: two-sample (two-sided)
    display = plotting.plot_glass_brain(
        thresholded_map_twosample,
        title=f"Grade-1&2 vs. -4 {contrast} (cluster-thr > {CLUSTER_SIZE}, unc p < {P_CORRECTION})",
        threshold=threshold_twosample,
        colorbar=True,
        display_mode='lyrz',  # show left, sagittal, coronal, axial views
        plot_abs=False,     # keep sign info if you have positive/negative values
        black_bg=False,
        # vmin=0,
    )
    display.add_contours(
        thresholded_map_twosample, 
        levels=[threshold_twosample],       # level(s) to contour
        colors='black',             # color of the outline
        linewidths=0.5
    )
    display.savefig(f"{dir}/Grade_{GRADE}_z-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}_glass_twosample.png", dpi=300)
    plotting.show()

    # glass brain:: pairwise two-sample (two-sided)
    display = plotting.plot_glass_brain(
        thresholded_map_1vs2,
        title=f"Grade 1 vs. 2 {contrast} (cluster-thr > {CLUSTER_SIZE}, unc p < {P_CORRECTION})",
        threshold=threshold_1vs2,
        colorbar=True,
        display_mode='lyrz',  # show left, sagittal, coronal, axial views
        plot_abs=False,     # keep sign info if you have positive/negative values
        black_bg=False,
        # vmin=0,
    )
    display.add_contours(
        thresholded_map_1vs2, 
        levels=[threshold_1vs2],       # level(s) to contour
        colors='black',             # color of the outline
        linewidths=0.5
    )
    display.savefig(f"{dir}/Grade_{GRADE}_z-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}_glass_1vs2.png", dpi=300)
    plotting.show()

    # glass brain:: pairwise two-sample (two-sided)
    display = plotting.plot_glass_brain(
        thresholded_map_1vs4,
        title=f"Grade 1 vs. 4 {contrast} (cluster-thr > {CLUSTER_SIZE}, unc p < {P_CORRECTION})",
        threshold=threshold_1vs4,
        colorbar=True,
        display_mode='lyrz',  # show left, sagittal, coronal, axial views
        plot_abs=False,     # keep sign info if you have positive/negative values
        black_bg=False,
        # vmin=0,
    )
    display.add_contours(
        thresholded_map_1vs4, 
        levels=[threshold_1vs4],       # level(s) to contour
        colors='black',             # color of the outline
        linewidths=0.5
    )
    display.savefig(f"{dir}/Grade_{GRADE}_z-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}_glass_1vs4.png", dpi=300)
    plotting.show()

    # glass brain:: pairwise two-sample (two-sided)
    display = plotting.plot_glass_brain(
        thresholded_map_2vs4,
        title=f"Grade 2 vs. 4 {contrast} (cluster-thr > {CLUSTER_SIZE}, unc p < {P_CORRECTION})",
        threshold=threshold_2vs4,
        colorbar=True,
        display_mode='lyrz',  # show left, sagittal, coronal, axial views
        plot_abs=False,     # keep sign info if you have positive/negative values
        black_bg=False,
        # vmin=0,
    )
    display.add_contours(
        thresholded_map_2vs4, 
        levels=[threshold_2vs4],       # level(s) to contour
        colors='black',             # color of the outline
        linewidths=0.5
    )
    display.savefig(f"{dir}/Grade_{GRADE}_z-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}_glass_2vs4.png", dpi=300)
    plotting.show()




# %%
# =========================================================
# ===  Brain regions specification with assiged cluster ===
# =========================================================

# === specify the target grade === 
tar_grade = GRADE


# Fetch Harvard-Oxford cortical + subcortical atlas
harvard_oxford_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
# Load the labels from the txt file
atlas_labels         = harvard_oxford_atlas.lut
# Path to atlas NIfTI file
atlas_img_nib        = harvard_oxford_atlas['maps']  # or .maps
# Save to temp file
atlas_path           = '/tmp/harvard_oxford_atlas.nii.gz'
atlas_img_nib.to_filename(atlas_path)


atlas_img = ants.image_read(atlas_path)
ped_img   = ants.image_read(str(TEMPLATE))
ped_nifti = nib.load(TEMPLATE)

warp_path   = str(MASK_DIR / 'MNI6_to_Pediatric_1Warp.nii.gz')
affine_path = str(MASK_DIR / 'MNI6_to_Pediatric_0GenericAffine.mat')

forward_transforms = [
    warp_path, 
    affine_path
]

# Transform
pediatric_atlas_ants = ants.apply_transforms(
    fixed=ped_img,
    moving=atlas_img,
    transformlist=forward_transforms,
    interpolator='genericLabel' 
)
pediatric_atlas_data = pediatric_atlas_ants.numpy().astype(np.int16)
# Convert pediatric atlas data back to NIfTI
pediatric_atlas      = nib.Nifti1Image(pediatric_atlas_data, ped_nifti.affine)


# -------------------------------
# % === Visualization ===
# -------------------------------
# Define single coordinates for each view
z_coord = -12  # One axial slice
y_coord = -57  # One coronal slice (middle of your range)
x_coord = -45  # One sagittal slice (middle of your range)
# configure the figure
plotting_config = {
    "display_mode": "ortho",
    "cut_coords": (x_coord, y_coord, z_coord),
    "draw_cross": False,
    "bg_img": TEMPLATE,
}

# plot the mask figure 
display_ventral = plotting.plot_roi(
    pediatric_atlas,
    colorbar=False,
    **plotting_config,
)
plotting.show()



# % === load the z-map === 
path      = OUT_DIR / MODEL / SPACE / CONTRASTS / MASK
file_name = f"GRADE-{tar_grade}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
filepath  = path / file_name
data_img  = nib.load(filepath)
data      = data_img.get_fdata()

# Find the minimum positive value
min_positive = data[data > 0].min()

# find the MNI coordinates of a cluster
cluster_mask = data != 0

# Label connected clusters
labeled_array, num_clusters = label(cluster_mask)
print(f"Found {num_clusters} clusters")


# for every voxels, givs indices of nearest labeled voxel
labeled_mask = pediatric_atlas_data > 0
distances, nearest_indices = distance_transform_edt(
    ~labeled_mask,            # background voxels
    return_indices=True
)

# Loop over clusters
for i in range(1, num_clusters + 1):

    cluster_data = data * (labeled_array == i)
    max_val = np.max(cluster_data)
    min_val = np.min(cluster_data)
    if max_val > 0:
        # Find the voxel of the maximum value in this cluster
        voxel_index = np.unravel_index(
            np.argmax(cluster_data), 
            cluster_data.shape
        )   
    elif min_val < 0:
        # Find the voxel of the minimum value in this cluster
        voxel_index = np.unravel_index(
            np.argmin(cluster_data), 
            cluster_data.shape
        )

    # get the label value
    atlas_index = pediatric_atlas_data[voxel_index]
    if atlas_index      != 0:
        outside = False
        atlas_label_row = atlas_labels.loc[atlas_labels['index'] == atlas_index]
    else:
        outside = True
        nearest_voxel = tuple(nearest_indices[:, voxel_index[0], voxel_index[1], voxel_index[2]])
        nearest_label_value = pediatric_atlas_data[nearest_voxel]
        atlas_label_row = atlas_labels.loc[atlas_labels['index'] == nearest_label_value]

    # Voxel -> MNI world coordinates
    voxel_homogeneous = np.array(voxel_index + (1,))
    mni_mm = ped_nifti.affine.dot(voxel_homogeneous)[:3]

    # cluster size
    cluster_size = np.sum(labeled_array == i)

    print("============================================")
    print(f"Cluster-{i}")
    print("nearest label") if outside else None
    print(atlas_label_row)
    print(f"MNI (mm) {mni_mm}")
    print(f"Peak value: {cluster_data[voxel_index]:.2f}, "
        f"Size: {cluster_size} voxels")
    print("============================================")


# %%
