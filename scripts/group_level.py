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

# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
DEMO_DIR     = ANALY_DIR / 'demographics'
TEMP_DIR     = ANALY_DIR / 'templates'

# === Parameters ===
MODEL          = 'glm'
METHOD         = 'parametric' # 'parametric' 'non-parametric'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_pseudo'
MASK           = 'ventral' # none, ventral, fusiform, WVFA
GRADE          = '1' # 1, 2, 4 or all
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
CORRECTION     = 'fdr' # fdr, fpr, bonferoni 
P_CORRECTION   = 0.05 # 0.001, 0.05
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

# get the sex info
path   = DEMO_DIR / 'scanning_protocol.xlsx'
sex_df = pd.read_excel(path)
sex_df = sex_df.iloc[1:].reset_index(drop=True)
sex_df = sex_df[['participant_id', 'age_years', 'age_months']]
sex_df['participant_id'] = sex_df['participant_id'].astype(str)
sex_df.loc[sex_df['participant_id'] == "435 (228)", 'participant_id'] = "435"
age_lists = []
for sub in subject_names:
    age = sex_df.loc[sex_df['participant_id'] == sub, 'age_years'].values[0]
    age_lists.append(age)



# model design
# one sample t-test (one-sided)
design_matrix = pd.DataFrame({
    "intercept": [1] * len(beta_paths),
    # "age": age_lists,
})
# brain-behavior (one-sided)
design_matrix_acc = pd.DataFrame({
    "intercept": [1] * len(beta_paths),
    "accuracy": acc_lists,
    "age": age_lists
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
if METHOD == 'parametric':
    second_level_model = second_level_model.fit(
        beta_paths,
        design_matrix=design_matrix,
    )
    second_level_model_acc = second_level_model_acc.fit(
        beta_paths,
        design_matrix=design_matrix_acc,
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

    if MASK == 'ventral':
        left_mask_fn  = TEMP_DIR / 'mask' / 'left_ventral_mask.nii.gz'
        right_mask_fn = TEMP_DIR / 'mask' / 'right_ventral_mask.nii.gz'
        left_mask_img = nib.load(left_mask_fn)
        right_mask_img = nib.load(right_mask_fn)
        # Combine masks (union)
        ventral_mask_data = (
                (left_mask_img.get_fdata() == 1) |
                (right_mask_img.get_fdata() == 1)
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
        
    thresholded_map, threshold = threshold_stats_img(
        z_map,
        two_sided=True,
        alpha=P_CORRECTION,
        cluster_threshold=CLUSTER_SIZE,
        height_control=CORRECTION,
    )

    # brain-behavior correlation
    z_map_acc = second_level_model_acc.compute_contrast(
        second_level_contrast="accuracy",
        output_type="z_score",
    )
    thresholded_map_acc, threshold_acc = threshold_stats_img(
        z_map_acc,
        two_sided=True,
        alpha=P_CORRECTION,
        cluster_threshold=CLUSTER_SIZE,
        height_control=CORRECTION,
    )


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
    path = OUT_DIR / MODEL / SPACE / CONTRASTS
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
    file_name   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map_acc.nii.gz"
    output_path = path / file_name
    thresholded_map_acc.to_filename(output_path)

    # one-sample test 
    file_name   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
    output_path = path / file_name
    thresholded_map.to_filename(output_path)

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
    display_FPR = plotting.plot_stat_map(
        thresholded_map,
        title=f"Grade-{GRADE} {contrast} (cluster-thr > {CLUSTER_SIZE}, p < {P_CORRECTION})",
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

# glass brain:: one-sample test
display = plotting.plot_glass_brain(
    thresholded_map,
    title=f"Grade-{GRADE} {contrast} (cluster-thr > {CLUSTER_SIZE}, unc p < {P_CORRECTION})",
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
display = plotting.plot_glass_brain(
    thresholded_map_acc,
    title=f"Grade-{GRADE} {contrast} (cluster-thr > {CLUSTER_SIZE}, unc p < {P_CORRECTION})",
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
plotting.show()


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
