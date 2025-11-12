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
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
import templateflow.api as tflow

# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'

# === Parameters ===
MODEL          = 'glm'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'audios_words-audios_pseudo'
GRADE          = '1' # 1, 2, 4
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
P_CORRECTION   = 0.001
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

# % =====================
# list the beta nii.files
subjects      = sorted(DERIV_DIR.glob(f"sub-{GRADE}*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]
subject_names = [p.name.replace('sub-', '') for p in subjects]

beta_lists = []
for sub in subjects:
    path = sub / 'glm' / SPACE / f"FWHM_{int(FWHM_SMOOTHING)}" 
    beta_path = next(path.glob(f"{CONTRASTS}_beta.nii.gz"), None)
    beta_lists.append(beta_path)
beta_paths = [str(p) for p in beta_lists]
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

second_level_model = second_level_model.fit(
    beta_paths,
    design_matrix=design_matrix,
)
z_map = second_level_model.compute_contrast(
    second_level_contrast="intercept",
    output_type="z_score",
)

# % =====================  Visualize
# compute threshold for z-value
p_val = P_CORRECTION
p001_unc = norm.isf(p_val)

# create output dir
dir = f"{FIG_DIR}/{MODEL}"
os.makedirs(dir, exist_ok=True)

# prepare the title
if '-' in CONTRASTS:
    contrast = CONTRASTS.replace('-', ' > ')
else:
    contrast = CONTRASTS

# Z-coordinates you want to visualize
if 'images' in CONTRASTS:
    z_coords = np.arange(-15, -4, 1)  # from -15 to -5 inclusive
elif 'audios' in CONTRASTS:
    z_coords = np.arange(-5, 10, 2) # from -5 to 5

# configure the figure
plotting_config = {
    "display_mode": "z",
    "cut_coords": z_coords,
    "draw_cross": False,
    "vmax": 5,
    "vmin": -5,
    "cmap": "cold_hot",
}

# plot the T1w figure 
display = plotting.plot_stat_map(
    z_map,
    title=f"Grade-{GRADE} {contrast} (unc p<{P_CORRECTION})",
    bg_img=TEMPLATE,
    threshold=p001_unc,
    **plotting_config,
)

display.savefig(f"{dir}/Grade_{GRADE}_z-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}.png", dpi=300)
plotting.show()

# %%
# plot the glass brain
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
