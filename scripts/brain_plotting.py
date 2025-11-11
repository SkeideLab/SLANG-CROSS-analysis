# %%
# ===  Load modules ===
from nilearn.plotting import plot_glass_brain, plot_stat_map, show
from nilearn import plotting, image
import matplotlib.pyplot as plt
from pathlib import Path
from bids import BIDSLayout
import glob
from nilearn import image
import os
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
CONTRASTS      = 'images_words-images_pseudo'
GRADE          = '4' # 1, 2, 4
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
EXC_SUBJECTS   = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434']
# Retrieve the T1-weighted template for cohort 4 (7.5-13.5yrs) at 2mm resolution
TEMPLATE       =  tflow.get(
    "MNIPediatricAsym",
    cohort="4",       # cohort 4 = age 7.5-13.5
    resolution=2,     # 2mm isotropic
    suffix="T1w"      # T1-weighted image
)

# List of the contrasts
"""             'images_words - images_pseudo' 
                'audios_words - audios_pseudo' 
                'images_words' 
                'audios_words'
                'images_pseudo'
                'audios_pseudo'
                '(images_words + audios_words) - (images_pseudo + audios_pseudo)' """

# %% Plot
# === Brain ===
subjects      = sorted(DERIV_DIR.glob(f"sub-{GRADE}*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]
subject_names = [p.name.replace('sub-', '') for p in subjects]
t_maps = []
for subject in subjects:
    f_name = Path(f"{subject}/glm/{SPACE}/FWHM_{int(FWHM_SMOOTHING)}/{CONTRASTS}_tstat.nii.gz")
    t_map  = image.load_img(f_name) 
    t_maps.append(t_map)

# %% On the T1w Template 
nrows, ncols = 6, 4
fig, axes    = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 16))
axes         = axes.flatten()  # flatten grid for simple 1D indexing
threshold    = 2

plotting_config = {
    "display_mode": "z",
    "cut_coords": [-14],
    "draw_cross": False,
    "vmax": 4,
    "vmin": -4,
    "cmap": "cold_hot",
}
for cidx, tmap in enumerate(t_maps):
    plot_stat_map(
        tmap,
        title=subject_names[cidx],
        axes=axes[cidx],
        bg_img=TEMPLATE,
        threshold=threshold,
        **plotting_config,
    )
for ax in axes[len(t_maps):]:
    ax.axis('off')
fig.suptitle("Subject-level t-maps for contrast: Visual Words > Pseudowords", fontsize=16, y=0.93)
dir = f"{FIG_DIR}/{MODEL}"
os.makedirs(dir, exist_ok=True)
fig.savefig(f"{dir}/Grade_{GRADE}_t-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}.png", dpi=300, bbox_inches='tight')
plt.show()

# %% On the Grass brain view
nrows, ncols = 5, 4
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 16))
axes = axes.flatten()  # flatten grid for simple 1D indexing
for cidx, tmap in enumerate(t_maps):
    plotting.plot_glass_brain(
        tmap,
        colorbar=True,
        threshold=2,
        title=subject_names[cidx],
        axes=axes[cidx],
        plot_abs=False,
        display_mode="z",
        vmin=-8,
        vmax=8,
    )
for ax in axes[len(t_maps):]:
    ax.axis('off')

fig.suptitle("Subject-level t-maps for contrast: Visual Words > Pseudowords", fontsize=16, y=0.93)
dir = f"{FIG_DIR}/{MODEL}"
os.makedirs(dir, exist_ok=True)
fig.savefig(f"{dir}/Grade_{GRADE}_t-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_glassbrain.png", dpi=300, bbox_inches='tight')
plt.show()

""" nii_path = DERIV_DIR / SUBJECT / MODEL / SPACE / 'images_words-images_pseudo_beta.nii.gz'
image    = nib.load(nii_path)

plotting_config = {
    "colorbar": True,
    "cmap": "inferno",
}

display = plotting.plot_glass_brain(
    image,
    title=SUBJECT,
    threshold=THRESHOLD,
    display_mode="lzr",
    **plotting_config,
)
display.savefig(f"{FIG_DIR}/glass_brain.png", dpi=300) """
# %%
