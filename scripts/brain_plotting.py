# %%
# ===  Load modules ===
from nilearn.plotting import plot_glass_brain
from nilearn import plotting, image
import matplotlib.pyplot as plt
from pathlib import Path
from bids import BIDSLayout
import glob
from nilearn import image
import os

# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'

# === Parameters ===
SUBJECT        = 'sub-101'
MODEL          = 'glm'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_words+audios_words-images_pseudo+audios_pseudo'
GRADE          = '4' # 1, 2, 4
FWHM_SMOOTHING = 6.0 # 6.0, 9.0, 12.0

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
subjects = sorted(DERIV_DIR.glob(f"sub-{GRADE}*"))
subject_names = [p.name.replace('sub-', '') for p in subjects]
t_maps=[]
for subject in subjects:
    f_name = Path(f"{subject}/glm/{SPACE}/FWHM_{int(FWHM_SMOOTHING)}/{CONTRASTS}_tstat.nii.gz")
    t_map  = image.load_img(f_name) 
    t_maps.append(t_map)


nrows, ncols = 6, 4

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 16))
axes = axes.flatten()  # flatten grid for simple 1D indexing
for cidx, tmap in enumerate(t_maps):
    plotting.plot_glass_brain(
        tmap,
        colorbar=True,
        threshold=3,
        title=subject_names[cidx],
        axes=axes[cidx],
        plot_abs=False,
        display_mode="z",
        vmin=-10,
        vmax=10,
    )
for ax in axes[len(t_maps):]:
    ax.axis('off')

fig.suptitle("Subject-level t-maps for contrast: Words > Pseudowords", fontsize=16, y=0.93)
dir = f"{FIG_DIR}/{MODEL}"
os.makedirs(dir, exist_ok=True)
fig.savefig(f"{dir}/Grade_{GRADE}_t-maps_{CONTRASTS}.png", dpi=300, bbox_inches='tight')
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
