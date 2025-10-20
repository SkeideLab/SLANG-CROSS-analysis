# %%
# ===  Load modules ===
from nilearn.plotting import plot_glass_brain
from nilearn import plotting, image

# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'

# === Parameters ===
SUBJECT      = 'sub-101'
MODEL        = 'glm'
SPACE        = 'MNI152NLin2009cAsym'

# %% Plot
# === Brain ===
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