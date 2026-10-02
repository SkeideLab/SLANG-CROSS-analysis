# %% [markdown]
# ## Figure 3. Regions of interest exhibiting significant (pFDR < .05) differences across school grades.
#
# **Pipeline Overview**
# 1. === STEP 1 ===: Install packages
# 2. === STEP 2 ===: Set parameters
# 3. === STEP 3 ===: Output Figure 3


# %%
# 1. === STEP 1 ===: Install packages
# -----------------------------------------------
import sys
from pathlib import Path
# Specify path
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
SCRIP_DIR    = ANALY_DIR / 'scripts'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
DEMO_DIR     = ANALY_DIR / 'demographics'
TEMP_DIR     = ANALY_DIR / 'templates'
MASK_DIR     = TEMP_DIR  / 'mask'
# install paclages in a python file
sys.path.append(str(SCRIP_DIR))
import my_packages
from my_packages import *



# %%
# 2. === STEP 2 ===: Set parameters
# -----------------------------------------------
MODEL          = 'glm'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
FWHM_SMOOTHING = 5.0 
HEMI           = 'left' 
EXC_SUBJECTS   = [
                '108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', 
                '201', '205', '206', '208', '220', '225', '226', '227', 
                '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434',
                ]
HO_ATLAS_MNI6  = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm') # Harvard-Oxford MNI6Asym
atlas_labels   = HO_ATLAS_MNI6.lut
ROIs           = {
                    "IFG": [
                        "Inferior Frontal Gyrus, pars triangularis",
                        "Inferior Frontal Gyrus, pars opercularis"
                    ],
                    "IPG": [
                        "Supramarginal Gyrus, anterior division",
                        "Supramarginal Gyrus, posterior division",
                        "Angular Gyrus"
                    ],
                    "STG": [
                        "Superior Temporal Gyrus, anterior division",
                        "Superior Temporal Gyrus, posterior division"
                    ],
                    "MTG": [
                        "Middle Temporal Gyrus, anterior division",
                        "Middle Temporal Gyrus, posterior division",
                        "Middle Temporal Gyrus, temporooccipital part"
                    ],
                    "Fusiform": [
                        "Temporal Fusiform Cortex, posterior division",
                        "Temporal Occipital Fusiform Cortex",
                    ]
                }
roi_color_map  = {
                    "Inferior Frontal Gyrus, pars triangularis": "darkgoldenrod",
                    "Inferior Frontal Gyrus, pars opercularis": "goldenrod",
                    "Supramarginal Gyrus, anterior division": "royalblue",
                    "Supramarginal Gyrus, posterior division": "dodgerblue",
                    "Angular Gyrus": "navy",
                    "Superior Temporal Gyrus, anterior division": "mediumvioletred",
                    "Superior Temporal Gyrus, posterior division": "darkmagenta",
                    "Middle Temporal Gyrus, anterior division": "deeppink",
                    "Middle Temporal Gyrus, posterior division": "hotpink",
                    "Middle Temporal Gyrus, temporooccipital part": "palevioletred",
                    "Temporal Fusiform Cortex, posterior division": "green",
                    "Temporal Occipital Fusiform Cortex": "limegreen",
                    }
roi_labels     = {
                    "Inferior Frontal Gyrus, pars triangularis": "trIFG",
                    "Inferior Frontal Gyrus, pars opercularis": "opIFG",
                    "Supramarginal Gyrus, anterior division": "aSMG",
                    "Supramarginal Gyrus, posterior division": "pSMG",
                    "Angular Gyrus": "AG",
                    "Superior Temporal Gyrus, anterior division": "aSTG",
                    "Superior Temporal Gyrus, posterior division": "pSTG",
                    "Middle Temporal Gyrus, anterior division": "aMTG",
                    "Middle Temporal Gyrus, posterior division": "pMTG",
                    "Middle Temporal Gyrus, temporooccipital part": "toMTG",
                    "Temporal Fusiform Cortex, posterior division": "pTFC",
                    "Temporal Occipital Fusiform Cortex": "TOFC",
                    }



# %%
# 3. === STEP 3 ===: Output Figure 3
# -----------------------------------------------
cond_list = [
    "word_spoken",
    "word_written",
    "word_multi",
]
cond_titles = {
    "word_spoken": "Spoken",
    "word_written": "Written",
    "word_multi": "Spoken & Written",
}
grade_pairs = [
    (1, 2),
    (1, 4),
    (2, 4),
]

# Explicit color scale
vmin = -np.log10(0.1)
vmax = -np.log10(0.001)

log_threshold = -np.log10(0.05)

# Harvard-Oxford atlas
atlas_img = HO_ATLAS_MNI6.maps
atlas_labels = HO_ATLAS_MNI6.labels

label_to_index = {
    name: i for i, name in enumerate(atlas_labels)
}

atlas_data = np.asarray(atlas_img.dataobj)

# Create figure
fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(10, 8)
)
# Loop over conditions
for row, cond in enumerate(cond_list):

    # Read statistical results for this condition
    path = OUT_DIR / "multimodal" / f"{HEMI}_pairwisecomp_{cond}.csv"
    df_ttest = pd.read_csv(path)

    # Create ROI -> p-value dictionary for each grade pair
    roi_value_maps = {}

    for A, B in grade_pairs:

        df_pair = df_ttest[
            (df_ttest["A"] == A) &
            (df_ttest["B"] == B)
        ]

        roi_value_maps[f"{A}_vs_{B}"] = dict(
            zip(
                df_pair["roi"],
                df_pair["p-corr"]
            )
        )

    # Create p-value maps
    pval_imgs = {}

    for comparison, roi_pvals in roi_value_maps.items():

        pval_data = np.zeros(
            atlas_img.shape,
            dtype=float
        )

        for roi, p_val in roi_pvals.items():

            if roi not in label_to_index:
                continue

            roi_idx = label_to_index[roi]

            # ROI voxels
            roi_mask = atlas_data == roi_idx

            # Assign p-value
            pval_data[roi_mask] = p_val

        pval_imgs[comparison] = nib.Nifti1Image(
            pval_data,
            atlas_img.affine,
            atlas_img.header
        )

    # Convert p-values to -log10(p)
    neglog_p_imgs = {}

    for comparison, img in pval_imgs.items():

        data = img.get_fdata().copy()

        # Avoid log10(0)
        data[data == 0] = np.nan

        neglog_p = -np.log10(data)

        neglog_p_imgs[comparison] = nib.Nifti1Image(
            neglog_p,
            img.affine,
            img.header
        )

    # Plot grade comparisons
    for col, (A, B) in enumerate(grade_pairs):

        comparison = f"{A}_vs_{B}"
        img = neglog_p_imgs[comparison]

        display = plotting.plot_glass_brain(
            img,
            display_mode="l",
            colorbar=False,
            threshold=log_threshold,
            vmin=vmin,
            vmax=vmax,
            cmap="hot",
            plot_abs=False,
            title=None,
            axes=axes[row, col],
        )
        # Show title only in the first row, centered
        if row == 0:
            axes[row, col].set_title(
                f"Grade {A} < {B}",
                fontsize=20,
                fontweight="bold",
                loc="left",
                pad=10,
            )
        # Black contour at p = .05
        display.add_contours(
            img,
            levels=[log_threshold],
            colors="black",
            linewidths=1.5,
        )
    # Row title
    axes[row, 0].text(
        -0.2,
        0.5,
        cond_titles[cond],
        transform=axes[row, 0].transAxes,
        fontsize=20,
        fontweight="bold",
        rotation=0,
        va="center",
        ha="right",
    )

# Shared colorbar
cmap = plt.get_cmap("hot")
norm = mpl.colors.Normalize(
    vmin=vmin,
    vmax=vmax
)
sm = mpl.cm.ScalarMappable(
    norm=norm,
    cmap=cmap
)
sm.set_array([])

# Colorbar centered underneath figure
cbar_ax = fig.add_axes([
    0.36,    # left
    0.045,   # bottom
    0.28,    # width
    0.025    # height
])
cbar = fig.colorbar(
    sm,
    cax=cbar_ax,
    orientation="horizontal"
)

# Colorbar label
cbar.set_label(
    r"$-\log_{10}$(FDR-corrected $p$-value)",
    fontsize=16,
    labelpad=8
)
cbar.ax.tick_params(
    labelsize=12,
    length=10,
    width=1.5
)

# P-value tick labels
cbar.set_ticks([
    -np.log10(0.1),
    -np.log10(0.05),
    -np.log10(0.01),
    -np.log10(0.001),
])
cbar.set_ticklabels([
    "0.10",
    "0.05",
    "0.01",
    "0.001",
])

# Layout
plt.subplots_adjust(
    left=0.12,
    right=0.95,
    top=0.93,
    bottom=0.15,
    wspace=0.05,
    hspace=0.25,
)

# Save figure as PDF
plt.savefig(
    FIG_DIR / "multimodal" / "glass_brain_fdr_pvalues.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()
