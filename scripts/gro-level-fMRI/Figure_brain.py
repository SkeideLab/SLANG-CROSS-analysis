# %% [markdown]
# ## fMRI Group-level Brain Figure creation (2nd-level)
#
# **Pipeline Overview**
# 1. === STEP 1 ===: Install packages
# 2. === STEP 2 ===: Set parameters
# 3. === STEP 3 ===: Visualize all regions



# %%
# 1. === STEP 1 ===: Install packages
# -----------------------------------------------

# install necessary packages
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

# ============================================================
# Settings
# ============================================================

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

# ------------------------------------------------------------
# Explicit color scale
# ------------------------------------------------------------

vmin = -np.log10(0.1)
vmax = -np.log10(0.001)

log_threshold = -np.log10(0.05)

# ============================================================
# Harvard-Oxford atlas
# ============================================================

atlas_img = HO_ATLAS_MNI6.maps
atlas_labels = HO_ATLAS_MNI6.labels

label_to_index = {
    name: i for i, name in enumerate(atlas_labels)
}

atlas_data = np.asarray(atlas_img.dataobj)

# ============================================================
# Create figure
# ============================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(10, 8)
)

# ============================================================
# Loop over conditions
# ============================================================

for row, cond in enumerate(cond_list):

    # --------------------------------------------------------
    # Read statistical results for this condition
    # --------------------------------------------------------

    path = OUT_DIR / "multimodal" / f"{HEMI}_pairwisecomp_{cond}.csv"

    df_ttest = pd.read_csv(path)

    # --------------------------------------------------------
    # Create ROI -> p-value dictionary for each grade pair
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Create p-value maps
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Convert p-values to -log10(p)
    # --------------------------------------------------------

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

    # ========================================================
    # Plot grade comparisons
    # ========================================================

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

        # ----------------------------------------------------
        # Black contour at p = .05
        # ----------------------------------------------------

        display.add_contours(
            img,
            levels=[log_threshold],
            colors="black",
            linewidths=1.5,
        )

    # --------------------------------------------------------
    # Row label
    # --------------------------------------------------------

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

# ============================================================
# Shared colorbar
# ============================================================

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

# ------------------------------------------------------------
# Colorbar centered underneath figure
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Colorbar label
# ------------------------------------------------------------

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
# ------------------------------------------------------------
# P-value tick labels
# ------------------------------------------------------------

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
# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.12,
    right=0.95,
    top=0.93,
    bottom=0.15,
    wspace=0.05,
    hspace=0.25,
)
# ============================================================
# Save figure as PDF
# ============================================================

plt.savefig(
    FIG_DIR / "multimodal" / "glass_brain_fdr_pvalues.pdf",
    format="pdf",
    bbox_inches="tight",
)

plt.show()





# %%
###########################
# Linear regression results
############################


# ============================================================
# Settings
# ============================================================

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

# ------------------------------------------------------------
# Explicit color scale
# ------------------------------------------------------------

vmin = -np.log10(0.1)
vmax = -np.log10(0.001)

log_threshold = -np.log10(0.05)


# ============================================================
# Harvard-Oxford atlas
# ============================================================

atlas_img = HO_ATLAS_MNI6.maps
atlas_labels = HO_ATLAS_MNI6.labels

label_to_index = {
    name: i for i, name in enumerate(atlas_labels)
}

atlas_data = np.asarray(atlas_img.dataobj)


# ============================================================
# Read ONE CSV containing all conditions
# ============================================================

path = OUT_DIR / "multimodal" / f'{HEMI}_FWHM_{int(FWHM_SMOOTHING)}_linear-model_grade.csv'

df_results = pd.read_csv(path)


# ============================================================
# Create figure
# ============================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(3, 6)
)

axes = np.atleast_1d(axes)


# ============================================================
# Loop over conditions
# ============================================================

for row, cond in enumerate(cond_list):

    # --------------------------------------------------------
    # Select condition
    # --------------------------------------------------------

    df_cond = df_results[
        df_results["condition"] == cond
    ].copy()


    # --------------------------------------------------------
    # Create ROI -> FDR-corrected p-value dictionary
    # --------------------------------------------------------

    roi_pvals = dict(
        zip(
            df_cond["roi"],
            df_cond["p_fdr"]
        )
    )


    # --------------------------------------------------------
    # Create p-value map
    # --------------------------------------------------------

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

        # Assign FDR-corrected p-value
        pval_data[roi_mask] = p_val


    pval_img = nib.Nifti1Image(
        pval_data,
        atlas_img.affine,
        atlas_img.header
    )


    # --------------------------------------------------------
    # Convert p-values to -log10(p)
    # --------------------------------------------------------

    data = pval_img.get_fdata().copy()

    # Avoid log10(0)
    data[data == 0] = np.nan

    neglog_p = -np.log10(data)

    neglog_p_img = nib.Nifti1Image(
        neglog_p,
        pval_img.affine,
        pval_img.header
    )


    # ========================================================
    # Plot glass brain
    # ========================================================

    display = plotting.plot_glass_brain(
        neglog_p_img,
        display_mode="l",
        colorbar=False,
        threshold=log_threshold,
        vmin=vmin,
        vmax=vmax,
        cmap="hot",
        plot_abs=False,
        title=None,
        axes=axes[row],
    )


    # --------------------------------------------------------
    # Black contour at FDR p = .05
    # --------------------------------------------------------

    display.add_contours(
        neglog_p_img,
        levels=[log_threshold],
        colors="black",
        linewidths=1.5,
    )


    # --------------------------------------------------------
    # Row label
    # --------------------------------------------------------

    axes[row].text(
        0,
        0.5,
        cond_titles[cond],
        transform=axes[row].transAxes,
        fontsize=13,
        fontweight="bold",
        rotation=0,
        va="center",
        ha="right",
    )


# ============================================================
# Shared colorbar
# ============================================================

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


# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

cbar_ax = fig.add_axes([
    0.30,
    0.035,
    0.40,
    0.025
])

cbar = fig.colorbar(
    sm,
    cax=cbar_ax,
    orientation="horizontal"
)


# ------------------------------------------------------------
# Colorbar label
# ------------------------------------------------------------

cbar.set_label(
    r"$-\log_{10}$(FDR-corrected $p$-value)",
    fontsize=13,
    labelpad=8
)

cbar.ax.tick_params(
    labelsize=9,
    length=12,
    width=1.5
)


# ------------------------------------------------------------
# P-value tick labels
# ------------------------------------------------------------

cbar.set_ticks([
    -np.log10(0.1),
    -np.log10(0.01),
    -np.log10(0.001),
])

cbar.set_ticklabels([
    "0.10",
    "0.01",
    "0.001",
])


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.18,
    right=0.95,
    top=0.95,
    bottom=0.12,
    hspace=0.20,
)


# ============================================================
# Save figure
# ============================================================

path = FIG_DIR / "multimodal" / "glass_brain_fdr_pvalues_model.pdf"

plt.savefig(
    path,
    format="pdf",
    bbox_inches="tight",
)

print(f"\nSuccessful: {path} is saved")

plt.show()










# %%
# 3. === STEP 3 ===: visualize TOFC with glass brain
# --------------------------------------------------

from nilearn import plotting
from nilearn.image import new_img_like, load_img
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Get atlas information
# ------------------------------------------------------------

atlas_img    = HO_ATLAS_MNI6.maps
atlas_labels = HO_ATLAS_MNI6.labels

label_to_index = {
    name: i for i, name in enumerate(atlas_labels)
}

# ------------------------------------------------------------
# Get ROI indices
# ------------------------------------------------------------

roi_indices = []
roi_names   = []

for group, regions in ROIs.items():
    for r in regions:
        if r in label_to_index:
            roi_indices.append(label_to_index[r])
            roi_names.append(r)

roi_indices = np.array(roi_indices)


# ------------------------------------------------------------
# Create volumetric ROI image
#
# Each ROI gets a unique integer:
#   1, 2, 3, ...
# Background = 0
# ------------------------------------------------------------

atlas_data = atlas_img.get_fdata()

roi_map = np.zeros_like(atlas_data, dtype=np.int16)

for new_idx, old_idx in enumerate(roi_indices, start=1):
    roi_map[atlas_data == old_idx] = new_idx

roi_img = new_img_like(
    atlas_img,
    roi_map
)


# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------

NAME = "Superior Temporal Gyrus, posterior division" # "Temporal Occipital Fusiform Cortex"

rgba_colors = []

for roi_name in roi_names:

    color = roi_color_map[roi_name]

    # TOFC = fully opaque
    # Other ROIs = translucent
    alpha = 1.0 if roi_name == NAME else 0.25

    rgba = (*plt.matplotlib.colors.to_rgb(color), alpha)
    rgba_colors.append(rgba)


# Background = transparent
cmap = ListedColormap(
    [(1, 1, 1, 0)] + rgba_colors
)


# ------------------------------------------------------------
# Plot glass brain
# ------------------------------------------------------------

fig = plt.figure(figsize=(5, 5))

display = plotting.plot_glass_brain(
    roi_img,
    display_mode="l",
    colorbar=False,
    cmap=cmap,
    threshold=0.5,
    plot_abs=False,
    black_bg=False,
    figure=fig
)


# ------------------------------------------------------------
# TOFC-only mask
# ------------------------------------------------------------

tofc_atlas_index = label_to_index[NAME]

tofc_mask = (
    atlas_data == tofc_atlas_index
).astype(np.int16)

tofc_img = new_img_like(
    atlas_img,
    tofc_mask
)


# ------------------------------------------------------------
# Black contour ONLY around TOFC
# ------------------------------------------------------------

display.add_contours(
    tofc_img,
    colors="black",
    linewidths=1,
)

# ------------------------------------------------------------
# Legend: TOFC and putative VWFA only
# ------------------------------------------------------------

from matplotlib.lines import Line2D

legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=roi_color_map[NAME],
        markeredgecolor="black",
        markersize=13,
        label=NAME
    )
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.02),
    frameon=False,
    fontsize=18,
    handletextpad=0.5,
    ncol=1
)
# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

plt.tight_layout()


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

roi_path = FIG_DIR / 'multimodal'
roi_path.mkdir(exist_ok=True, parents=True)

path = roi_path / f"{HEMI}_{NAME}_glassbrain.pdf"

plt.savefig(
    path,
    format='pdf',
    dpi=300,
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.3
)

print(f"\nSuccessful: {path} is saved")

plt.show()


















































##################
# Previous code not used anymore
##################
# %%
# ------------------------------------------------------------
# Project volumetric atlas to surface
# ------------------------------------------------------------

texture = surface.vol_to_surf(
    atlas_img,
    surf_mesh     = mesh,
    inner_mesh    = white_mesh,
    interpolation = 'nearest',
    n_samples     = 1
)

roi_map = texture.copy()


# ------------------------------------------------------------
# Keep only selected ROIs
# ------------------------------------------------------------

roi_mask = np.isin(roi_map, roi_indices)

roi_map_masked = roi_map.copy()
roi_map_masked[~roi_mask] = np.nan

# ------------------------------------------------------------
# Create surface map containing mean_r
# ------------------------------------------------------------

stat_map = np.full_like(roi_map, np.nan, dtype=float)

for roi_idx, value in zip(roi_indices, roi_values):
    stat_map[roi_map == roi_idx] = value

# ------------------------------------------------------------
# Fill only NaN vertices that are directly adjacent
# to valid vertices
# ------------------------------------------------------------

mesh_obj = surface.load_surf_mesh(mesh)

# Faces of the triangular mesh
faces = mesh_obj.faces

# Build vertex adjacency
from scipy.sparse import lil_matrix

n_vertices = len(stat_map)

adjacency = lil_matrix(
    (n_vertices, n_vertices),
    dtype=bool
)

for face in faces:
    a, b, c = face

    adjacency[a, b] = True
    adjacency[b, a] = True

    adjacency[a, c] = True
    adjacency[c, a] = True

    adjacency[b, c] = True
    adjacency[c, b] = True

adjacency = adjacency.tocsr()

# ------------------------------------------------------------
# Identify NaN vertices directly neighboring valid vertices
# ------------------------------------------------------------

valid = np.isfinite(stat_map)
nan_idx = np.where(~valid)[0]

fill_idx = []

for idx in nan_idx:

    neighbors = adjacency[idx].indices

    # Fill only if at least one neighboring vertex is valid
    if np.any(valid[neighbors]):
        fill_idx.append(idx)

fill_idx = np.array(fill_idx)

print("Total NaNs:", len(nan_idx))
print("NaNs adjacent to valid vertices:", len(fill_idx))

# ------------------------------------------------------------
# Assign the mean of neighboring valid ROI values
# ------------------------------------------------------------

for idx in fill_idx:

    neighbors = adjacency[idx].indices
    valid_neighbors = neighbors[valid[neighbors]]

    stat_map[idx] = np.mean(stat_map[valid_neighbors])

# ------------------------------------------------------------
# Color scale
# ------------------------------------------------------------

vmax = np.nanmax(np.abs(stat_map))
vmin = np.nanmin(np.abs(stat_map))

# ------------------------------------------------------------
# Views
# ------------------------------------------------------------

if HEMI == 'left':
    views = ["lateral", "ventral"]

elif HEMI == 'right':
    views = ["lateral", "ventral"]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig = plt.figure(figsize=(8, 5))

axes = [
    fig.add_subplot(1, 2, 1, projection='3d'),
    fig.add_subplot(1, 2, 2, projection='3d'),
]

for ax, view in zip(axes, views):

    plotting.plot_surf_stat_map(
        surf_mesh = mesh,
        stat_map  = stat_map,
        hemi      = HEMI,
        view      = view,
        cmap      = "hot",
        colorbar  = False,
        bg_map    = None,
        vmin      = vmin,
        vmax      = vmax,
        axes      = ax
    )
# ------------------------------------------------------------
# Shared color scale at the bottom
# ------------------------------------------------------------

sm = plt.cm.ScalarMappable(
    cmap="hot_r",
    norm=plt.Normalize(vmin=vmin, vmax=vmax)
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    orientation="horizontal",
    fraction=0.05,
    pad=0.03,
    shrink=0.6
)

cbar.set_label(r"$p$-value (FDR-corrected)")

# %
# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

roi_path = FIG_DIR / "multimodal"
roi_path.mkdir(exist_ok=True, parents=True)

path = roi_path / f"{HEMI}_{cond}_mean_similarity.pdf"

plt.savefig(
    path,
    format      = "pdf",
    dpi         = 300,
    transparent = True,
    bbox_inches = "tight",
    pad_inches  = 0.3
)

print(f"\nSuccessful: {path} is saved")

plt.show()








# %%
# ------------------------------------------------------------

# fsaverage surface
fsaverage = datasets.fetch_surf_fsaverage()

if HEMI == 'left':
    mesh       = fsaverage.pial_left
    white_mesh = fsaverage.white_left
    infl_mesh  = fsaverage.infl_left

elif HEMI == 'right':
    mesh       = fsaverage.pial_right
    white_mesh = fsaverage.white_right
    infl_mesh  = fsaverage.infl_right


# %%
# --------------------------------------------------------
# Create volumetric ROI p-value map
# --------------------------------------------------------

atlas_data = atlas_img.get_fdata()

stat_data = np.zeros_like(
    atlas_data,
    dtype=float
)

for roi_idx, p_value in zip(
    roi_indices,
    roi_values
):

    stat_data[
        atlas_data == roi_idx
    ] = p_value


stat_img = nib.Nifti1Image(
    stat_data,
    affine=atlas_img.affine,
    header=atlas_img.header
)

# --------------------------------------------------------
# Glass brain
# --------------------------------------------------------

fig = plt.figure(
    figsize=(10, 4)
)

display = plotting.plot_glass_brain(
    stat_img,
    display_mode="lyrz",
    colorbar=True,
    cmap="hot_r",
    threshold=0.05,
    vmin=0,
    vmax=0.05,
    plot_abs=False,
    black_bg=False,
    figure=fig,
)

display.title(
    f"Grade {pair.replace('_vs_', ' vs ')}",
    y=1.05
)

# --------------------------------------------------------
# Save
# --------------------------------------------------------

glass_path = (
    roi_path /
    f"{HEMI}_{pair}_{cond}_glassbrain_pvalue.pdf"
)

plt.savefig(
    glass_path,
    format="pdf",
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.3
)

print(
    f"Successful: {glass_path} is saved"
)

plt.show()
plt.close(fig)





















# %%
# --- keep only ROIs significant after FDR correction ---
df_ttest = df_ttest[df_ttest["significant_fdr"] == True].copy()


# ------------------------------------------------------------
# Create ROI → mean_r mapping
# ------------------------------------------------------------
roi_value_map = dict(
    zip(df_ttest["roi"], df_ttest["t"])
)
# ------------------------------------------------------------
# Collect ROIs that exist in both atlas and CSV
# ------------------------------------------------------------

roi_indices = []
roi_values  = []

for roi, mean_r in roi_value_map.items():

    if roi in label_to_index:
        roi_indices.append(label_to_index[roi])
        roi_values.append(mean_r)

roi_indices = np.array(roi_indices)
roi_values  = np.array(roi_values)


# ------------------------------------------------------------
# Project volumetric atlas to surface
# ------------------------------------------------------------

texture = surface.vol_to_surf(
    atlas_img,
    surf_mesh     = mesh,
    inner_mesh    = white_mesh,
    interpolation = 'nearest',
    n_samples     = 1
)

roi_map = texture.copy()


# ------------------------------------------------------------
# Keep only selected ROIs
# ------------------------------------------------------------

roi_mask = np.isin(roi_map, roi_indices)

roi_map_masked = roi_map.copy()
roi_map_masked[~roi_mask] = np.nan

# ------------------------------------------------------------
# Create surface map containing mean_r
# ------------------------------------------------------------

stat_map = np.full_like(roi_map, np.nan, dtype=float)

for roi_idx, value in zip(roi_indices, roi_values):
    stat_map[roi_map == roi_idx] = value

# ------------------------------------------------------------
# Color scale
# ------------------------------------------------------------

vmax = np.nanmax(np.abs(stat_map))
vmin = np.nanmin(np.abs(stat_map))

# ------------------------------------------------------------
# Views
# ------------------------------------------------------------

if HEMI == 'left':
    views = ["lateral", "ventral", "medial"]

elif HEMI == 'right':
    views = ["medial", "ventral", "lateral"]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig = plt.figure(figsize=(8, 5))

axes = [
    fig.add_subplot(1, 3, 1, projection='3d'),
    fig.add_subplot(1, 3, 2, projection='3d'),
    fig.add_subplot(1, 3, 3, projection='3d')
]

for ax, view in zip(axes, views):

    plotting.plot_surf_stat_map(
        surf_mesh = mesh,
        stat_map  = stat_map,
        hemi      = HEMI,
        view      = view,
        cmap      = "hot",
        colorbar  = False,
        bg_map    = None,
        vmin      = vmin-3,
        vmax      = vmax+3,
        axes      = ax
    )
# ------------------------------------------------------------
# Shared color scale at the bottom
# ------------------------------------------------------------

sm = plt.cm.ScalarMappable(
    cmap="hot",
    norm=plt.Normalize(vmin=vmin-3, vmax=vmax+3)
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    orientation="horizontal",
    fraction=0.05,
    pad=0.03,
    shrink=0.6
)

cbar.set_label(r"$t$-value")

# %
# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

roi_path = FIG_DIR / "multimodal"
roi_path.mkdir(exist_ok=True, parents=True)

path = roi_path / f"{HEMI}_{cond}_t-val_similarity.pdf"

plt.savefig(
    path,
    format      = "pdf",
    dpi         = 300,
    transparent = True,
    bbox_inches = "tight",
    pad_inches  = 0.3
)

print(f"\nSuccessful: {path} is saved")

plt.show()





# %%
# 3. === STEP 3 ===: Read statistical results by school grade
# ------------------------------------------------------------

path = (
    OUT_DIR
    / "multimodal"
    / f"{HEMI}_FWHM_{int(FWHM_SMOOTHING)}_one-sample-tttest-by-grade.csv"
)

df_ttest = pd.read_csv(path)

# ------------------------------------------------------------
# Keep only the desired condition
# ------------------------------------------------------------

df_ttest = df_ttest[
    df_ttest["condition"] == cond
].copy()

# ------------------------------------------------------------
# Keep only ROIs significant after correction
# ------------------------------------------------------------

df_ttest = df_ttest[
    df_ttest["significant_fdr"] == True
].copy()


# %
# ------------------------------------------------------------
# Project volumetric atlas to surface
# ------------------------------------------------------------

texture = surface.vol_to_surf(
    atlas_img,
    surf_mesh=mesh,
    inner_mesh=white_mesh,
    interpolation="nearest",
    n_samples=1
)

roi_map = texture.copy()


# %
# ------------------------------------------------------------
# Views
# ------------------------------------------------------------

if HEMI == "left":
    views = ["lateral", "ventral", "medial"]

elif HEMI == "right":
    views = ["medial", "ventral", "lateral"]


# %
# ------------------------------------------------------------
# Grades
# ------------------------------------------------------------

grades = [1, 2, 4]


# ------------------------------------------------------------
# Create figure
# 3 rows = grades
# 3 columns = views
# ------------------------------------------------------------

fig = plt.figure(figsize=(8, 8))

axes = []

for row, grade in enumerate(grades):

    # --------------------------------------------------------
    # Select current grade
    # --------------------------------------------------------

    df_grade = df_ttest[
        df_ttest["grade"] == grade
    ].copy()

    # --------------------------------------------------------
    # Create ROI → t-value mapping
    # --------------------------------------------------------

    roi_value_map = dict(
        zip(
            df_grade["roi"],
            df_grade["t"]
        )
    )

    # --------------------------------------------------------
    # Collect ROIs that exist in both atlas and CSV
    # --------------------------------------------------------

    roi_indices = []
    roi_values = []

    for roi, t_value in roi_value_map.items():

        if roi in label_to_index:

            roi_indices.append(
                label_to_index[roi]
            )

            roi_values.append(
                t_value
            )

    roi_indices = np.array(roi_indices)
    roi_values = np.array(roi_values)

    # --------------------------------------------------------
    # Create surface statistical map
    # --------------------------------------------------------

    stat_map = np.full(
        roi_map.shape,
        np.nan,
        dtype=float
    )

    for roi_idx, value in zip(
        roi_indices,
        roi_values
    ):

        stat_map[
            roi_map == roi_idx
        ] = value

    # --------------------------------------------------------
    # Plot three views for this grade
    # --------------------------------------------------------

    row_axes = []

    for col, view in enumerate(views):

        ax = fig.add_subplot(
            len(grades),
            len(views),
            row * len(views) + col + 1,
            projection="3d"
        )

        plotting.plot_surf_stat_map(
            surf_mesh=mesh,
            stat_map=stat_map,
            hemi=HEMI,
            view=view,
            cmap="hot",
            colorbar=False,
            bg_map=None,
            vmin=vmin - 3,
            vmax=vmax + 3,
            axes=ax
        )

        row_axes.append(ax)

    axes.append(row_axes)

    # --------------------------------------------------------
    # Add grade label on the left
    # --------------------------------------------------------
    row_axes[0].text2D(
        -0.15,
        0.5,
        f"Grade {grade}",
        transform=row_axes[0].transAxes,
        rotation=0,
        va="center",
        ha="center",
        fontsize=12
    )


# %
# ------------------------------------------------------------
# Shared color scale at the bottom
# ------------------------------------------------------------

sm = plt.cm.ScalarMappable(
    cmap="hot",
    norm=plt.Normalize(
        vmin=vmin - 3,
        vmax=vmax + 3
    )
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=np.array(axes).flatten().tolist(),
    orientation="horizontal",
    fraction=0.04,
    pad=0.03,
    shrink=0.6
)

cbar.set_label(r"$t$-value")

# %
# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

roi_path = FIG_DIR / "multimodal"
roi_path.mkdir(
    exist_ok=True,
    parents=True
)

path = (
    roi_path
    / f"{HEMI}_{cond}_t-val_similarity_gradewise.pdf"
)

plt.savefig(
    path,
    format="pdf",
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.3
)

print(f"\nSuccessful: {path} is saved")

plt.show()




# %%
# 3. === STEP 3 ===: Read results from linear model
# ------------------------------------------------------------

path = (
    OUT_DIR
    / "multimodal"
    / f"{HEMI}_FWHM_{int(FWHM_SMOOTHING)}_linear-model.csv"
)

df_linear = pd.read_csv(path)

# ------------------------------------------------------------
# Keep only the desired condition
# ------------------------------------------------------------

df_linear  = df_linear [
    df_linear["condition"] == cond
].copy()

# ------------------------------------------------------------
# Keep only ROIs significant after correction
# ------------------------------------------------------------

df_linear = df_linear[
    df_linear["significant_fdr"] == True
].copy()
min = df_linear["beta"].min()
max = df_linear["beta"].max()

# %
# ------------------------------------------------------------
# Project volumetric atlas to surface
# ------------------------------------------------------------

texture = surface.vol_to_surf(
    atlas_img,
    surf_mesh=mesh,
    inner_mesh=white_mesh,
    interpolation="nearest",
    n_samples=1
)

roi_map = texture.copy()

# %
# ------------------------------------------------------------
# Views
# ------------------------------------------------------------

if HEMI == "left":
    views = ["lateral", "ventral", "medial"]

elif HEMI == "right":
    views = ["medial", "ventral", "lateral"]


# ------------------------------------------------------------
# Create figure
# 1 row = all data
# 3 columns = views
# ------------------------------------------------------------


fig = plt.figure(figsize=(8, 5))

axes = []


# ------------------------------------------------------------
# Create ROI → t-value mapping
# ------------------------------------------------------------

roi_value_map = dict(
    zip(
        df_linear["roi"],
        df_linear["beta"]
    )
)

# ------------------------------------------------------------
# Collect ROIs that exist in both atlas and CSV
# ------------------------------------------------------------

roi_indices = []
roi_values = []

for roi, t_value in roi_value_map.items():

    if roi in label_to_index:

        roi_indices.append(
            label_to_index[roi]
        )

        roi_values.append(
            t_value
        )

roi_indices = np.array(roi_indices)
roi_values = np.array(roi_values)

# ------------------------------------------------------------
# Create surface statistical map
# ------------------------------------------------------------

stat_map = np.full(
    roi_map.shape,
    np.nan,
    dtype=float
)

for roi_idx, value in zip(
    roi_indices,
    roi_values
):

    stat_map[
        roi_map == roi_idx
    ] = value

# ------------------------------------------------------------
# Plot three views
# ------------------------------------------------------------

for col, view in enumerate(views):

    ax = fig.add_subplot(
        1,
        len(views),
        col + 1,
        projection="3d"
    )

    plotting.plot_surf_stat_map(
        surf_mesh=mesh,
        stat_map=stat_map,
        hemi=HEMI,
        view=view,
        cmap="hot",
        colorbar=False,
        bg_map=None,
        vmin=0,
        vmax=max + 0.03,
        axes=ax
    )

    axes.append(ax)

# %
# ------------------------------------------------------------
# Shared color scale at the bottom
# ------------------------------------------------------------

sm = plt.cm.ScalarMappable(
    cmap="hot",
    norm=plt.Normalize(
        vmin=0,
        vmax=max + 0.03
    )
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=np.array(axes).flatten().tolist(),
    orientation="horizontal",
    fraction=0.04,
    pad=0.03,
    shrink=0.6
)

cbar.set_label(r"$β$-value")

# %
# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

roi_path = FIG_DIR / "multimodal"
roi_path.mkdir(
    exist_ok=True,
    parents=True
)

path = (
    roi_path
    / f"{HEMI}_{cond}_beta-val_similarity.pdf"
)

plt.savefig(
    path,
    format="pdf",
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.3
)

print(f"\nSuccessful: {path} is saved")

plt.show()
# %%
