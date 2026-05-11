# %% [markdown]
# ## fMRI Group-level Brain Figure creation (2nd-level)
#
# **Pipeline Overview**
# 1. === STEP 1 ===: Install packages
# 2. === STEP 2 ===: Set parameters
# 3. === STEP 3 ===: Visualize all regions
# 4. === STEP 4 ===: Visualize significant regions (word)
# 5. === STEP 5 ===: Visualize significant regions (pseudoword)



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
CONTRASTS      = [
                'images_words', 
                'audios_words',
                'images_pseudo',
                'audios_pseudo',
                'images_words-images_pseudo',
                'audios_words-audios_pseudo',
                ]
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
HEMI           = 'left' 
EXC_SUBJECTS   = [
                '108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', 
                '201', '205', '206', '208', '220', '225', '226', '227', 
                '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434',
                ]
HO_ATLAS_MNI6  = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm') # Harvard-Oxford MNI6Asym
atlas_labels   = HO_ATLAS_MNI6.lut
roi_sig_word_map  = {
                        "Superior Frontal Gyrus": "goldenrod",
                        "Inferior Frontal Gyrus, pars opercularis": "goldenrod",
                        "Precentral Gyrus": "goldenrod",
                        "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "goldenrod",
                        "Cingulate Gyrus, anterior division": "goldenrod",
                        "Frontal Orbital Cortex": "goldenrod",

                        "Postcentral Gyrus": "dodgerblue",
                        "Superior Parietal Lobule": "dodgerblue",
                        "Supramarginal Gyrus, anterior division": "dodgerblue",
                        "Supramarginal Gyrus, posterior division": "dodgerblue",
                        "Angular Gyrus": "dodgerblue",

                        "Superior Temporal Gyrus, posterior division": "mediumvioletred",
                        "Middle Temporal Gyrus, temporooccipital part": "mediumvioletred",
                        "Inferior Temporal Gyrus, temporooccipital part": "mediumvioletred",

                        "Lateral Occipital Cortex, superior division": "limegreen",
                        "Lateral Occipital Cortex, inferior division": "limegreen",
                        "Temporal Occipital Fusiform Cortex": "limegreen",
                }
roi_sig_pseudo_map  = {
                        "Superior Frontal Gyrus": "goldenrod",
                        "Inferior Frontal Gyrus, pars triangularis": "goldenrod",
                        "Inferior Frontal Gyrus, pars opercularis": "goldenrod",
                        "Precentral Gyrus": "goldenrod",
                        "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "goldenrod",
                        "Frontal Orbital Cortex": "goldenrod",

                        "Postcentral Gyrus": "dodgerblue",
                        "Superior Parietal Lobule": "dodgerblue",
                        "Supramarginal Gyrus, anterior division": "dodgerblue",
                        "Supramarginal Gyrus, posterior division": "dodgerblue",
                        "Angular Gyrus": "dodgerblue",

                        "Middle Temporal Gyrus, temporooccipital part": "mediumvioletred",
                        "Inferior Temporal Gyrus, temporooccipital part": "mediumvioletred",

                        "Lateral Occipital Cortex, superior division": "limegreen",
                        "Lateral Occipital Cortex, inferior division": "limegreen",
                        "Temporal Occipital Fusiform Cortex": "limegreen",
                        "Occipital Pole": "limegreen",
                }
roi_color_map  = {
                        "Frontal Pole": "goldenrod",
                        "Superior Frontal Gyrus": "goldenrod",
                        "Middle Frontal Gyrus": "goldenrod",
                        "Inferior Frontal Gyrus, pars triangularis": "goldenrod",
                        "Inferior Frontal Gyrus, pars opercularis": "goldenrod",
                        "Precentral Gyrus": "goldenrod",
                        "Frontal Medial Cortex": "goldenrod",
                        "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "goldenrod",
                        "Subcallosal Cortex": "goldenrod",
                        "Paracingulate Gyrus": "goldenrod",
                        "Cingulate Gyrus, anterior division": "goldenrod",
                        "Cingulate Gyrus, posterior division": "goldenrod",
                        "Frontal Orbital Cortex": "goldenrod",
                        "Frontal Opercular Cortex": "goldenrod",
                        "Insular Cortex": "goldenrod",

                        "Postcentral Gyrus": "dodgerblue",
                        "Superior Parietal Lobule": "dodgerblue",
                        "Supramarginal Gyrus, anterior division": "dodgerblue",
                        "Supramarginal Gyrus, posterior division": "dodgerblue",
                        "Angular Gyrus": "dodgerblue",
                        "Precuneous Cortex": "dodgerblue",
                        "Central Opercular Cortex": "dodgerblue",
                        "Parietal Opercular Cortex": "dodgerblue",

                        "Temporal Pole": "mediumvioletred",
                        "Superior Temporal Gyrus, anterior division": "mediumvioletred",
                        "Superior Temporal Gyrus, posterior division": "mediumvioletred",
                        "Middle Temporal Gyrus, anterior division": "mediumvioletred",
                        "Middle Temporal Gyrus, posterior division": "mediumvioletred",
                        "Middle Temporal Gyrus, temporooccipital part": "mediumvioletred",
                        "Inferior Temporal Gyrus, anterior division": "mediumvioletred",
                        "Inferior Temporal Gyrus, posterior division": "mediumvioletred",
                        "Inferior Temporal Gyrus, temporooccipital part": "mediumvioletred",
                        "Parahippocampal Gyrus, anterior division": "mediumvioletred",
                        "Parahippocampal Gyrus, posterior division": "mediumvioletred",
                        "Planum Polare": "mediumvioletred",
                        "Heschl's Gyrus (includes H1 and H2)": "mediumvioletred",
                        "Planum Temporale": "mediumvioletred",

                        "Temporal Fusiform Cortex, anterior division": "limegreen",
                        "Temporal Fusiform Cortex, posterior division": "limegreen",
                        "Lateral Occipital Cortex, superior division": "limegreen",
                        "Lateral Occipital Cortex, inferior division": "limegreen",
                        "Intracalcarine Cortex": "limegreen",
                        "Cuneal Cortex": "limegreen",
                        "Lingual Gyrus": "limegreen",
                        "Occipital Fusiform Gyrus": "limegreen",
                        "Temporal Occipital Fusiform Cortex": "limegreen",
                        "Supracalcarine Cortex": "limegreen",
                        "Occipital Pole": "limegreen",
                }



# %%
# 3. === STEP 3 ===: Visualize all regions
# ------------------------------------------------

# fsaverage surface
fsaverage  = datasets.fetch_surf_fsaverage()
if HEMI == 'left':
    sulc       = fsaverage.sulc_left
    mesh       = fsaverage.pial_left
    white_mesh = fsaverage.white_left
    infl_mesh  = fsaverage.infl_left

elif HEMI == 'right':
    sulc       = fsaverage.sulc_right
    mesh       = fsaverage.pial_right
    white_mesh = fsaverage.white_right
    infl_mesh  = fsaverage.infl_right

# Get the list of names with label from the atlas
atlas_img      = HO_ATLAS_MNI6.maps
atlas_labels   = HO_ATLAS_MNI6.labels
label_to_index = {name: i for i, name in enumerate(atlas_labels)}

# Get the label of ROIs
roi_indices = []
roi_colors  = []
for region, color in roi_color_map.items():
    if region in label_to_index:
        roi_indices.append(label_to_index[region])
        roi_colors.append(color)
            
roi_indices = np.array(roi_indices)

# Transfrom volumetric HO-Atlas into urface 
texture_left = surface.vol_to_surf(
            atlas_img,
            surf_mesh     = mesh,
            inner_mesh    = white_mesh,
            interpolation = 'nearest',
            n_samples     = 1
        )
roi_map_left = texture_left.copy()

# assign NaN to non ROI regions
roi_mask                       = np.isin(roi_map_left, roi_indices)
roi_map_left_masked            = roi_map_left.copy()
roi_map_left_masked[~roi_mask] = np.nan

 # Specify view angle of the figure
views = ["lateral", "ventral", "medial"]


# convert original atlas indices to start from 0
index_map         = {idx: i for i, idx in enumerate(roi_indices)}
roi_map_reindexed = np.full_like(roi_map_left_masked, np.nan)
for old_idx, new_idx in index_map.items():
    roi_map_reindexed[roi_map_left_masked == old_idx] = new_idx


# Build colormap
cmap   = ListedColormap(roi_colors)
levels = list(range(len(roi_colors)))

# Create one figure with two sub-panels
fig  = plt.figure(figsize=(7, 5))
axes = [
    fig.add_subplot(1, 3, 1, projection='3d'),
    fig.add_subplot(1, 3, 2, projection='3d'),
    fig.add_subplot(1, 3, 3, projection='3d')

]
for ax, v in zip(axes, views):
    # plot ROIs with contours on surface

    plotting.plot_surf_contours(
        surf_mesh = infl_mesh,
        roi_map   = roi_map_reindexed,
        hemi      = HEMI,
        view      = v,
        levels    = levels,
        colors    = roi_colors,
        axes      = ax
    )

plt.tight_layout()

# save the figure
roi_path = FIG_DIR / 'multimodal'
roi_path.mkdir(exist_ok=True, parents=True)
path     = roi_path / f"{HEMI}_all-regions.pdf"
plt.savefig(
    path,
    format      = 'pdf',
    dpi         = 300,
    transparent = True,
    bbox_inches = 'tight',
    pad_inches  = 0.3
)
print(f"\nSuccessful: {path} is saved ")
plt.show()




# %%
# 4. === STEP 4 ===: Visualize significant regions (word)
# ------------------------------------------------

# fsaverage surface
fsaverage  = datasets.fetch_surf_fsaverage()
if HEMI == 'left':
    sulc       = fsaverage.sulc_left
    mesh       = fsaverage.pial_left
    white_mesh = fsaverage.white_left
    infl_mesh  = fsaverage.infl_left

elif HEMI == 'right':
    sulc       = fsaverage.sulc_right
    mesh       = fsaverage.pial_right
    white_mesh = fsaverage.white_right
    infl_mesh  = fsaverage.infl_right

# Get the list of names with label from the atlas
atlas_img      = HO_ATLAS_MNI6.maps
atlas_labels   = HO_ATLAS_MNI6.labels
label_to_index = {name: i for i, name in enumerate(atlas_labels)}

# Get the label of ROIs
roi_indices = []
roi_colors  = []
for region, color in roi_sig_word_map.items():
    if region in label_to_index:
        roi_indices.append(label_to_index[region])
        roi_colors.append(color)
            
roi_indices = np.array(roi_indices)

# Transfrom volumetric HO-Atlas into urface 
texture_left = surface.vol_to_surf(
            atlas_img,
            surf_mesh     = mesh,
            inner_mesh    = white_mesh,
            interpolation = 'nearest',
            n_samples     = 1
        )
roi_map_left = texture_left.copy()

# assign NaN to non ROI regions
roi_mask                       = np.isin(roi_map_left, roi_indices)
roi_map_left_masked            = roi_map_left.copy()
roi_map_left_masked[~roi_mask] = np.nan

 # Specify view angle of the figure
views = ["lateral", "ventral", "medial"]


# convert original atlas indices to start from 0
index_map         = {idx: i for i, idx in enumerate(roi_indices)}
roi_map_reindexed = np.full_like(roi_map_left_masked, np.nan)
for old_idx, new_idx in index_map.items():
    roi_map_reindexed[roi_map_left_masked == old_idx] = new_idx


# Build colormap
cmap   = ListedColormap(roi_colors)
levels = list(range(len(roi_colors)))

# Create one figure with two sub-panels
fig  = plt.figure(figsize=(7, 5))
axes = [
    fig.add_subplot(1, 3, 1, projection='3d'),
    fig.add_subplot(1, 3, 2, projection='3d'),
    fig.add_subplot(1, 3, 3, projection='3d')

]
for ax, v in zip(axes, views):
    # plot ROIs with contours on surface

    # Fill ROI regions
    plotting.plot_surf_roi(
        surf_mesh   = infl_mesh,
        roi_map     = roi_map_reindexed,
        hemi        = HEMI,
        view        = v,
        cmap        = cmap,
        axes        = ax,
        colorbar    = False,
        darkness    = 1
    )
    #　Draw ONLY edge lines on top
    plotting.plot_surf_contours(
        surf_mesh = infl_mesh,
        roi_map   = roi_map_reindexed,
        hemi      = HEMI,
        view      = v,
        levels    = levels,
        colors    = ["black"] * len(levels), 
        linewidths= 0.8,
        axes      = ax
    )

plt.tight_layout()

# save the figure
roi_path = FIG_DIR / 'multimodal'
roi_path.mkdir(exist_ok=True, parents=True)
path     = roi_path / f"{HEMI}_sig-word-regions.pdf"
plt.savefig(
    path,
    format      = 'pdf',
    dpi         = 300,
    transparent = True,
    bbox_inches = 'tight',
    pad_inches  = 0.3
)
print(f"\nSuccessful: {path} is saved ")
plt.show()




# %%
# 5. === STEP 5 ===: Visualize significant regions (pseudoword)
# ------------------------------------------------

# fsaverage surface
fsaverage  = datasets.fetch_surf_fsaverage()
if HEMI == 'left':
    sulc       = fsaverage.sulc_left
    mesh       = fsaverage.pial_left
    white_mesh = fsaverage.white_left
    infl_mesh  = fsaverage.infl_left

elif HEMI == 'right':
    sulc       = fsaverage.sulc_right
    mesh       = fsaverage.pial_right
    white_mesh = fsaverage.white_right
    infl_mesh  = fsaverage.infl_right

# Get the list of names with label from the atlas
atlas_img      = HO_ATLAS_MNI6.maps
atlas_labels   = HO_ATLAS_MNI6.labels
label_to_index = {name: i for i, name in enumerate(atlas_labels)}

# Get the label of ROIs
roi_indices = []
roi_colors  = []
for region, color in roi_sig_pseudo_map.items():
    if region in label_to_index:
        roi_indices.append(label_to_index[region])
        roi_colors.append(color)
            
roi_indices = np.array(roi_indices)

# Transfrom volumetric HO-Atlas into urface 
texture_left = surface.vol_to_surf(
            atlas_img,
            surf_mesh     = mesh,
            inner_mesh    = white_mesh,
            interpolation = 'nearest',
            n_samples     = 1
        )
roi_map_left = texture_left.copy()

# assign NaN to non ROI regions
roi_mask                       = np.isin(roi_map_left, roi_indices)
roi_map_left_masked            = roi_map_left.copy()
roi_map_left_masked[~roi_mask] = np.nan

 # Specify view angle of the figure
views = ["lateral", "ventral", "medial"]


# convert original atlas indices to start from 0
index_map         = {idx: i for i, idx in enumerate(roi_indices)}
roi_map_reindexed = np.full_like(roi_map_left_masked, np.nan)
for old_idx, new_idx in index_map.items():
    roi_map_reindexed[roi_map_left_masked == old_idx] = new_idx


# Build colormap
cmap   = ListedColormap(roi_colors)
levels = list(range(len(roi_colors)))

# Create one figure with two sub-panels
fig  = plt.figure(figsize=(7, 5))
axes = [
    fig.add_subplot(1, 3, 1, projection='3d'),
    fig.add_subplot(1, 3, 2, projection='3d'),
    fig.add_subplot(1, 3, 3, projection='3d')

]
for ax, v in zip(axes, views):
    # plot ROIs with contours on surface

    # Fill ROI regions
    plotting.plot_surf_roi(
        surf_mesh   = infl_mesh,
        roi_map     = roi_map_reindexed,
        hemi        = HEMI,
        view        = v,
        cmap        = cmap,
        axes        = ax,
        colorbar    = False,
        darkness    = 1
    )
    # Draw ONLY edge lines on top
    plotting.plot_surf_contours(
        surf_mesh = infl_mesh,
        roi_map   = roi_map_reindexed,
        hemi      = HEMI,
        view      = v,
        levels    = levels,
        colors    = ["black"] * len(levels),  
        linewidths= 0.8,
        axes      = ax
    )

plt.tight_layout()

# save the figure
roi_path = FIG_DIR / 'multimodal'
roi_path.mkdir(exist_ok=True, parents=True)
path     = roi_path / f"{HEMI}_sig-pseudo-regions.pdf"
plt.savefig(
    path,
    format      = 'pdf',
    dpi         = 300,
    transparent = True,
    bbox_inches = 'tight',
    pad_inches  = 0.3
)
print(f"\nSuccessful: {path} is saved ")
plt.show()
# %%
