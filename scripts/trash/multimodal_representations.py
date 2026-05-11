# %% [markdown]
# ## fMRI Group-level multimodal analysis (2nd-level)
#
# **Pipeline Overview**
# 1. === STEP 1 ===: Create Harvard-Oxford cortical ROIs figure
# 2. === STEP 2 ===: Extract the beta maps for each subject
# 3. === STEP 3 ===: Pick ROI, extract voxel, convert to vector
# 4. === STEP 4 ===: Compute Pearson correlations (written vs spoken)
# 5. === STEP 5 ===: Fisher's z transformation
# 6. === STEP 6 ===: Save as CSV file
# 7. === STEP 7 ===: Compute mean and SD
# 8. === STEP 8 ===: Two-way mixed ANOVA (2 factors: grade and condition)
# 8. === STEP 9 ===: FDR multiple comaprison correction (Benjamini–Hochberg)


# 7. === STEP 8 ===: Multiple linear regression
# 8. === STEP 8 ===: Plot significant ROI on surface
# 9. === STEP 9 ===: Plot 3D figure (grade, language score, similarity)

# %%
# === Packages ===
from   pathlib import Path
from   statsmodels.stats.multitest import multipletests
from   statsmodels.stats.anova import anova_lm
from   scipy.stats import pearsonr
from   itertools import combinations
from   nilearn import plotting, image, datasets, surface
from   matplotlib.colors import ListedColormap, to_rgba
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib import cm
from scipy.stats import rankdata
import ptitprince as pt

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import math
from collections import Counter
from nilearn.surface import load_surf_mesh
import matplotlib.patches as mpatches
import pingouin as pg
from matplotlib.patches import Patch
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

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
CONTRASTS      = [
                'images_pseudo', # images_words, images_pseudo
                'audios_pseudo', # audios_words, audios_pseudo
                ]
GRADES         = ['1', '2', '4'] # 1, 2, 4 or all
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
EXC_SUBJECTS   = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434']
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
                    "Fusiform": [
                        "Temporal Fusiform Cortex, anterior division",
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
                        "Temporal Fusiform Cortex, anterior division": "green",
                        "Temporal Fusiform Cortex, posterior division": "limegreen",
                        "Temporal Occipital Fusiform Cortex": "lime"
                    }

# %%
# =============================================================
# 1. === STEP 1 ===: Create Harvard-Oxford cortical ROIs figure
# =============================================================

# fsaverage surface
fsaverage  = datasets.fetch_surf_fsaverage()
sulc       = fsaverage.sulc_left
mesh       = surface.load_surf_mesh(fsaverage.pial_left)
white_mesh = surface.load_surf_mesh(fsaverage.white_left)

# Get the list of names with label from the atlas
atlas_img      = HO_ATLAS_MNI6.maps
atlas_labels   = HO_ATLAS_MNI6.labels
label_to_index = {name: i for i, name in enumerate(atlas_labels)}

# Get the label of ROIs
roi_indices = []
for group, regions in ROIs.items():
    for r in regions:
        if r in label_to_index:
            roi_indices.append(label_to_index[r])
roi_indices = np.array(roi_indices)

# Transfrom volumetric HO-Atlas into left surface 
texture_left = surface.vol_to_surf(
            atlas_img,
            surf_mesh     = mesh,
            inner_mesh    = white_mesh,
            interpolation = 'nearest',
            n_samples     = 1
        )
roi_map_left = texture_left.copy()

# assign NaN to not ROI regions
roi_mask = np.isin(roi_map_left, roi_indices)
roi_map_left_masked = roi_map_left.copy()
roi_map_left_masked[~roi_mask] = np.nan

 # Specify view angle of the figure
views = ["lateral", "ventral"]

# Specify color for each ROI
roi_color_map = {
    "Inferior Frontal Gyrus, pars triangularis": "darkgoldenrod",
    "Inferior Frontal Gyrus, pars opercularis": "goldenrod",
    "Supramarginal Gyrus, anterior division": "royalblue",
    "Supramarginal Gyrus, posterior division": "dodgerblue",
    "Angular Gyrus": "navy",
    "Superior Temporal Gyrus, anterior division": "mediumvioletred",
    "Superior Temporal Gyrus, posterior division": "darkmagenta",
    "Temporal Fusiform Cortex, anterior division": "green",
    "Temporal Fusiform Cortex, posterior division": "limegreen",
    "Temporal Occipital Fusiform Cortex": "lime",
}
roi_colors = []
for roi_name, color in roi_color_map.items():
    if roi_name in label_to_index:
        roi_colors.append(color)

# convert original atlas indices to start from 0
index_map         = {idx: i for i, idx in enumerate(roi_indices)}
roi_map_reindexed = np.full_like(roi_map_left_masked, np.nan)
for old_idx, new_idx in index_map.items():
    roi_map_reindexed[roi_map_left_masked == old_idx] = new_idx

# prepare legend 
legend_handles = [
    Line2D(
        [0], [0],
        color = color,
        lw    = 2, # line width
        label = roi_name
    )
    for roi_name, color in roi_color_map.items()
    if roi_name in label_to_index
]

# Build colormap
cmap   = ListedColormap(roi_colors)
levels = list(range(len(roi_colors)))

# Create one figure with two sub-panels
fig  = plt.figure(figsize=(7, 5))
axes = [
    fig.add_subplot(1, 2, 1, projection='3d'),
    fig.add_subplot(1, 2, 2, projection='3d')
]
for ax, v in zip(axes, views):
    # plot ROIs with contours on surface

    plotting.plot_surf_contours(
        surf_mesh = fsaverage.infl_left,
        roi_map   = roi_map_reindexed,
        hemi      = "left",
        view      = v,
        levels    = levels,
        colors    = roi_colors,
        axes      = ax
    )
    # specify legend locations
    if v   == 'lateral':
        legend_handles_clop = legend_handles[:7]
        bbox_to_anchor      = (0.05, 0.30)
    elif v == 'ventral':
        legend_handles_clop = legend_handles[7:]
        bbox_to_anchor      =(0.55, 0.30)

    # add legend
    fig.legend(
        handles        =legend_handles_clop,
        loc            = "upper left",        # position
        bbox_to_anchor = bbox_to_anchor,
        frameon        = False,
        ncol           = 1,
        columnspacing  = 1.5,   # spacing between columns
        handletextpad  = 0.5,    # spacing between line and text
        fontsize       = 9
    )
plt.tight_layout()

# save the figure
roi_path = FIG_DIR / 'roi'
roi_path.mkdir(exist_ok=True, parents=True)
path     = roi_path / "Figure_S2.pdf"
plt.savefig(
    path,
    format      = 'pdf',
    dpi         = 300,
    transparent = True,
    bbox_inches = 'tight',
    pad_inches  = 0.3
)
print(f"\nSuccessful: Figure S2 is saved ")
plt.show()
    


# %%
# ======================================================
# === STEP 2 ===: Extract the beta maps for each subject
# ======================================================

all_correlations = {}

# Get the list of names from the atlas (skipping 'Background')
# Get the label of ROIs
roi_names = []
for group, regions in ROIs.items():
    for r in regions:
        if r in label_to_index:
            roi_names.append(r)

# roi_names = [name for i, name in enumerate(atlas_labels) if name != 'Background']

for MASK in roi_names:
    all_correlations[MASK] = {'left': {}, 'right': {}}

    for GRADE in GRADES:
        subjects      = sorted(DERIV_DIR.glob(f"sub-{GRADE}*"))
        exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
        subjects      = [s for s in subjects if s.name not in exclude]
        subject_names = [p.name.replace('sub-', '') for p in subjects]

        # get the beta paths
        beta_paths = {contrast: {} for contrast in CONTRASTS}
        for contrast in CONTRASTS:
            beta_lists = []
            for sub in subjects:
                path = sub / 'glm' / SPACE / f"FWHM_{int(FWHM_SMOOTHING)}" 
                beta_path = next(path.glob(f"{contrast}_beta.nii.gz"), None)
                beta_lists.append(beta_path)
            beta_paths[contrast] = [str(p) for p in beta_lists]


        # %
        # ===============================================
        # === STEP 3 ===: Pick ROI, and extract the betas
        # ===============================================

        vector = {MASK: {}}
        for hemi in ['right', 'left']:
            vector[MASK][hemi] = {}
            # roi mask for each hemisphere
            roi_path  = MASK_DIR / f"{hemi}_{MASK}_pediatric_MNI.nii.gz"
            roi       = nib.load(roi_path)
            roi_data  = roi.get_fdata().astype(bool)

            for contrast in CONTRASTS:
                vector[MASK][hemi][contrast] = {}

                for sub_name, beta_path in zip(subject_names, beta_paths[contrast]):

                    beta_img  = nib.load(beta_path)
                    beta_data = beta_img.get_fdata()
                            
                    # Apply mask and extract voxels
                    roi_voxels  = beta_data[roi_data]
                    
                    # Convert to 1D vector
                    roi_vector  = roi_voxels.flatten()
                    
                    vector[MASK][hemi][contrast][sub_name] = roi_vector


        # %
        # ====================================================================================
        # === STEP 4 ===: Compute pearson correlations (r) between written and spoken modality
        # ====================================================================================

        for hemi in ['left', 'right']:

            r_vals   = [] # to store r
            z_vals   = [] # to store Fisher's r
            euc_vals = []  # to store Euclidean distances
            cos_vals = []  # to store cosine distances
            corr_dist_vals = []
            spearman_dist_vals = []

            for sub in subject_names:
                # Stack vectors across contrasts
                contrast_vectors = []

                for contrast in CONTRASTS:
                    vec = vector[MASK][hemi][contrast][sub]
                    contrast_vectors.append(vec)

                contrast_matrix = np.vstack(contrast_vectors)

                # Compute Pearson correlation matrix
                corr_matrix = np.corrcoef(contrast_matrix)

                # Extract the specific correlation of interest (r)
                r_val = corr_matrix[1, 0]
                r_vals.append(r_val)
                

                # %
                # =====================================
                # === STEP 5 ===: Fisher transformation
                # =====================================
                # z_val = np.arctanh(np.clip(r_val, -0.999, 0.999))
                # z_vals.append(z_val)


                # %
                # ===========================================
                # === STEP 6 ===: Distance 
                # ===========================================
                # Calculate raw distance
                # 2. Compute Correlation Distance (1 - r)
                rank_visual = rankdata(contrast_matrix[0])
                rank_audio  = rankdata(contrast_matrix[1])

                # 2. Stack the ranked vectors
                X_ranked = np.vstack([rank_visual, rank_audio])

                # 3. Compute the Spearman Correlation Distance (1 - rho)
                # Because we are using ranked data, 'correlation' metric here 
                # effectively computes the Spearman distance.
                spearman_dist_val = pdist(X_ranked, metric='correlation')[0]
                rho = 1 - spearman_dist_val

                # Fisher Z transform
                z_val = np.arctanh(np.clip(rho, -0.999, 0.999))
                z_vals.append(z_val)

                # pdist returns a condensed distance matrix (a 1D array)
                corr_dist_val = pdist(contrast_matrix, metric='correlation')[0]

                # 3. Compute Euclidean Distance (mean centered)
                row_means = np.mean(contrast_matrix, axis=1, keepdims=True)
                contrast_matrix_centered = contrast_matrix - row_means
                euc_dist_val = pdist(contrast_matrix_centered, metric='euclidean')[0]

                # 4. Compute Cosine Distance
                cos_dist_val = pdist(contrast_matrix, metric='cosine')[0]
                
                corr_dist_vals.append(corr_dist_val)
                euc_vals.append(euc_dist_val)
                cos_vals.append(cos_dist_val)
                spearman_dist_vals.append(spearman_dist_val)

            # Store the Z-score instead of the r
            all_correlations[MASK][hemi][GRADE] = {
                                                    'r': r_vals,
                                                    'z': z_vals,
                                                    'euc': euc_vals,
                                                    'cos': cos_vals,
                                                    'corr-dist': corr_dist_vals,
                                                    'spear-dist': spearman_dist_vals,
                                                    }

# %
# =====================================
# === STEP 6 ===: save it as csv file
# =====================================

# convert nested directory to pandas dataframe
rows = []

for roi, hemis in all_correlations.items():
    for hemi, grades in hemis.items():
        for grade, vals in grades.items():

            r_list         = vals['r']
            z_list         = vals['z']
            euc_list       = vals['euc']
            cos_list       = vals['cos']
            corr_dist_list = vals['corr-dist']
            spear_dist_list = vals['spear-dist']

            for i, (r_val, z_val, euc_val, cos_val, corr_dist_var, spear_dist_var) in enumerate(zip(r_list, z_list, euc_list, cos_list, corr_dist_list, spear_dist_list)):
                rows.append({
                    'ROI': roi,
                    'Hemisphere': hemi,
                    'Grade': grade,
                    'Subject_Idx': i,
                    'Pearson_r': float(r_val),
                    'Fisher_Z': float(z_val),
                    'Euclidian': float(euc_val),
                    'Cosine': float(cos_val),
                    'Dist': float(corr_dist_var),
                    'Spear': float(spear_dist_var),
                })

# Create the DataFrame
df_corrs = pd.DataFrame(rows)

# Define the filename
f_name = f'fisher_z_correlations_{CONTRASTS[0]}.csv'
output_filename = OUT_DIR / 'multimodal' / f_name
output_filename.parent.mkdir(exist_ok=True, parents=True)

# Save the DataFrame
df_corrs.to_csv(output_filename, index=False)
print(f"Successfully saved results to: {output_filename}")

# Check the first few rows
print(df_corrs.head())



# %%
# ========================================
# === STEP 7 ===: Describe the mean and SD
# ========================================

# Define the filename
f_name = f'fisher_z_correlations_{CONTRASTS[0]}.csv'
condition = f_name.split('_')[-1].replace('.csv', '')
output_filename = OUT_DIR / 'multimodal' / f_name

# Read the similarity csv file
df_results = pd.read_csv(output_filename)

pearson_stats = []

for roi in df_results['ROI'].unique():
    for hemi in ['left', 'right']:
        for grade in sorted(df_results['Grade'].unique()):

            d = df_results[
                (df_results['ROI'] == roi) &
                (df_results['Hemisphere'] == hemi) &
                (df_results['Grade'] == grade)
            ]

            if not d.empty:
                mean_r    = d['Pearson_r'].mean()
                sd_r      = d['Pearson_r'].std()
                mean_euc  = d['Euclidian'].mean()
                sd_euc    = d['Euclidian'].std()
                mean_cos  = d['Cosine'].mean()
                sd_cos    = d['Cosine'].std()
                mean_dist = d['Dist'].mean()
                sd_dist   = d['Dist'].std()
                mean_spear = d['Spear'].mean()
                sd_spear   = d['Spear'].std()

                pearson_stats.append({
                    'ROI': roi,
                    'Hemisphere': hemi,
                    'Grade': grade,
                    'mean_Pearson_r': mean_r,
                    'sd_Pearson_r': sd_r,
                    'mean_Euclidian': mean_euc,
                    'sd_Euclidian': sd_euc,
                    'mean_Cos': mean_cos,
                    'sd_Cos': sd_cos,
                    'mean_Dist': mean_dist,
                    'sd_Dist': sd_dist,
                    'mean_Spear': mean_spear,
                    'sd_Spear': sd_spear,
                    'n_subjects': len(d)
                })

# Convert to DataFrame
df_pearson_stats = pd.DataFrame(pearson_stats)

# Print in the requested layout using df_pearson_stats
for roi in df_pearson_stats['ROI'].unique():

    print(f"\n=== {roi} ===")

    roi_df = df_pearson_stats[df_pearson_stats['ROI'] == roi]

    for grade in sorted(roi_df['Grade'].unique()):

        g = roi_df[roi_df['Grade'] == grade]

        left  = g[g['Hemisphere'] == 'left']
        right = g[g['Hemisphere'] == 'right']

        if not left.empty:
            print(" ")
            print(f"Grade {grade}")

            l_mean     = left['mean_Pearson_r'].values[0]
            l_sd       = left['sd_Pearson_r'].values[0]

            r_mean     = right['mean_Pearson_r'].values[0]
            r_sd       = right['sd_Pearson_r'].values[0]

            l_mean_euc = left['mean_Euclidian'].values[0]
            l_sd_euc   = left['sd_Euclidian'].values[0]

            r_mean_euc = right['mean_Euclidian'].values[0]
            r_sd_euc   = right['sd_Euclidian'].values[0]

            l_mean_cos = left['mean_Cos'].values[0]
            l_sd_cos   = left['sd_Cos'].values[0]

            r_mean_cos = right['mean_Cos'].values[0]
            r_sd_cos   = right['sd_Cos'].values[0]

            l_mean_dis = left['mean_Dist'].values[0]
            l_sd_dis   = left['sd_Dist'].values[0]

            r_mean_dis = right['mean_Dist'].values[0]
            r_sd_dis   = right['sd_Dist'].values[0]

            l_mean_spear = left['mean_Spear'].values[0]
            l_sd_spear   = left['sd_Spear'].values[0]

            r_mean_spear = right['mean_Spear'].values[0]
            r_sd_spear   = right['sd_Spear'].values[0]

            print("Pearson")
            print(f"L: {l_mean:.2f} ({l_sd:.2f})")
            print(f"R: {r_mean:.2f} ({r_sd:.2f})")
            print("Euclidian")
            print(f"L: {l_mean_euc:.2f} ({l_sd_euc:.2f})")
            print(f"R: {r_mean_euc:.2f} ({r_sd_euc:.2f})")
            print("Cosine")
            print(f"L: {l_mean_cos:.2f} ({l_sd_cos:.2f})")
            print(f"R: {r_mean_cos:.2f} ({r_sd_cos:.2f})")
            print("Corr-Distance")
            print(f"L: {l_mean_dis:.2f} ({l_sd_dis:.2f})")
            print(f"R: {r_mean_dis:.2f} ({r_sd_dis:.2f})")
            print("Spear-Distance")
            print(f"L: {l_mean_spear:.2f} ({l_sd_spear:.2f})")
            print(f"R: {r_mean_spear:.2f} ({r_sd_spear:.2f})")





# %%
# =====================================================================
#  === STEP 8 ===: Two-way mixed ANOVA (2 factors: grade and condition)
# =====================================================================

# get the data
f_name_w          = 'fisher_z_correlations_images_words.csv'
f_name_p          = 'fisher_z_correlations_images_pseudo.csv'
path_w            = OUT_DIR / 'multimodal' / f_name_w
path_p            = OUT_DIR / 'multimodal' / f_name_p
df_w              = pd.read_csv(path_w)
df_p              = pd.read_csv(path_p)
df_w["Condition"] = "word"
df_p["Condition"] = "pseudo"
df                = pd.concat([df_w, df_p], ignore_index=True)
df["Subject_ID"]  = df["Grade"].astype(str) + df["Subject_Idx"].astype(str)

# ANOVA
DV_COL            = "Euclidian"
df["Grade"]       = df["Grade"].astype("category")
df["Condition"]   = df["Condition"].astype("category")
df["Subject_ID"]  = df["Subject_ID"].astype("category")

results = []
for roi in df["ROI"].unique():
    df_roi = df[(df["ROI"] == roi) & (df["Hemisphere"] == "left")]

    aov = pg.mixed_anova(
        dv=DV_COL,
        within="Condition",
        between="Grade",
        subject="Subject_ID",
        data=df_roi
    )

    aov["ROI"] = roi
    results.append(aov)

anova_roiwise = pd.concat(results, ignore_index=True)
print(anova_roiwise)

# %%
# =====================================
#  === STEP 9 ===: Bonferoni correction
# =====================================

# paramters
alpha       = 0.05
n           = len(df["ROI"].unique())
bon_thr     = alpha / n

# apply the threshold
df_anova          = anova_roiwise.copy()
df_anova["p-bon"] = df_anova["p-unc"] < bon_thr
sig_df            = df_anova[df_anova["p-bon"]==True]
sig_rois          = np.array(sig_df["ROI"])
print(sig_df)



# %%
# ==================================================
#  === STEP 10 ===: VIsualize similarity of all ROIs
# ==================================================
df_left = df = df[df["Hemisphere"] == "left"]
agg = df_left.groupby(["ROI", "Grade", "Condition"])["Spear"].mean().reset_index()

roi_names = []
for group, regions in ROIs.items():
    for r in regions:
            roi_names.append(r)

# --- Encode categories ---
grades = sorted(agg["Grade"].unique())
conditions = ["word", "pseudo"]

spacing = 1.5   # <-- increase this (try 1.5, 2, 3)
roi_to_num = {roi: i * spacing for i, roi in enumerate(roi_names)}

grade_to_num = {grade: i * spacing for i, grade in enumerate(grades)}
cond_to_offset = {"word": -0.2, "pseudo": 0.2}  # <-- key step

agg["roi_num"] = agg["ROI"].map(roi_to_num)
agg["grade_num"] = agg["Grade"].map(grade_to_num)

# --- Plot ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

dx = 0.35
dy = 0.35

alpha_map = {
    "word": 1.0,
    "pseudo": 0.0
}

for _, row in agg.iterrows():
    x = row["roi_num"] + cond_to_offset[row["Condition"]]  # shift by condition
    y = row["grade_num"]
    z = 0
    dz = row["Spear"]
    alpha = alpha_map[row["Condition"]]   # <-- key line

    color = roi_color_map[row["ROI"]]   # <-- key line
    ax.bar3d(x, y, z, 
             dx, dy, dz, 
             color=color,
            edgecolor="black",   # <-- border color
            linewidth=0.4,
            alpha=alpha)

# --- Labels ---
ax.set_ylabel("Grade")
ax.set_zlabel("Dissimilarity")

# --- Ticks ---
ax.set_xticks(list(roi_to_num.values()))
x_labels = ['IFG-t', 'IFG-o', 'SMG-a', 'SMG-p', 'Ang', 'STG-a', 'STG-p', 'TFC-a', 'TFC-p', 'TOF']

ax.set_xticklabels(x_labels, rotation=45, ha="right")

ax.set_yticks(list(grade_to_num.values()))
ax.set_yticklabels(list(grade_to_num.keys()))

ax.set_zticks(np.arange(0,1.5,0.5))

# --- View ---
ax.view_init(elev=0, azim=120)

legend_elements = [
    Patch(facecolor="gray", edgecolor="black", alpha=1.0, label="word"),
    Patch(facecolor="none", edgecolor="black", alpha=1.0, label="pseudo")
]

ax.legend(
    handles=legend_elements, 
    title="Condition", 
    loc="upper right", 
    bbox_to_anchor=(0.90, 0.75), # Moves it slightly in from the top-right edge
    frameon=True,                # Adds a box around the legend
    facecolor='white',           # Ensures the background of the legend is opaque
    edgecolor='lightgray'
)
plt.tight_layout()
# save the figure
roi_path = FIG_DIR / 'roi'
roi_path.mkdir(exist_ok=True, parents=True)
path     = roi_path / "Figure.pdf"
plt.savefig(
    path,
    format='pdf',
    dpi=300,
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.3
)
print(f"\nSuccessful: Figure is saved ")
plt.show()


# %%
# ============================================
#  === STEP 10 ===: VIsualize significant ROIs
# ============================================
# Create one figure with two sub-panels
fig  = plt.figure(figsize=(7, 5))
# plot ROIs with contours on surface
# Specify color for each ROI
sig_index = []
for i, region in enumerate(roi_color_map.keys()):
    if region in sig_rois:
        sig_index.append(i)
roi_masked = roi_map_reindexed.copy()
roi_masked[~np.isin(roi_masked, sig_index)] = np.nan
colors = [roi_colors[i] for i in sig_index]
plotting.plot_surf_contours(
    surf_mesh = fsaverage.infl_left,
    roi_map   = roi_masked,
    hemi      = "left",
    view      = "lateral",
    levels    = sig_index,
    colors    = colors
)
# save the figure
roi_path = FIG_DIR / 'roi'
roi_path.mkdir(exist_ok=True, parents=True)
path     = roi_path / "Figure_sig.pdf"
plt.savefig(
    path,
    format      = 'pdf',
    dpi         = 300,
    transparent = True,
    bbox_inches = 'tight',
    pad_inches  = 0.3
)
print(f"\nSuccessful: Figure_sig is saved ")
plt.show()



# %%
# =============================================
#  === STEP 11 ===: VIsualize each distribution
# =============================================
condition = "word"

df_left = df = df[(df["Hemisphere"] == "left") & (df["Condition"] == condition)]
df_sig = df_left[df_left["ROI"].isin(sig_rois)].copy() 

titles = ['IFG-o', 'SMG-a', 'SMG-p', 'STG-p']

# 1. Setup the figure grid (2x2)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten to easily loop with a single index

# Colors and configuration
grade_map = {1: 0, 2: 1, 4: 2} # Map Grade labels to X-axis indices

for i, roi in enumerate(sig_rois):
    ax = axes[i]
    df_roi = df_sig[df_sig["ROI"] == roi]
    num_grades = df_roi["Grade"].nunique()

    dot_color = roi_color_map[roi]
    cloud_color = roi_color_map[roi]

    # --- Statistics: Pairwise T-Tests ---
    ttest = pg.pairwise_tests(data=df_roi, dv='Fisher_Z', between='Grade', 
                            padjust='bonf', effsize='hedges')

    # --- Plotting: RainCloud ---
    pt.RainCloud(x="Grade", y="Fisher_Z", data=df_roi,
                color=cloud_color, 
                palette=[dot_color] * num_grades, 
                point_size=7,
                rain_edgecolor='white',
                rain_alpha=0.6,
                width_viol=0.8, 
                width_box=0.15, 
                cloud_alpha=1,
                offset=0.15, 
                pointplot=True,
                linecolor="black",
                ax=ax)

    # --- Adding Significance Marks ---
    y_max = df_roi["Fisher_Z"].max()
    y_min = df_roi["Fisher_Z"].min()
    # Only plot marks for significant post-hocs
    sig_comparisons = ttest[ttest['p-corr'] < 0.05]
    
    for idx, row in sig_comparisons.iterrows():
        # Define x positions and height for brackets
        x1, x2 = grade_map[row['A']], grade_map[row['B']]
        # Stack bars slightly if there are multiple significant results
        gap_height = 0.2
        level = y_max + 0.1 + (idx * gap_height) 
        
        # Draw bracket
        ax.plot([x1, x1, x2, x2], [level, level+0.02, level+0.02, level], lw=1, c='black')
        # Add stars - Check from most significant to least significant
        p_val = row['p-corr']
        if p_val < 0.001:
            stars = "***"
        elif p_val < 0.01:
            stars = "**"
        elif p_val < 0.05:
            stars = "*"
            
        ax.text((x1+x2)*0.5, level+0.02, stars, ha='center', va='bottom', fontsize=12)

    # --- Aesthetics ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([y_min - 0.2, y_max + 0.8]) # Dynamic limit to fit stars
    
    # Use a short version of the ROI name for the title
    short_title = titles[i] # Cuts off "pars opercularis" if desired, or use custom mapping
    ax.set_title(short_title, fontsize=17, fontweight='bold')
    
    ax.set_xlabel("Grade", fontsize=13)
    ax.set_ylabel(r"Similarity (Fisher's $r$)", fontsize=13)

# Adjust layout to prevent overlapping
plt.tight_layout()

# --- Save Final Figure ---
roi_path = FIG_DIR / 'multimodal'
roi_path.mkdir(exist_ok=True, parents=True)
plt.savefig(roi_path / f"All_Sig_ROIs_{condition}.pdf", format='pdf', dpi=300, transparent=True, bbox_inches='tight')

print(f"Successful: 4-panel figure saved to {roi_path}")
plt.show()
# %%

# plot ROIs with contours on surface
# Specify color for each ROI
sig_index = []
for i, region in enumerate(roi_color_map.keys()):
    if region in sig_rois:
        sig_index.append(i)
roi_masked = roi_map_reindexed.copy()
roi_masked[~np.isin(roi_masked, sig_index)] = np.nan
colors = [roi_colors[i] for i in sig_index]
plotting.plot_surf_contours(
    surf_mesh = fsaverage.infl_left,
    roi_map   = roi_masked,
    hemi      = "left",
    view      = "lateral",
    levels    = sig_index,
    colors    = colors
)
# save the figure
roi_path = FIG_DIR / 'roi'
roi_path.mkdir(exist_ok=True, parents=True)
path     = roi_path / "Figure_sig.pdf"
plt.savefig(
    path,
    format      = 'pdf',
    dpi         = 300,
    transparent = True,
    bbox_inches = 'tight',
    pad_inches  = 0.3
)
print(f"\nSuccessful: Figure_sig is saved ")
plt.show()


# %%
# ==========================================
# === STEP 8 ===: Multiple linear regression
# ==========================================
# Similarity = β0​ + β1​*(grade) + β2​*(task accuracy) + ε

# Define the filename
f_name = f'fisher_z_correlations_{CONTRASTS[0]}.csv'
condition = f_name.split('_')[-1].replace('.csv', '')
output_filename = OUT_DIR / 'multimodal' / f_name

# Read the similarity csv file
df_results = pd.read_csv(output_filename)

subjects      = sorted(DERIV_DIR.glob("sub-*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]

# Read the task performance csv file
results = []
for sub in subjects:
    task_path = sub / 'behavior' / 'accuracy_summary.csv'
    task_df  = pd.read_csv(task_path)
    mean_acc = task_df['accuracy_all'].mean()
    # Store results as a dict
    results.append({
        'subject': sub.name,
        'mean_accuracy': mean_acc
    })
# Convert list of dicts to a DataFrame
task_df = pd.DataFrame(results)

# Use all ROIs (not a single TARGET_ROI) for multiple-testing correction
df = df_results.copy()
df["Hemisphere"] = df["Hemisphere"].str.lower()
grade_map = {1:1, 2:2, 4:3}
df["grade_rank"] = df["Grade"].map(grade_map)


rows = []
for roi in df["ROI"].dropna().unique():
    for hemi in ["left", "right"]:
        d = df[(df["ROI"] == roi) & (df["Hemisphere"] == hemi)].copy()

        # Assuming 'subject' column exists in both df and task_df
        d['mean_acc'] = task_df['mean_accuracy'].values

        # FULL model
        m = smf.ols("Fisher_Z ~ grade_rank + mean_acc", data=d).fit()

        # REDUCED model 
        reduced_model_acc = smf.ols("Fisher_Z ~ mean_acc", data=d).fit()
        reduced_model_grade = smf.ols("Fisher_Z ~ grade_rank", data=d).fit()

        # This is the Prob(F-statistic)
        f_model = m.fvalue 
        p_model = m.f_pvalue
        df_model = m.df_model
        residu_model = m.df_resid 

        # R² values
        r2_full          = m.rsquared
        r2_reduced_acc   = reduced_model_acc.rsquared
        r2_reduced_grade = reduced_model_grade.rsquared

        # partial R²
        partial_r2_grade = (r2_full - r2_reduced_acc) / (1 - r2_reduced_acc)
        partial_r2_acc   = (r2_full - r2_reduced_grade) / (1 - r2_reduced_grade)

        # print(f"\n=== {hemi.upper()} {roi} ===")
        # print(m.summary())

        rows.append({
            "ROI": roi,
            "Hemisphere": hemi,
            "n": int(m.nobs),

            "beta_grade": m.params["grade_rank"],
            "p_grade": m.pvalues["grade_rank"],
            "t_grade": m.tvalues["grade_rank"],
            "se_grade": m.bse["grade_rank"],
            "ci2.5_grade": m.conf_int().loc["grade_rank", 0],
            "ci97.5_grade": m.conf_int().loc["grade_rank", 1],

            "beta_acc": m.params["mean_acc"],
            "p_acc": m.pvalues["mean_acc"],
            "t_acc": m.tvalues["mean_acc"],
            "ci2.5_acc": m.conf_int().loc["mean_acc", 0],
            "ci97.5_acc": m.conf_int().loc["mean_acc", 1],

            "r2": m.rsquared,
            "r2_partial_grade": partial_r2_grade,
            "r2_partial_acc": partial_r2_acc,

            "p_model": p_model,
            "f_model": f_model,
            "df_model": df_model,
            "residu_model": residu_model    
        })

res = pd.DataFrame(rows)

# FDR (Benjamini-Hochberg) across ALL ROI x hemisphere tests
res_list = []

# Whole-brain FDR: across all ROI x hemisphere tests together
res_corrected = res.copy()

# FDR for overall model significance
rej_model, q_model, _, _ = multipletests(res_corrected["p_model"], alpha=0.05, method="fdr_by")
res["q_model_fdr"] = q_model
res["sig_model_fdr_0.05"] = rej_model

# Filter significant models
significant_models = res[res["sig_model_fdr_0.05"]]

pd.options.display.float_format = "{:.2f}".format
for hemi in ["left", "right"]:
    hemi_df = significant_models[significant_models["Hemisphere"] == hemi]

    # sort by r2 (highest first)
    hemi_df = hemi_df.sort_values(by="r2", ascending=False)

    print(f"\n=== {hemi.upper()} hemisphere: ROIs showing significant model effect (FDR < 0.05) ===")

    if not hemi_df.empty:
        print(hemi_df[[
            "ROI", "Hemisphere", "r2",
            "f_model", "q_model_fdr",
            "beta_grade","t_grade", "p_grade", "ci2.5_grade", "ci97.5_grade", "r2_partial_grade",
            "beta_acc", "t_acc", "p_acc","ci2.5_acc", "ci97.5_acc", "r2_partial_acc"
        ]])
    else:
        print("No ROIs survived FDR correction.")

    print(f"Summary ({hemi}): {len(hemi_df)} out of {len(res[res['Hemisphere']==hemi])} models were significant.")




# %%
# =====================================================
# === STEP 8 ===: Plot significant ROIs on pial surface
# ===================================================== 
RSA_DIR = FIG_DIR / 'multimodal'
RSA_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the nested dictionary structure
sig_rois = {
    "left": {},
    "right": {}
}
# Fill the dictionary
for _, row in significant_models.iterrows():
    hemi = row["Hemisphere"].lower()
    roi_name = row["ROI"]
    r2_val = row[f"r2"]
    
    # Assign the beta value to the specific ROI under the correct hemisphere
    sig_rois[hemi][roi_name] = r2_val

# MNI atlas
atlas      = HO_ATLAS_MNI6.maps
atlas_data = atlas.get_fdata()

# fsaverage surface
fsaverage = datasets.fetch_surf_fsaverage()

hemis = ['left', 'right']

# Prepare figure: 1 row x 4 columns (left-lateral, left-medial, right-lateral, right-medial)
fig, axes = plt.subplots(
    2, 2,
    figsize=(4, 4),
    subplot_kw={'projection': '3d'}
)
fig.subplots_adjust(wspace=0, hspace=0)
cmap = plt.get_cmap('hot')
norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
views = ['lateral', 'medial']
for col, hemi in enumerate(["left", "right"]):

    if hemi == "left":
        mesh       = surface.load_surf_mesh(fsaverage.pial_left)
        white_mesh = surface.load_surf_mesh(fsaverage.white_left)
        sulc       = fsaverage.sulc_left
    else:
        mesh       = surface.load_surf_mesh(fsaverage.pial_right)
        white_mesh = surface.load_surf_mesh(fsaverage.white_right)
        sulc       = fsaverage.sulc_right

    rois = sig_rois[hemi]
    master_beta_surf = np.zeros(len(mesh[0]))

    for roi_name, beta in rois.items():
        idx_row = atlas_labels.loc[atlas_labels['name'] == roi_name, 'index'].values
        roi_mask = np.isin(atlas_data, idx_row)
        roi_atlas_data = np.where(roi_mask, atlas_data, 0)
        roi_img = nib.Nifti1Image(roi_atlas_data, atlas.affine)

        roi_map_data = surface.vol_to_surf(
            roi_img,
            surf_mesh=mesh,
            inner_mesh=white_mesh,
            interpolation='linear',
            n_samples=1
        )

        master_beta_surf[roi_map_data > 0] = beta

    master_beta_surf[master_beta_surf == 0] = np.nan

    for row, view in enumerate(views):

        ax = axes[row, col]

        plotting.plot_surf_roi(
            surf_mesh=mesh,
            roi_map=master_beta_surf,
            hemi=hemi,
            view=view,
            bg_map=sulc,
            cmap=cmap,
            vmin=0,
            vmax=0.5,
            alpha=1,
            colorbar=False,
            darkness=None,
            axes=ax
        )

        ax.set_axis_off()

# Add a single horizontal colorbar at the bottom
cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.03])  # [left, bottom, width, height]
sm      = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar    = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', format='%.2f')
cbar.set_label(r'$R$-squared', fontsize=6, labelpad=5)
cbar.ax.tick_params(labelsize=5)


# save first
path = RSA_DIR / f"{condition}_surface.pdf"
fig.savefig(path, format='pdf', transparent=True, bbox_inches='tight')

plt.show()


# %%
# ======================================================================
# === STEP 9 ===: Plot 3D figure 
# (y-axis: grade, x-axis: language score, z-axis: multimodal similarity)
# ======================================================================


roi = "Superior Temporal Gyrus, posterior division"

hemis   = ['left', 'right']

grade_colors = {
    1: "darkgoldenrod", 
    2: "darkcyan", 
    4: "firebrick"
}


fig = plt.figure(figsize=(8,8))

axes = [
    fig.add_subplot(221, projection='3d'),
    fig.add_subplot(222, projection='3d'),
    fig.add_subplot(223, projection='3d'),
    fig.add_subplot(224, projection='3d')
]

fig.subplots_adjust(hspace=0.0, wspace=0.1)


ax_idx = 0
r_squared = []

for hemi in hemis:

    df = df_results[
        (df_results['Hemisphere'] == hemi) &
        (df_results['ROI'] == roi)
    ].copy()

    df['Score'] = task_df['mean_accuracy'].values

    grade_map = {1:1, 2:2, 4:3}
    df["grade_rank"] = df["Grade"].map(grade_map)

    X = df[['grade_rank','Score']].values
    y = df['Fisher_Z'].values

    x = X[:,0]
    y2 = X[:,1]

    x_pred = np.linspace(x.min(), x.max(), 30)
    y_pred = np.linspace(y2.min(), y2.max(), 30)

    xx, yy = np.meshgrid(x_pred, y_pred)
    grid = np.c_[xx.ravel(), yy.ravel()]

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    r_squared.append(r2)

    z_pred = model.predict(grid)
    zz = z_pred.reshape(xx.shape)

    rank_to_grade = {1:1,2:2,3:4}
    xticks = [1,2,3]
    xticklabels = [rank_to_grade[i] for i in xticks]

    colors = [grade_colors[g] for g in df["Grade"]]

    # two views per hemisphere
    for view in range(2):

        ax = axes[ax_idx]

        ax.scatter(X[:,0], X[:,1], y, color=colors, alpha=0.5, s=30)
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='indigo')

        ax.set_ylabel('Language (%)')
        ax.set_xlabel('Grade')
        ax.set_zlabel("Similarity ($z$)")

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        ax.set_zticks(np.arange(0,2.6,0.5))
        ax.set_yticks(np.arange(0,101,50))

        if view == 0:
            ax.view_init(elev=10, azim=100)
        else:
            ax.view_init(elev=10, azim=170)

        ax_idx += 1



# title
fig.suptitle("STG (posterior)", fontsize=16, y=0.92)

# R-squared
fig.text(0.5, 0.85, '$R^2 = %.2f$' % r_squared[0], rotation=0,
         va='center', ha='center', fontsize=14)

fig.text(0.5, 0.45, '$R^2 = %.2f$' % r_squared[1], rotation=0,
         va='center', ha='center', fontsize=14)

# hemisphere text
fig.text(0.15, 0.85, "Left", rotation=0,
         va='center', ha='center', fontsize=14)

fig.text(0.15, 0.45, "Right", rotation=0,
         va='center', ha='center', fontsize=14)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='1st',
           markerfacecolor=grade_colors[1], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='2nd',
           markerfacecolor=grade_colors[2], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='4th',
           markerfacecolor=grade_colors[4], markersize=8)
]

fig.legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.92),
    frameon=False
)

# save the figure 
path = RSA_DIR / f"{roi}_both_hemispheres_3dplot.pdf"
fig.savefig(
    path,
    format='pdf',
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.3
)

plt.show()







