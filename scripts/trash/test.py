# %% [markdown]
# ## fMRI Group-level multimodal analysis (2nd-level)
#
# **Pipeline Overview**

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
from scipy import stats
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
MASK_DIR     = TEMP_DIR  / 'mask'


# === Parameters ===
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
                    }

# %%
# 1. === STEP 1 ===: Create Harvard-Oxford cortical ROIs figure

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
# 2. === STEP 2 ===: Extract all the availoabel beta maps for each rusubject
# ROIs
roi_names = []
for group, regions in ROIs.items():
    for r in regions:
        roi_names.append(r)

# Subjects
subjects      = sorted(DERIV_DIR.glob(f"sub-*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]
subject_names = [p.name.replace('sub-', '') for p in subjects]


all_results = {}
# subject loop
for subject in subjects:
    # subject = subjects[0]
    glm_path = subject / MODEL / SPACE / 'FWHM_9'
    folders = [p for p in glm_path.rglob('*') if p.is_dir()]

    # run loop
    written_beta_lists = []
    spoken_beta_lists  = []
    written_pseudo_beta_lists = []
    spoken_pseudo_beta_lists  = []
    written_semantic_beta_lists = []
    spoken_semantic_beta_lists  = []

    for folder in folders:
        written_beta_path          = folder / f'{CONTRASTS[0]}_beta.nii.gz'
        spoken_beta_path           = folder / f'{CONTRASTS[1]}_beta.nii.gz'
        written_pseudo_beta_path   = folder / f'{CONTRASTS[2]}_beta.nii.gz'
        spoken_pseudo_beta_path    = folder / f'{CONTRASTS[3]}_beta.nii.gz'
        written_semantic_beta_path = folder / f'{CONTRASTS[4]}_beta.nii.gz'
        spoken_semantic_beta_path  = folder / f'{CONTRASTS[5]}_beta.nii.gz'

        written_beta_lists.append(written_beta_path)
        spoken_beta_lists.append(spoken_beta_path)
        written_pseudo_beta_lists.append(written_pseudo_beta_path)
        spoken_pseudo_beta_lists.append(spoken_pseudo_beta_path)
        written_semantic_beta_lists.append(written_semantic_beta_path)
        spoken_semantic_beta_lists.append(spoken_semantic_beta_path)

    combined = written_beta_lists + spoken_beta_lists + written_pseudo_beta_lists + spoken_pseudo_beta_lists + written_semantic_beta_lists + spoken_semantic_beta_lists

    # %
    # 3. === STEP 3 ===: Pick ROI, extract voxel, convert to vector
    vector = {}
    for roi_name in roi_names:
        roi_data = nib.load(
            MASK_DIR / f"left_{roi_name}_pediatric_MNI.nii.gz"
        ).get_fdata().astype(bool)

        vec = np.vstack([
            nib.load(p).get_fdata()[roi_data].flatten()
            for p in combined
        ])
        stds = np.std(vec, axis=1)
        idx  = np.where(stds == 0)[0]
        bad_paths = [combined[i] for i in idx]
        bad_runs = {p.parts[-2] for p in bad_paths}  # extracts "run-08"
        filtered_paths = [
            p for p in combined
            if p.parts[-2] not in bad_runs
        ]

        vector[roi_name] = np.vstack([
            nib.load(p).get_fdata()[roi_data].flatten()
            for p in filtered_paths
        ])
    

    # %
    # 5. === STEP 5 ===: Representational Dissimilarity Matrix (written vs spoken)
    corr_matrices = {
        roi: 1 - np.corrcoef(data)
        for roi, data in vector.items()
    }

    roi_target = 'Temporal Fusiform Cortex, anterior division'
    corr = corr_matrices[roi_target]

    plt.imshow(corr, vmin=0, vmax=2)
    cbar = plt.colorbar(label='Dissimilarity (1 - r)')
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['0', '1', '2'])
    plt.title(roi_target, fontsize=14)
    plt.xlabel('Run', fontsize=11)
    plt.ylabel('Run', fontsize=11)
    # ticks starting from 1
    n = corr.shape[0]
    # repeating 1 – max(run) 
    labels = np.tile(np.arange(1, n//len(CONTRASTS) + 1), len(CONTRASTS))
    plt.xticks(range(n), labels)
    plt.yticks(range(n), labels)
    # save the figure
    roi_path = FIG_DIR / 'multimodal'
    roi_path.mkdir(exist_ok=True, parents=True)
    path     = roi_path / "RDM.pdf"
    """     plt.savefig(
        path,
        format      = 'pdf',
        dpi         = 300,
        transparent = True,
        bbox_inches = 'tight',
        pad_inches  = 0.3
    )
    print(f"\nSuccessful: Figure RDM is saved ") """
    plt.show()

    # %
    # 6. === STEP 6 ===: Compute each RDM metrics
    results = {}
    for region, corr in corr_matrices.items():
        n_runs = int(corr.shape[0] / 6)

        word   = range(1, n_runs + 1)
        spoken = range(n_runs + 1, 2 * n_runs + 1)

        base_multi_index = []

        for i in spoken:
            for j in word:
                if j >= i - n_runs + 1:
                    base_multi_index.append((i, j))

        for i in spoken:
            for j in word:
                if j <= i - n_runs - 1:
                    base_multi_index.append((i, j))

        idx_word_multi     = [(i-1, j-1) for i, j in base_multi_index]
        idx_pseudo_multi   = [(i+(n_runs*2), j+(n_runs*2)) for i, j in idx_word_multi]
        idx_semantic_multi = [(i+(n_runs*3), j+(n_runs*3)) for i, j in idx_word_multi]


        results[region] = {
            "word_multi": 1 - np.mean([corr[i, j] for i, j in idx_word_multi]),
            "pseudo_multi": 1 - np.mean([corr[i, j] for i, j in idx_pseudo_multi]),
            "semantic_multi": 1 - np.mean([corr[i, j] for i, j in idx_semantic_multi]),
        }

    all_results[subject.name] = results

# %
# save as csv file
rows = []

for subject, rois in all_results.items():
    for roi, metrics in rois.items():
        row = {
            "subject": subject,
            "roi": roi
        }
        row.update(metrics)
        rows.append(row)

df = pd.DataFrame(rows)
df['grade'] = df['subject'].str.extract(r'-(\d+)').astype(int) // 100
df.isna().any().any()

# combine RT
results = []
for sub in subjects:
    task_path = sub / 'behavior' / 'accuracy_summary.csv'
    task_df  = pd.read_csv(task_path)
    mean_rt  = task_df['RT_all'].mean()

    # Store results as a dict
    results.append({
        'subject': sub.name,
        'rt': mean_rt,

    })
task_df = pd.DataFrame(results)

df_merged = df.merge(
    task_df[["subject", "rt"]],
    on="subject",
    how="left"
)


path = OUT_DIR / 'multimodal' / 'RDM metrics.csv'
df_merged.to_csv(path, index=False)





# %%
path = OUT_DIR / 'multimodal' / 'RDM metrics.csv'
df = pd.read_csv(path)


# %
# remove outliers
def get_removed_outliers(df, group_col, value_col, k=1.5): 
    def _mask(group): 
        q1 = group[value_col].quantile(0.25) 
        q3 = group[value_col].quantile(0.75) 
        iqr = q3 - q1 
        lower = q1 - k * iqr 
        upper = q3 + k * iqr 
        return (group[value_col] < lower) | (group[value_col] > upper) 
    mask = df.groupby(group_col, group_keys=False).apply(_mask) 
    return df[mask]

conditions = ['word_multi', 'pseudo_multi', 'semantic_multi']

# store outputs
df_clean_dict = {}
removed_dict  = {}

for cond in conditions:
    # subset
    df_cond = df[['roi', 'grade', 'rt', cond]].copy()

    # --- detect outliers ---
    removed = get_removed_outliers(df_cond, 'roi', cond)

    # --- remove outliers ---
    df_clean = df_cond.drop(removed.index).copy()

    # --- Fisher transform ---
    df_clean["Fisher"] = np.arctanh(
        df_clean[cond].clip(-0.999999, 0.999999)
    )

    # --- store ---
    df_clean_dict[cond] = df_clean
    removed_dict[cond]  = removed

    # --- optional print ---
    print(f"\nRemoved rows for {cond}:")
    print(removed)


fig, axes = plt.subplots(1, 3, figsize=(20, 6))
axes = axes.flatten()  # Flatten to easily loop with a single index

titles    = ['Word', 'Pseudo', 'Semantic']
xlabels   = ['IFG-pt', 'IFG-po', 'SMG-a', 'SMG-p', 'Angular', 'STG-a', 'STG-p', 'TFC-a', 'TFC-p']
order     = list(roi_color_map.keys())
palette   = list(roi_color_map.values())

for i, cond in enumerate(conditions):
    ax = axes[i]
    df_cond = df_clean_dict[cond]

    # --- Plotting: RainCloud ---
    pt.RainCloud(x="roi", y=cond, data=df_cond,
                order=order,
                palette=palette,
                point_size=3,
                rain_alpha=0.6,
                width_viol=1, 
                width_box=0.3, 
                cloud_alpha=1,
                offset=0.2, 
                box_showfliers=False,  
                ax=ax)

    # --- Aesthetics ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([-0.5, 1]) # Dynamic limit to fit stars
    ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Use a short version of the ROI name for the title
    short_title = titles[i] # Cuts off "pars opercularis" if desired, or use custom mapping
    ax.set_title(short_title, fontsize=17, fontweight='bold')
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.axhline(y=0, linestyle='--', linewidth=1, color='black', alpha=0.7)
    ax.set_xlabel("ROIs", fontsize=13)
    ax.set_ylabel(r"Multimodal similarity ($r$)", fontsize=13)

# Adjust layout to prevent overlapping
plt.tight_layout()
# save the figure
roi_path = FIG_DIR / 'multimodal'
roi_path.mkdir(exist_ok=True, parents=True)
path     = roi_path / "Correlations.pdf"
plt.savefig(
    path,
    format      = 'pdf',
    dpi         = 300,
    transparent = True,
    bbox_inches = 'tight',
    pad_inches  = 0.3
)
print(f"\nSuccessful: Figure is saved ")
plt.show()




# %%
# bar plot
# sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
axes      = axes.flatten()  # Flatten to easily loop with a single index
palette   = list(roi_color_map.values())
ylim      = (-0.2, 0.6)

# ---------------- WORD ----------------
ax1 = sns.barplot(
    data=df_clean_dict["word_multi"],
    x="grade",
    y="word_multi",
    hue="roi",
    palette=roi_color_map,
    ax=axes[0]
)
axes[0].tick_params(axis="both", which="major", length=3, width=1)
axes[0].set_ylim(ylim)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
axes[0].set_title("Word", fontsize=15, fontweight="bold")
axes[0].set_xlabel("Grade", fontsize=15)
axes[0].set_ylabel(r"Multimodal similarity ($r$)", fontsize=15)
# remove legend
if ax1.legend_:
    ax1.legend_.remove()


# ---------------- PSEUDO ----------------
ax2 = sns.barplot(
    data=df_clean_dict["pseudo_multi"],
    x="grade",
    y="pseudo_multi",
    hue="roi",
    palette=roi_color_map,
    ax=axes[1]
)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
axes[1].tick_params(axis="both", which="major", length=3, width=1)
axes[1].set_ylim(ylim)
axes[1].set_title("Pseudo", fontsize=15, fontweight="bold")
axes[1].set_xlabel("Grade", fontsize=15)
axes[1].set_ylabel(r"Multimodal similarity ($r$)", fontsize=15)
# remove legend
if ax2.legend_:
    ax2.legend_.remove()

# ---------------- Semantic ----------------
ax3 = sns.barplot(
    data=df_clean_dict["semantic_multi"],
    x="grade",
    y="semantic_multi",
    hue="roi",
    palette=roi_color_map,
    ax=axes[2]
)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
axes[2].tick_params(axis="both", which="major", length=3, width=1)
axes[2].set_ylim(ylim)
axes[2].set_title("Semantic", fontsize=18, fontweight="bold")
axes[2].set_xlabel("Grade", fontsize=18)
axes[2].set_ylabel(r"Multimodal similarity ($r$)", fontsize=18)
# remove legend
if ax3.legend_:
    ax3.legend_.remove()


for ax in axes:
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.axhline(y=0, linestyle='--', linewidth=1, color='black', alpha=0.7)
    ax.grid(True, axis='y', color='gray', alpha=0.3)
plt.tight_layout()

# save the figure
roi_path = FIG_DIR / 'multimodal'
roi_path.mkdir(exist_ok=True, parents=True)
path     = roi_path / "Correlations_gradewise.pdf"
plt.savefig(
    path,
    format      = 'pdf',
    dpi         = 300,
    transparent = True,
    bbox_inches = 'tight',
    pad_inches  = 0.3
)
print(f"\nSuccessful: Figure is saved ")
plt.show()


    
# %%
# one sample t-test
results = []
for cond in conditions:
    data=df_clean_dict[cond]
    

    for roi, g in data.groupby("roi"):

        t_stat, p_val = stats.ttest_1samp(g["Fisher"], 0)

        results.append({
            "condition": cond,
            "roi": roi,
            "t": t_stat,
            "p": p_val,
            "n": len(g)
        })
print(results)

all_results = []

for cond in conditions:
    data = df_clean_dict[cond]

    results = []  # ← per-condition list of dicts

    for roi, g in data.groupby("roi"):
        x = g["Fisher"].dropna()

        if len(x) < 2 or x.nunique() < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat, p_val = stats.ttest_1samp(x, 0)

        results.append({
            "condition": cond,
            "roi": roi,
            "t": t_stat,
            "p": p_val,
            "n": len(x)
        })

    # convert to DataFrame
    df_cond = pd.DataFrame(results)

    # --- FDR correction per condition ---
    valid_mask = df_cond["p"].notna()
    pvals = df_cond.loc[valid_mask, "p"].values

    reject, p_fdr, _, _ = multipletests(
        pvals,
        alpha=0.05,
        method="fdr_bh"
    )

    df_cond["p_fdr"] = np.nan
    df_cond["significant_fdr"] = False

    df_cond.loc[valid_mask, "p_fdr"] = p_fdr
    df_cond.loc[valid_mask, "significant_fdr"] = reject

    # store DataFrame (NOT dicts)
    all_results.append(df_cond)


df_all = pd.concat(all_results, ignore_index=True)

print(df_all)
# %%
# apply linear regression 


all_results_ml = []
for cond in conditions:
    data=df_clean_dict[cond]

    results = []
    for roi, g in data.groupby("roi"):

        model = smf.ols("Fisher ~ grade", data=g).fit()

        results.append({
            "condition": cond,
            "roi": roi,
            "beta_grade": model.params.get("grade", np.nan),
            "t": model.tvalues.get("grade", np.nan),
            "p": model.pvalues.get("grade", np.nan),
            "r2": model.rsquared,
            "n": len(g)
        })

    df_lm = pd.DataFrame(results)
    # extract p-values
    pvals = df_lm["p"].values
    # FDR correction (Benjamini–Hochberg)
    reject, p_fdr, _, _ = multipletests(
        pvals,
        alpha=0.05,
        method="fdr_bh"
    )
    # attach to dataframe
    df_lm["p_fdr"] = p_fdr
    df_lm["significant_fdr"] = reject

    all_results_ml.append(df_lm)

# --- combine all conditions ---
df_lm_all = pd.concat(all_results_ml, ignore_index=True)
print(df_lm_all)




# %%

# apply linear regression on RT


all_results_ml = []
for cond in conditions:
    data=df_clean_dict[cond]

    results = []
    for roi, g in data.groupby("roi"):

        model = smf.ols("Fisher ~ rt", data=g).fit()

        results.append({
            "condition": cond,
            "roi": roi,
            "beta_rt": model.params.get("rt", np.nan),
            "t": model.tvalues.get("rt", np.nan),
            "p": model.pvalues.get("rt", np.nan),
            "r2": model.rsquared,
            "n": len(g)
        })

    df_lm = pd.DataFrame(results)
    # extract p-values
    pvals = df_lm["p"].values
    # FDR correction (Benjamini–Hochberg)
    reject, p_fdr, _, _ = multipletests(
        pvals,
        alpha=0.05,
        method="fdr_bh"
    )
    # attach to dataframe
    df_lm["p_fdr"] = p_fdr
    df_lm["significant_fdr"] = reject

    all_results_ml.append(df_lm)

# --- combine all conditions ---
df_lm_all = pd.concat(all_results_ml, ignore_index=True)
print(df_lm_all)
# %%
