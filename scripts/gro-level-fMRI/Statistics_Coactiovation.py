# %% [markdown]
# ## fMRI Group-level Statistics RDM (2nd-level)
#
# **Pipeline Overview**
# 1. === STEP 1 ===: Install packages
# 2. === STEP 2 ===: Set parameters
# 3. === STEP 3 ===: read RDM csv file and remove outliers
# 4. === STEP 4 ===: Distirbution of RDM metrcis 
# 5. === STEP 5 ===: Distirbution of RDM metrcis by grade
# 6. === STEP 6 ===: Statistics: One-sample t-tetst
# 7. === STEP 7 ===: Statistics: Linear regression



# %%
# 1. === STEP 1 ===: Install packages
# -----------------------------------------------

# install necessary packages
import sys
from pathlib import Path
# Specify path
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
BIDS_DIR     = Path('/work_beegfs/suknp132/SLANG-CROSS-conversion')
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
from scipy.stats import zscore
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
FWHM_SMOOTHING = 5.0 
HEMI           = 'left'
MODAL          = 'multi' # multi, written, spoken 
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
# 3. === STEP 3 ===: read RDM csv file 
# -----------------------------------------------
# --- read RDM ---
path     = OUT_DIR / 'multimodal' / f'{HEMI}_RDM_metrics_FWHM_{int(FWHM_SMOOTHING)}.csv'
df_csv   = pd.read_csv(path)
# --- read Coactivation ---
coactive_path = OUT_DIR / 'multimodal' / f'{HEMI}_Coactivation_metrics_FWHM_8.csv'
df_coactive   = pd.read_csv(coactive_path)
df_coactive = df_coactive[["subject", "mean_overlap", "mean_written_voxels", "mean_spoken_voxels", "roi"]]
df_coactive["mean_overlap_log"] = (
    df_coactive.groupby("roi")["mean_overlap"]
    .transform(np.log1p)
)

# --- read demographics ---
demo_path = BIDS_DIR / "participants.tsv" 
df_demo = pd.read_csv(demo_path, sep="\t")
df_demo = df_demo.rename(
    columns={"participant_id": "subject"}
)
# Keep only required columns
df_demo = df_demo[["subject", "age", "sex"]]
# --- read Head motion ---
# Subjects
subjects      = sorted(DERIV_DIR.glob(f"sub-*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]
# mriqc_path
mriqc_path = BIDS_DIR / "derivatives" / "mriqc"

df_list = []
for subject in subjects:
    path = subject / "behavior" / "accuracy_summary.csv"
    if not path.exists():
        print(f"Missing: {path}")
        continue
    df = pd.read_csv(path)
    # Remove rows with invalid accuracy
    df = df[df["accuracy_all"] != -1].copy()
    # Extract participant ID:
    df["subject"] = (
        df["file_name"]
        .str.extract(r"sub-(\d+)")[0]
        .astype(int)
    )
    f_name = df["file_name"].str.replace("_events.tsv", "", regex=False)
    participant_id = subject.name
    fd_path = mriqc_path / participant_id / "ses-01" / participant_id / "ses-01" / "func"
    fd_means = []
    for name in f_name:
        file = fd_path / f"{name}_timeseries.tsv"
        # Read the TSV
        confounds = pd.read_csv(file, sep="\t")
        # Mean framewise displacement
        mean_fd = confounds["framewise_displacement"].mean()
        fd_means.append(mean_fd)
    # Add participant/run-level FD to your dataframe
    df["fd"] = fd_means
    df_list.append(df)
# Concatenate all subjects
df_all = pd.concat(df_list, ignore_index=True)
# Average FD for each subject
fd_subject = (
    df_all.groupby("subject")[["fd", "responses_all", "accuracy_all", "RT_all"]]
    .mean()
    .reset_index()
    .rename(columns={
        "fd": "fd_mean",
        "responses_all": "response_mean",
        "accuracy_all": "accuracy_mean",
        "RT_all": "rt_mean",
    })
)
# Convert subject to the same format as the ROI dataset
fd_subject["subject"] = "sub-" + fd_subject["subject"].astype(str)

print(fd_subject)

# Merge fd_mean into the ROI dataset
df = df_csv.merge(
    fd_subject,
    on="subject",
    how="left"
).merge(
    df_demo,
    on="subject",
    how="left"
).merge(
    df_coactive,
    on=["subject", "roi"],
    how="left"
)

print(df)

conditions = [
    f'word_{MODAL}',
    f'pseudo_{MODAL}',
    f'semantic_{MODAL}',
    'word_written',
    'word_spoken',
]

df_clean_dict = {}
for cond in conditions:
    df_cond             = df[['subject', 'roi', 'grade', 'sex', 'number_run', 'fd_mean', 'response_mean', 'accuracy_mean', 'rt_mean','mean_overlap',"mean_overlap_log", cond]].copy()
    df_clean_dict[cond] = df_cond


# %%
# -------------------------------------------------------------------------
# 各ROIごとに word_multi と mean_overlap の Spearman 相関
# 学年は分けない
# -------------------------------------------------------------------------

results = []

for roi, df_roi in df.groupby('roi'):

    # 欠測値を除外
    tmp = df_roi[['word_multi', 'mean_overlap']].dropna()

    # Spearman correlation
    corr = pg.corr(
        x=tmp['word_multi'],
        y=tmp['mean_overlap'],
        method='spearman',
        alternative='two-sided'
    )

    results.append({
        'roi': roi,
        'n': corr['n'].iloc[0],
        'r': corr['r'].iloc[0],
        'p-unc': corr['p-val'].iloc[0]
    })

# DataFrame化
result_df = pd.DataFrame(results)

# -------------------------------------------------------------------------
# 12 ROIについて Benjamini-Hochberg FDR補正
# -------------------------------------------------------------------------

reject, p_corr, _, _ = multipletests(
    result_df['p-unc'],
    method='fdr_bh'
)

result_df['p-corr'] = p_corr
result_df['significant'] = reject

# 相関係数の大きい順
result_df = result_df.sort_values('r', ascending=False).reset_index(drop=True)

result_df



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# -------------------------------------------------------------------------
# Calculate Spearman correlation for each ROI
# -------------------------------------------------------------------------

results = []
roi_data = {}

for roi, df_roi in df.groupby('roi'):

    tmp = df_roi[['word_multi', 'mean_overlap']].dropna()

    r, p = spearmanr(
        tmp['word_multi'],
        tmp['mean_overlap']
    )

    results.append({
        'roi': roi,
        'n': len(tmp),
        'r': r,
        'p-unc': p
    })

    roi_data[roi] = tmp

# Results dataframe
result_df = pd.DataFrame(results)

# FDR correction across ROIs
result_df['p-corr'] = multipletests(
    result_df['p-unc'],
    method='fdr_bh'
)[1]

result_df['significant'] = result_df['p-corr'] < 0.05

# Sort by correlation
result_df = result_df.sort_values('r', ascending=False).reset_index(drop=True)

print(result_df)

# %%
# -------------------------------------------------------------------------
# Plot: one scatter plot for each ROI
# -------------------------------------------------------------------------
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
# ROI order is determined by roi_color_map
roi_order = list(roi_color_map.keys())

n_rois = len(roi_order)

ncols = 4
nrows = int(np.ceil(n_rois / ncols))

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(13, 3 * nrows)
)

axes = np.asarray(axes).flatten()


# ============================================================
# Determine common x- and y-axis limits across all ROIs
# ============================================================

all_x = pd.concat(
    [roi_data[roi]['word_multi'] for roi in roi_order]
).dropna()

all_y = pd.concat(
    [roi_data[roi]['mean_overlap'] for roi in roi_order]
).dropna()

# Common limits
x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()

# Add 5% padding
x_pad = (x_max - x_min) * 0.05
y_pad = (y_max - y_min) * 0.05

xlim = (x_min - x_pad, x_max + x_pad)
ylim = (y_min - y_pad, y_max + y_pad)


# ============================================================
# Plot
# ============================================================

for i, roi in enumerate(roi_order):

    ax = axes[i]

    tmp = roi_data[roi]
    roi_color = roi_color_map[roi]
    label = roi_labels[roi]

    # Scatter plot
    sns.scatterplot(
        data=tmp,
        x='word_multi',
        y='mean_overlap',
        ax=ax,
        s=55,
        alpha=0.75,
        color=roi_color
    )

    # Regression line
    sns.regplot(
        data=tmp,
        x='word_multi',
        y='mean_overlap',
        scatter=False,
        ax=ax,
        ci=95,
        color=roi_color,
        line_kws={'linewidth': 1.5}
    )

    # Get statistics
    row = result_df[result_df['roi'] == roi].iloc[0]

    r = row['r']
    p_corr = row['p-corr']
    n = row['n']

    # Significance marker
    if p_corr < 0.05:
        sig = '*'
    else:
        sig = ''

    # ROI label
    ax.set_title(
        f"{label}\n",
        fontsize=18
    )

    # Statistics
    ax.text(
        0.02, 1.25,
        f"\nSpearman $r$ = {r:.2f}{sig}",
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment='top',
        horizontalalignment='left'
    )

    ax.set_xlabel(
        r"Similarity (Fisher's $z$)",
        fontsize=14
    )

    ax.set_ylabel(
        'Coactivation (voxels)',
        fontsize=14
    )

    # IMPORTANT: identical scales for every ROI
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.grid(alpha=0.2)


# Remove unused panels
for j in range(n_rois, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(h_pad=2)
plt.savefig(
    FIG_DIR / "multimodal" / "spearman_coactivation.pdf",
    format="pdf",
    bbox_inches="tight",
)

plt.show()






# %%
n_rois = len(roi_data)

ncols = 3
nrows = int(np.ceil(n_rois / ncols))

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(15, 4.5 * nrows)
)

axes = np.asarray(axes).flatten()

for i, roi in enumerate(result_df['roi']):

    ax = axes[i]

    tmp = roi_data[roi]

    # Scatter plot
    sns.scatterplot(
        data=tmp,
        x='word_multi',
        y='mean_overlap',
        ax=ax,
        s=55,
        alpha=0.75
    )

    # Optional linear regression line for visualization only
    sns.regplot(
        data=tmp,
        x='word_multi',
        y='mean_overlap',
        scatter=False,
        ax=ax,
        ci=95
    )

    # Get statistics
    row = result_df[result_df['roi'] == roi].iloc[0]

    r = row['r']
    p_unc = row['p-unc']
    p_corr = row['p-corr']
    n = row['n']

    # Significance marker
    if p_corr < 0.001:
        sig = '***'
    elif p_corr < 0.01:
        sig = '**'
    elif p_corr < 0.05:
        sig = '*'
    else:
        sig = 'n.s.'

    # ROI title
    ax.set_title(
        f"{roi}\n"
        f"Spearman ρ = {r:.2f}, FDR p = {p_corr:.3f} {sig}, n = {n}",
        fontsize=10
    )

    ax.set_xlabel('Cross-modal RSA (word_multi)')
    ax.set_ylabel('Mean overlap')

    ax.grid(alpha=0.2)

# Remove unused panels
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

plt.show()












# %%
