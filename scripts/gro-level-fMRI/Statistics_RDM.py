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


# --- remove outliers ---
def get_removed_outliers(df, group_col, value_col, k=1.5): 
    def _mask(group): 
        q1    = group[value_col].quantile(0.25) 
        q3    = group[value_col].quantile(0.75) 
        iqr   = q3 - q1 
        lower = q1 - k * iqr 
        upper = q3 + k * iqr 
        return (group[value_col] < lower) | (group[value_col] > upper) 
    mask      = df.groupby(group_col, group_keys=False).apply(_mask) 
    return df[mask]

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

""" removed_dict  = {}

# --- remove it for each condition ---
for cond in conditions:
    df_cond             = df[['roi', 'grade', 'number_run', 'fd_mean', cond]].copy()
    removed             = get_removed_outliers(df_cond, 'roi', cond)
    df_clean            = df_cond.drop(removed.index).copy()
    df_clean_dict[cond] = df_clean
    removed_dict[cond]  = removed
    # --- print out ---
    print(f"\nRemoved rows for {cond}:")
    print(removed) """


# %%
# 4. === STEP 4 ===: Two-way mixed ANOVA
# -----------------------------------------------
tar_cond="semantic_multi"
aov = pg.mixed_anova(
    data=df,
    dv=tar_cond,
    between="grade",
    within="roi",
    subject="subject"
)

print(aov)


# %%
# 5. === STEP 5 ===: Post-hoc pairwise comparisons for roi effect
# ------------------------------------------------
posthoc_roi = pg.pairwise_tests(
    data=df,
    dv=tar_cond,
    within="roi",
    parametric=True,
    padjust="fdr_bh",
    effsize="hedges",
    subject="subject"
)

# Show only significant comparisons
sig_roi = posthoc_roi.loc[
    posthoc_roi["p-corr"] < 0.05,
    ["A", "B", "p-unc", "p-corr", "T", "dof","hedges"]
]
sig_roi

# %%
# ------------------------------------------------
# Pairwise comparisons between school grades within each ROI

posthoc_list = []

for roi in df["roi"].unique():

    df_roi = df[df["roi"] == roi].copy()

    # Pairwise comparisons between grades
    ph = pg.pairwise_tests(
        data=df_roi,
        dv=tar_cond,
        between="grade",
        parametric=True,  
        padjust=None,       # correction applied across all ROIs below
        effsize="hedges",
        correction='auto' # welch's tests
    )

    ph["roi"] = roi
    posthoc_list.append(ph)


# Combine all ROIs
posthoc = pd.concat(posthoc_list, ignore_index=True)

# %
# 6. === STEP 6 ===: FDR correction across all 36 comparisons
# ------------------------------------------------------------

from statsmodels.stats.multitest import multipletests

reject, p_corr, _, _ = multipletests(
    posthoc["p-unc"],
    alpha=0.05,
    method="fdr_bh"
)

posthoc["p-corr"] = p_corr
posthoc["significant"] = reject

# --- Labels ---
csv_path = OUT_DIR / 'multimodal'
csv_name = f"{HEMI}_pairwisecomp_{tar_cond}.csv"
# Save pairwise comparison results
posthoc.to_csv(csv_path / csv_name, index=False)
posthoc



# %%
# 4. === STEP 4 ===: Plot pairwisecomparison results
# ------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

# --- Labels ---
fig_path = FIG_DIR / 'multimodal'
fig_name = f"{HEMI}_{MODAL}_Word_Correlations_byGrade_{tar_cond}.pdf"

order   = list(roi_color_map.keys())
xlabels = [name for name in roi_labels.values()]


df_cond = df_clean_dict[tar_cond].copy()

# Grade order
df_cond["grade"] = pd.Categorical(
    df_cond["grade"],
    categories=[1, 2, 4],
    ordered=True
)

# Numeric x-position for each ROI
roi_to_x = {roi: i for i, roi in enumerate(order)}

# Horizontal positions of grades
grade_offset = {
    1: -0.22,
    2:  0.00,
    4:  0.22
}

# Colors
grade_colors = {
    1: "darkgoldenrod",
    2: "darkcyan",
    4: "firebrick"
}


# ============================================================
# Plot boxplots + individual observations
# ============================================================

for roi in order:

    for grade in [1, 2, 4]:

        data = df_cond[
            (df_cond["roi"] == roi) &
            (df_cond["grade"] == grade)
        ]

        if len(data) == 0:
            continue

        values = data[tar_cond].dropna().values

        if len(values) == 0:
            continue

        x = roi_to_x[roi] + grade_offset[grade]

        # --- Box plot ---
        bp = ax.boxplot(
            values,
            positions=[x],
            widths=0.16,
            patch_artist=True,
            showfliers=False,

            medianprops=dict(
                color="black",
                linewidth=1.2
            ),

            whiskerprops=dict(
                color=grade_colors[grade],
                linewidth=1
            ),

            capprops=dict(
                color=grade_colors[grade],
                linewidth=1
            ),

            boxprops=dict(
                edgecolor=grade_colors[grade],
                linewidth=1
            )
        )

        # Fill box
        for box in bp["boxes"]:
            box.set_facecolor(grade_colors[grade])
            box.set_alpha(0.6)

        # --- Individual observations ---
        jitter = np.random.uniform(
            -0.04,
            0.04,
            size=len(values)
        )

        ax.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=18,
            alpha=0.6,
            color=grade_colors[grade],
            edgecolors="none"
        )


# ============================================================
# Add significant grade comparisons
# ============================================================

def significance_stars(p):

    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return None


# Three possible comparisons
comparison_levels = {
    (1, 2): 0,
    (1, 4): 1,
    (2, 4): 2
}


for roi in order:

    # Post-hoc results for this ROI
    roi_results = posthoc[
        posthoc["roi"] == roi
    ]

    # Data for determining vertical position
    roi_data = df_cond[
        df_cond["roi"] == roi
    ][tar_cond].dropna()

    if len(roi_data) == 0:
        continue

    # Starting height for significance brackets
    y_base = roi_data.max() + 0.00

    for _, row in roi_results.iterrows():

        A = int(row["A"])
        B = int(row["B"])
        p = row["p-corr"]

        stars = significance_stars(p)

        # Only show significant comparisons
        if stars is None:
            continue

        # x positions of the two grades
        x1 = roi_to_x[roi] + grade_offset[A]
        x2 = roi_to_x[roi] + grade_offset[B]

        # Vertical level of this comparison
        level = comparison_levels[
            tuple(sorted([A, B]))
        ]

        y = y_base + level * 0.08
        h = 0.005

        # Bracket
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            lw=1,
            color="black"
        )

        # Stars
        ax.text(
            (x1 + x2) / 2,
            y + h,
            stars,
            ha="center",
            va="bottom",
            fontsize=10
        )


# ============================================================
# Aesthetics
# ============================================================

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim(
    -0.6,
    len(order) - 0.4
)

ax.set_ylim(
    -0.4,
    1.4
)

ax.set_xticks(range(len(order)))

ax.set_xticklabels(
    xlabels,
    fontsize=14,
    rotation=60,
    ha='right'
)

ax.yaxis.grid(
    True,
    which='major',
    linestyle='-',
    linewidth=0.5,
    alpha=0.5
)

ax.axhline(
    y=0,
    linestyle='--',
    linewidth=1,
    color='black',
    alpha=0.7
)

ax.set_xlabel("")

ax.set_ylabel(
    r"Similarity (Fisher's $z$)",
    fontsize=15
)

if tar_cond=="word_spoken":
    title="Spoken"
elif tar_cond=="word_written":
    title="Written"
elif tar_cond=="word_multi":
    title="Spoken and Written"
ax.set_title(
    title,
    fontsize=15
)


# ============================================================
# Legend
# ============================================================

from matplotlib.patches import Patch

legend_handles = [
    Patch(
        facecolor=grade_colors[1],
        alpha=0.6,
        label="Grade 1"
    ),
    Patch(
        facecolor=grade_colors[2],
        alpha=0.6,
        label="Grade 2"
    ),
    Patch(
        facecolor=grade_colors[4],
        alpha=0.6,
        label="Grade 4"
    )
]

ax.legend(
    handles=legend_handles,
    title="School",
    fontsize=11,
    title_fontsize=12,
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(0.2, 1.1)
)


# ============================================================
# Save and display
# ============================================================

plt.tight_layout()

fig_path.mkdir(
    exist_ok=True,
    parents=True
)

plt.savefig(
    fig_path / fig_name,
    format='pdf',
    dpi=300,
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.3
)

print("Successful: Figure is saved")

plt.show()





# %%
# 4. === STEP 4 ===: multiple linear regression models 
# to test increase or decrease trend?
# Similarity = B0 + B1*grade + B2+Motion + B3+rt_mean  B4*Sex + error 
# ----------------------------------------------------
# regression model
dependent="grade"


all_results_lr = []
all_clean_data = []

for cond in conditions:

    data = df_clean_dict[cond]
    rows = []

    for roi, g in data.groupby("roi"):

        # g['r'] = np.tanh(g[cond])
        # descriptive statistics by grade
        grade_stats = (
            g.groupby("grade")[cond]
            .agg(mean="mean", sd="std")
            .reset_index()
        )

        # columns become:
        # mean_g1, sd_g1, mean_g2, sd_g2, ...
        grade_dict = {}

        for grade, vals in grade_stats.iterrows():
            grade_dict[f"mean_g{grade}"] = vals["mean"]
            grade_dict[f"sd_g{grade}"]   = vals["sd"]
        model = smf.ols(f"{cond} ~ {dependent} + fd_mean + rt_mean + sex", data=g).fit()

        # Influence diagnostics
        influence = model.get_influence()
        # Cook's distance
        cooks_d = influence.cooks_distance[0]
        # Sample size
        # Sample size used by the model
        n = int(model.nobs)
        # Cook's distance threshold
        cook_threshold = 4 / n
        high_influence = cooks_d > cook_threshold
        # Get the original dataframe indices used by the model
        model_indices = model.model.data.row_labels
        # Indices of high-influence observations
        remove_indices = model_indices[high_influence]
        # Remove them
        g_clean = g.drop(index=remove_indices).copy()
        print(f"n = {n}")
        print(f"Cook's D threshold = {cook_threshold:.3f}")
        print(f"Removed {high_influence.sum()} observations")
        print(f"Remaining observations = {len(g_clean)}")

        # ------------------------------------------------------------
        # 5. Add cleaned data to list
        # ------------------------------------------------------------
        all_clean_data.append(g_clean)

        model = smf.ols(f"{cond} ~ {dependent} + fd_mean + rt_mean + sex", data=g_clean).fit()
        beta = model.params.get(dependent, np.nan)
        tval = model.tvalues.get(dependent, np.nan)
        pval = model.pvalues.get(dependent, np.nan)
        se   = model.bse.get(dependent, np.nan)          
        resi = model.df_resid                       
        r2   = model.rsquared
        n    = len(g_clean)

        # combine everything
        row = {
            "condition": cond,
            "roi": roi,
            "beta": beta,
            "se": se,
            "t": tval,
            "residual": resi,
            "p": pval,
            "r2": r2,
            "n": n,
            "predictor": dependent,
            **grade_dict
        }

        rows.append(row)
    df_lm = pd.DataFrame(rows)

    # --- FDR correction ---
    df_lm["p_fdr"] = np.nan
    df_lm["significant_fdr"] = False

    valid = df_lm["p"].notna()

    if valid.any():
        reject, p_fdr, _, _ = multipletests(
            df_lm.loc[valid, "p"],
            alpha=0.05,
            method="fdr_bh"
        )

        df_lm.loc[valid, "p_fdr"] = p_fdr
        df_lm.loc[valid, "significant_fdr"] = reject

    all_results_lr.append(df_lm)

df_lm_all = pd.concat(all_results_lr, ignore_index=True)
df_clean_all = pd.concat(
    all_clean_data,
    ignore_index=True
)

# save it as csv file
path      = OUT_DIR / 'multimodal' / f'{HEMI}_FWHM_{int(FWHM_SMOOTHING)}_linear-model_{dependent}.csv'
df_lm_all.to_csv(path, index=False)
print(f"Data is saved in the path: {path}")
df_lm_all[
    (df_lm_all['significant_fdr'] == True)
]




































# %%
# 4. === STEP 4 ===: multiple linear regression models 
# to test increase or decrease trend?
# Similarity = B0 + B1*coactivation + B2+Motion + B3+rt_mean  B4*Sex + error 
# ----------------------------------------------------
# regression model
dependent="mean_overlap_log"
# condition="mean_overlap"

all_results_lr = []
all_clean_data = []

for cond in conditions:

    data = df_clean_dict[cond]
    # Keep only ROIs with significant pairwise grade differences
    # data = data[data["roi"].isin(sig_rois)].copy()
    rows = []

    for roi, g in data.groupby("roi"):

        # g['r'] = np.tanh(g[cond])
        # descriptive statistics by grade
        grade_stats = (
            g.groupby("grade")[cond]
            .agg(mean="mean", sd="std")
            .reset_index()
        )

        # columns become:
        # mean_g1, sd_g1, mean_g2, sd_g2, ...
        grade_dict = {}

        for grade, vals in grade_stats.iterrows():
            grade_dict[f"mean_g{grade}"] = vals["mean"]
            grade_dict[f"sd_g{grade}"]   = vals["sd"]
        model = smf.ols(f"{cond} ~ {dependent} + fd_mean + rt_mean + sex", data=g).fit()

        # Influence diagnostics
        influence = model.get_influence()
        # Cook's distance
        cooks_d = influence.cooks_distance[0]
        # Sample size
        # Sample size used by the model
        n = int(model.nobs)
        # Cook's distance threshold
        cook_threshold = 4 / n
        high_influence = cooks_d > cook_threshold
        # Get the original dataframe indices used by the model
        model_indices = model.model.data.row_labels
        # Indices of high-influence observations
        remove_indices = model_indices[high_influence]
        # Remove them
        g_clean = g.drop(index=remove_indices).copy()
        print(f"n = {n}")
        print(f"Cook's D threshold = {cook_threshold:.3f}")
        print(f"Removed {high_influence.sum()} observations")
        print(f"Remaining observations = {len(g_clean)}")

        # ------------------------------------------------------------
        # 5. Add cleaned data to list
        # ------------------------------------------------------------
        all_clean_data.append(g_clean)

        model = smf.ols(f"{cond} ~ {dependent} + fd_mean + rt_mean + sex", data=g_clean).fit()
        beta = model.params.get(dependent, np.nan)
        tval = model.tvalues.get(dependent, np.nan)
        pval = model.pvalues.get(dependent, np.nan)
        se   = model.bse.get(dependent, np.nan)          
        resi = model.df_resid                       
        r2   = model.rsquared
        n    = len(g_clean)

        # combine everything
        row = {
            "condition": cond,
            "roi": roi,
            "beta": beta,
            "se": se,
            "t": tval,
            "residual": resi,
            "p": pval,
            "r2": r2,
            "n": n,
            "predictor": dependent,
            **grade_dict
        }

        rows.append(row)
    df_lm = pd.DataFrame(rows)

    # --- FDR correction ---
    df_lm["p_fdr"] = np.nan
    df_lm["significant_fdr"] = False

    valid = df_lm["p"].notna()

    if valid.any():
        reject, p_fdr, _, _ = multipletests(
            df_lm.loc[valid, "p"],
            alpha=0.05,
            method="fdr_bh"
        )

        df_lm.loc[valid, "p_fdr"] = p_fdr
        df_lm.loc[valid, "significant_fdr"] = reject

    all_results_lr.append(df_lm)

df_lm_all = pd.concat(all_results_lr, ignore_index=True)
df_clean_all = pd.concat(
    all_clean_data,
    ignore_index=True
)

# save it as csv file
path      = OUT_DIR / 'multimodal' / f'{HEMI}_FWHM_{int(FWHM_SMOOTHING)}_linear-model_{dependent}.csv'
df_lm_all.to_csv(path, index=False)
print(f"Data is saved in the path: {path}")
df_lm_all[
    (df_lm_all['significant_fdr'] == True)
]



# %%
# ============================================================
# Plot pSTG: similarity and coactivation
# ============================================================
""" df_target = df[
    df["roi"]=="Superior Temporal Gyrus, posterior division"
].copy() """

g = sns.jointplot(
    data=df,
    x=dependent,
    y="word_multi",
    hue="roi",
    palette=roi_color_map,
    height=6,
    ratio=4,
    joint_kws={
        "s": 70,
        "alpha": 0.7,
        "edgecolor": "black",
        "linewidth": 0.5
    },
    marginal_kws={
        "fill": False,
        "linewidth": 1.5
    }
)


# ============================================================
# Axes
# ============================================================

# Y-axis
g.ax_joint.set_xlim(-0.5, 7)
g.ax_joint.set_ylim(-0.7, 1.2)


# ============================================================
# Axis labels
# ============================================================

g.set_axis_labels(
    r"Univariate coactivation (log scale)",
    r"Similarity (Fisher's $z$)",
    fontsize=14
)


# ============================================================
# Legend
# ============================================================

legend = g.ax_joint.legend(
    ncols=4,
    frameon=True,
    loc="lower left",
    bbox_to_anchor=(0, 0)
)
for text in legend.get_texts():
    old_label = text.get_text()
    text.set_text(roi_labels.get(old_label, old_label))
    text.set_fontsize(9)
legend.set_title("")
legend.get_frame().set_facecolor("white")
legend.get_frame().set_alpha(1.0)
# ============================================================
# Save figure
# ============================================================

fig_path = FIG_DIR / "multimodal"
fig_path.mkdir(parents=True, exist_ok=True)

fig_name = f"{HEMI}_similarity-coactivation.pdf"

g.figure.savefig(
    fig_path / fig_name,
    format="pdf",
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.3
)

# ============================================================
# Display
# ============================================================

plt.show()

























































# %%
