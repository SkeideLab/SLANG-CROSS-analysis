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
# 3. === STEP 3 ===: read RDM csv file and remove outliers
# -----------------------------------------------
# --- read RDM ---
path     = OUT_DIR / 'multimodal' / f'{HEMI}_RDM_metrics_FWHM_{int(FWHM_SMOOTHING)}.csv'
df_csv   = pd.read_csv(path)
# --- read Coactivation ---
coactive_path = OUT_DIR / 'multimodal' / f'{HEMI}_Coactivation_metrics_FWHM_{int(FWHM_SMOOTHING)}.csv'
df_coactive   = pd.read_csv(coactive_path)
df_coactive = df_coactive[["subject", "mean_overlap", "roi"]]
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


conditions = [
    'word_spoken',
    'word_written',
    f'word_{MODAL}',
]



# ----------------------------------------------------
# %%
# 4. === STEP 4 ===: Multiple linear regression models
# Test developmental trend while controlling for motion, RT, and sex
#
# Similarity = B0 + B1*grade + B2*FD + B3*RT + B4*Sex + error
# ------------------------------------------------------------

target="Temporal Occipital Fusiform Cortex" # "Superior Temporal Gyrus, posterior division" "Temporal Occipital Fusiform Cortex"

df_clean_dict = {}
for cond in conditions:
    df_cond = df[
        df["roi"].str.contains(target, case=True, na=False)
    ][
        [
            "subject", "roi", "grade", "sex", "number_run",
            "fd_mean", "response_mean", "accuracy_mean", "rt_mean",
            "mean_overlap", "mean_overlap_log", cond
        ]
    ].copy()

    df_clean_dict[cond] = df_cond



dependent = "grade"

all_results_lr = []
all_clean_data = []

for cond in conditions:

    # --------------------------------------------------------
    # Data for this condition
    # --------------------------------------------------------
    g = df_clean_dict[cond].copy()

    # --------------------------------------------------------
    # Descriptive statistics by grade
    # --------------------------------------------------------
    grade_stats = (
        g.groupby("grade")[cond]
        .agg(mean="mean", sd="std")
        .reset_index()
    )

    grade_dict = {}

    for _, vals in grade_stats.iterrows():
        grade = vals["grade"]
        grade_dict[f"mean_g{grade}"] = vals["mean"]
        grade_dict[f"sd_g{grade}"] = vals["sd"]

    # --------------------------------------------------------
    # Initial linear model
    # --------------------------------------------------------
    model = smf.ols(
        f"{cond} ~ {dependent} + fd_mean + rt_mean + sex",
        data=g
    ).fit()

    # --------------------------------------------------------
    # Influence diagnostics
    # --------------------------------------------------------
    influence = model.get_influence()

    # Cook's distance
    cooks_d = influence.cooks_distance[0]

    # Sample size used by the model
    n = int(model.nobs)

    # Cook's distance threshold
    cook_threshold = 4 / n

    high_influence = cooks_d > cook_threshold

    # Get indices used by the model
    model_indices = model.model.data.row_labels

    # Indices of observations to remove
    remove_indices = model_indices[high_influence]

    # Remove influential observations
    g_clean = g.drop(index=remove_indices).copy()

    print(f"\nCondition: {cond}")
    print(f"n = {n}")
    print(f"Cook's D threshold = {cook_threshold:.3f}")
    print(f"Removed {high_influence.sum()} observations")
    print(f"Remaining observations = {len(g_clean)}")

    # --------------------------------------------------------
    # Save cleaned data
    # --------------------------------------------------------
    all_clean_data.append(g_clean)

    # --------------------------------------------------------
    # Final regression after removing influential observations
    # --------------------------------------------------------
    model = smf.ols(
        f"{cond} ~ {dependent} + fd_mean + rt_mean + sex",
        data=g_clean
    ).fit()

    # --------------------------------------------------------
    # Extract statistics for grade
    # --------------------------------------------------------
    beta = model.params.get(dependent, np.nan)
    se = model.bse.get(dependent, np.nan)
    tval = model.tvalues.get(dependent, np.nan)
    pval = model.pvalues.get(dependent, np.nan)

    residual = model.df_resid
    r2 = model.rsquared
    n = int(model.nobs)

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------
    row = {
        "condition": cond,
        "roi": target,
        "beta": beta,
        "se": se,
        "t": tval,
        "residual": residual,
        "p": pval,
        "r2": r2,
        "n": n,
        "predictor": dependent,
        **grade_dict
    }

    all_results_lr.append(row)

# %
# ------------------------------------------------------------
# Combine regression results
# ------------------------------------------------------------

df_lm_all = pd.DataFrame(all_results_lr)


# ------------------------------------------------------------
# Combine cleaned data
# ------------------------------------------------------------

df_clean_all = pd.concat(
    all_clean_data,
    ignore_index=True
)




# %%
# ============================================================
# Plot developmental trajectories with violin distributions
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

dependent = "grade"

condition_colors = {
    "word_written": "tab:blue",
    "word_spoken": "tab:orange",
    "word_multi": "green",
}

# Small horizontal offsets so the three conditions are visible
condition_offsets = {
    "word_spoken": -0.20,
    "word_written": 0.00,
    "word_multi": 0.20,
}

grades = [1, 2, 4]


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 5))


# ============================================================
# Loop over conditions
# ============================================================

for cond in conditions:

    # --------------------------------------------------------
    # Sample for this condition
    # --------------------------------------------------------
    sample = df_clean_all[
        df_clean_all[cond].notna()
    ].copy()

    sample["sex"] = sample["sex"].astype("category")

    color = condition_colors[cond]
    offset = condition_offsets[cond]



    # ========================================================
    # Multiple linear regression
    # ========================================================

    linear_model = smf.ols(
        f"{cond} ~ {dependent} + fd_mean + rt_mean + sex",
        data=sample
    ).fit()

    beta = linear_model.params[dependent]
    p_value = linear_model.pvalues[dependent]
    sig = "*" if p_value < 0.05 else ""


    # --------------------------------------------------------
    # Prediction grid
    # --------------------------------------------------------

    x_pred = np.linspace(
        sample[dependent].min(),
        sample[dependent].max(),
        200
    )

    prediction_grid = pd.DataFrame({
        dependent: x_pred,
        "fd_mean": sample["fd_mean"].mean(),
        "rt_mean": sample["rt_mean"].mean(),
        "sex": sample["sex"].mode()[0]
    })

    prediction_grid["sex"] = pd.Categorical(
        prediction_grid["sex"],
        categories=sample["sex"].cat.categories
    )


    # --------------------------------------------------------
    # Model predictions and 95% CI
    # --------------------------------------------------------

    pred_linear = linear_model.get_prediction(
        prediction_grid
    ).summary_frame(alpha=0.05)

    prediction_grid["prediction"] = pred_linear["mean"]
    prediction_grid["ci_lower"] = pred_linear["mean_ci_lower"]
    prediction_grid["ci_upper"] = pred_linear["mean_ci_upper"]


    # ========================================================
    # Condition-specific line style
    # ========================================================

    # Default values
    linewidth = 2
    linestyle = "-"
    fill = False

    if cond == "word_written":
        linewidth = 1
        linestyle = "--"
        label = f"Written ($\\beta$ = {beta:.2f}{sig})"

    elif cond == "word_spoken":
        linewidth = 1
        linestyle = "--"
        label = f"Spoken ($\\beta$ = {beta:.2f}{sig})"

    elif cond == "word_multi":
        linewidth = 2
        linestyle = "-"
        label = f"Spoken & Written ($\\beta$ = {beta:.2f}{sig})"
        fill = True


    # --------------------------------------------------------
    # All other ROIs
    # --------------------------------------------------------
    else:

        if cond == "word_written":
            label = f"Written ($\\beta$ = {beta:.2f})"

        elif cond == "word_spoken":
            label = f"Spoken ($\\beta$ = {beta:.2f})"

        elif cond == "word_multi":
            label = f"Spoken & Written ($\\beta$ = {beta:.2f})"


    # ========================================================
    # Regression line
    # ========================================================

    ax.plot(
        prediction_grid[dependent],
        prediction_grid["prediction"],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        zorder=5
    )


    # ========================================================
    # Confidence interval
    # ========================================================

    if fill:

        ax.fill_between(
            prediction_grid[dependent],
            prediction_grid["ci_lower"],
            prediction_grid["ci_upper"],
            color=color,
            alpha=0.10,
            zorder=1
        )


# ============================================================
# Axis settings
# ============================================================

if dependent == "grade":

    ax.set_xlabel(
        "School grade",
        fontsize=18
    )
    if target == "Temporal Occipital Fusiform Cortex":
        ax.set_ylim(
            0,
            0.5
        )
    elif target == "Superior Temporal Gyrus, posterior division":
        ax.set_ylim(
            0,
            0.6
        )

    # Explicitly label the three school grades
    ax.set_xticks(grades)
    ax.set_xticklabels(
        ["1", "2", "4"]
    )

    loc = "upper left"


ax.set_ylabel(
    r"Similarity (Fisher's $z$)",
    fontsize=18
)


# ============================================================
# Figure appearance
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="both",
    labelsize=14
)

ax.legend(
    frameon=False,
    fontsize=12,
    bbox_to_anchor=(0, 1.1),
    loc=loc
)

plt.tight_layout()


# ============================================================
# Save figure
# ============================================================

fig_path = FIG_DIR / "multimodal"

fig_name = (
    f"{HEMI}_{target}_model.pdf"
)

plt.savefig(
    fname=fig_path / fig_name,
    format="pdf",
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.3
)

plt.show()

# %%
