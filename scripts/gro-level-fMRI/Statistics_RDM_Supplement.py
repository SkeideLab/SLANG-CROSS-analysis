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



# %
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
roi_abbrev      = {
                        "Frontal Pole": "FP",
                        "Superior Frontal Gyrus": "SFG",
                        "Middle Frontal Gyrus": "MFG",
                        "Inferior Frontal Gyrus, pars triangularis": "trIFG",
                        "Inferior Frontal Gyrus, pars opercularis": "opIFG",
                        "Precentral Gyrus": "PreCG",
                        "Frontal Medial Cortex": "FMC",
                        "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "SMA",
                        "Subcallosal Cortex": "SubC",
                        "Paracingulate Gyrus": "PCG",
                        "Cingulate Gyrus, anterior division": "aCG",
                        "Cingulate Gyrus, posterior division": "pCG",
                        "Frontal Orbital Cortex": "FOrC",
                        "Frontal Opercular Cortex": "FOpC",
                        "Insular Cortex": "Ins",
                        
                        "Postcentral Gyrus": "PostCG",
                        "Superior Parietal Lobule": "SPL",
                        "Supramarginal Gyrus, anterior division": "aSMG",
                        "Supramarginal Gyrus, posterior division": "pSMG",
                        "Angular Gyrus": "AG",
                        "Precuneous Cortex": "Prec",
                        "Central Opercular Cortex": "COC",
                        "Parietal Opercular Cortex": "POC",
                        
                        "Temporal Pole": "TP",
                        "Superior Temporal Gyrus, anterior division": "aSTG",
                        "Superior Temporal Gyrus, posterior division": "pSTG",
                        "Middle Temporal Gyrus, anterior division": "aMTG",
                        "Middle Temporal Gyrus, posterior division": "pMTG",
                        "Middle Temporal Gyrus, temporooccipital part": "toMTG",
                        "Inferior Temporal Gyrus, anterior division": "aITG",
                        "Inferior Temporal Gyrus, posterior division": "pITG",
                        "Inferior Temporal Gyrus, temporooccipital part": "toITG",
                        "Parahippocampal Gyrus, anterior division": "aPHG",
                        "Parahippocampal Gyrus, posterior division": "pPHG",
                        "Planum Polare": "PP",
                        "Heschl's Gyrus (includes H1 and H2)": "HG",
                        "Planum Temporale": "PT",

                        "Temporal Fusiform Cortex, anterior division": "aTFC",
                        "Temporal Fusiform Cortex, posterior division": "pTFC",
                        "Lateral Occipital Cortex, superior division": "sLOC",
                        "Lateral Occipital Cortex, inferior division": "iLOC",
                        "Intracalcarine Cortex": "ICC",
                        "Cuneal Cortex": "Cuneus",
                        "Lingual Gyrus": "Ling",
                        "Occipital Fusiform Gyrus": "OFG",
                        "Temporal Occipital Fusiform Cortex": "TOFC",
                        "Supracalcarine Cortex": "SCC",
                        "Occipital Pole": "OP"
                    }


# %
# 3. === STEP 3 ===: read RDM csv file and remove outliers
# -----------------------------------------------
# --- read RDM ---
path = OUT_DIR / 'multimodal' / f'{HEMI}_RDM_metrics_supplements.csv'
df   = pd.read_csv(path)
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
conditions = ['word_multi', 'pseudo_multi', 'semantic_multi']
df_clean_dict = {}
removed_dict  = {}
# --- remove it for each condition ---
for cond in conditions:
    df_cond            = df[['roi', 'grade', 'rt', cond]].copy()
    removed            = get_removed_outliers(df_cond, 'roi', cond)
    df_clean           = df_cond.drop(removed.index).copy()
    # --- Fisher transformation ---
    df_clean["Fisher"] = np.arctanh(
        df_clean[cond].clip(-0.999999, 0.999999)
    )
    df_clean_dict[cond] = df_clean
    removed_dict[cond]  = removed
    # --- print out ---
    print(f"\nRemoved rows for {cond}:")
    print(removed)



# %
# 4. === STEP 4 ===: Distirbution of RDM metrcis 
# -----------------------------------------------
 # --- figure ---
fig, axes = plt.subplots(3, 1, figsize=(20, 15))
axes = axes.flatten() 
 # --- labels ---
fig_path   = FIG_DIR / 'multimodal'
fig_name   = f"{HEMI}_Correlations_Supplements.pdf"
titles     = ['Word', 'Pseudoword', 'Semantic']
xlabels    = [label for label in roi_abbrev.values()]
order      = list(roi_color_map.keys())
palette    = list(roi_color_map.values())   # IMPORTANT: use dict, not list
legend_map = {
                "Frontal lobe": "goldenrod",
                "Parietal lobe": "dodgerblue",
                "Temporal lobe": "mediumvioletred",
                "Occipital lobe": "limegreen"
            }
# --- plot ---
for i, cond in enumerate(conditions):
    ax      = axes[i]
    df_cond = df_clean_dict[cond]
    df_cond["roi"] = pd.Categorical(
                                        df_cond["roi"],
                                        categories=order,
                                        ordered=True
                                    )
    df_cond = df_cond.sort_values("roi")

    # --- Plotting: RainCloud ---
    pt.RainCloud(x              = "roi", 
                 y              = cond, 
                 data           = df_cond,
                 order          = order,
                 palette        = palette,
                 point_size     = 3,
                 rain_alpha     = 0.6,
                 width_viol     = 1, 
                 width_box      = 0.3, 
                 cloud_alpha    = 1,
                 offset         = 0.2, 
                 box_showfliers = False,  
                 ax             = ax)
    # --- Aesthetics ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([-0.5, 1]) # Dynamic limit to fit stars
    ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)
    short_title = titles[i]
    ax.set_title(short_title, fontsize=17)
    ax.set_xticklabels(xlabels, fontsize=9, rotation=60)
    ax.axhline(y=0, linestyle='--', linewidth=1, color='black', alpha=0.7)
    ax.set_xlabel(" ", fontsize=15)
    ax.set_ylabel(r"Multimodal similarity ($r$)", fontsize=13)
# add legend
legend_handles = [
    Patch(facecolor=color, edgecolor='black', linewidth=1, label=lobe)
    for lobe, color in legend_map.items()
]
fig.legend(
    handles=legend_handles,
    loc="center right",
    title=HEMI.capitalize(),
    bbox_to_anchor=(1.03, 1),
    fontsize=15,
    title_fontsize=16,
)
# add panel logo
if HEMI == "left":
    logo = "A"
elif HEMI == "right":
    logo = "B"
fig.text(
    0.01, 1.05, logo,
    fontsize=30,
    fontweight="bold",
    ha="left",
    va="top"
)
fig.subplots_adjust(hspace=0.5)
plt.tight_layout()
# --- Save ---
fig_path.mkdir(exist_ok=True, parents=True)
plt.savefig(
    fname       = fig_path / fig_name,
    format      = 'pdf',
    dpi         = 300,
    transparent = True,
    bbox_inches = 'tight',
    pad_inches  = 0.3
)
print(f"\nSuccessful: Figure is saved ")
plt.show()



# %
# 5. === STEP 5 ===: Distirbution of RDM metrcis by grade
# -----------------------------------------------
# --- figure ---
fig, axes = plt.subplots(3, 1, figsize=(20, 15))
axes      = axes.flatten()

# --- key, title, fontsize ---
fig_path  = FIG_DIR / 'multimodal'
fig_name  = f"{HEMI}_Correlations_gradewise_supplements.pdf"
configs   = [
             ("word_multi",     "Word",     15),
             ("pseudo_multi",   "Pseudoword",   15),
             ("semantic_multi", "Semantic", 15),
            ]
xlabels    = [label for label in roi_abbrev.values()]
legend_map = {
                1: "darkgoldenrod",
                2: "darkcyan",
                4: "firebrick",
            }
# --- function for consistent axes ---
def style_axis(ax, title, fontsize, xlabels):
    ax.set_title(title, fontsize=17)
    ax.set_xlabel(" ", fontsize=fontsize)
    ax.set_xticklabels(xlabels, fontsize=9, rotation=60)
    ax.set_ylabel(r"Multimodal similarity ($r$)", fontsize=fontsize)
    ax.set_ylim(-0.2, 0.6)
    ax.tick_params(axis="both", which="major", length=3, width=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ax.legend_:
        ax.legend_.remove()
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.axhline(y=0, linestyle='--', linewidth=1, color='black', alpha=0.7)
    ax.grid(True, axis='y', color='gray', alpha=0.3)
# --- plot ---
for ax, (key, title, fs) in zip(axes, configs):
    sns.barplot(
        data=df_clean_dict[key],
        x="roi",
        y=key,
        hue="grade",         
        palette=legend_map, 
        edgecolor="black",          
        ax=ax
    )
    # adjust transparency post hoc
    for patch in ax.patches:
        patch.set_alpha(0.8)   # decrease alpha (0–1)

    style_axis(ax, title, fs, xlabels)
# add legend
legend_handles = [
    Patch(facecolor=color, edgecolor='black', linewidth=1, label=grade)
    for grade, color in legend_map.items()
]
fig.legend(
    handles=legend_handles,
    loc="center right",
    title="Grade",
    bbox_to_anchor=(1.03, 1),
    fontsize=15,
    title_fontsize=16,
)
# add panel logo
if HEMI == "left":
    logo = "A"
elif HEMI == "right":
    logo = "B"
fig.text(
    0.01, 1.05, logo,
    fontsize=30,
    fontweight="bold",
    ha="left",
    va="top"
)
fig.subplots_adjust(hspace=0.5)
plt.tight_layout()
# --- save ---
fig_path.mkdir(exist_ok=True, parents=True)
plt.savefig(
    fname       = fig_path / fig_name,
    format      = 'pdf',
    dpi         = 300,
    transparent = True,
    bbox_inches = 'tight',
    pad_inches  = 0.3
)
print("\nSuccessful: Figure is saved")
plt.show()




# %
# 6. === STEP 6 ===: Statistics: One-sample t-tetst
# -----------------------------------------------
all_results = []
# --- each condtion ---
for cond in conditions:
    data    = df_clean_dict[cond]
    results = []
    for roi, g in data.groupby("roi"):
        x             = g["Fisher"].dropna() # Fisher's correlation coefficients
        t_stat, p_val = stats.ttest_1samp(x, 0)
        results.append({
                            "condition": cond,
                            "roi":       roi,
                            "t":         t_stat,
                            "p":         p_val,
                            "n":         len(x)
                        })
    df_cond = pd.DataFrame(results)
    # --- FDR correction ---
    valid_mask = df_cond["p"].notna()
    pvals      = df_cond.loc[valid_mask, "p"].values
    reject, p_fdr, _, _ = multipletests(
                                         pvals,
                                         alpha  = 0.05,
                                         method = "fdr_bh"
                                        )
    df_cond["p_fdr"]                           = np.nan
    df_cond["significant_fdr"]                 = False
    df_cond.loc[valid_mask, "p_fdr"]           = p_fdr
    df_cond.loc[valid_mask, "significant_fdr"] = reject
    # --- store ---
    all_results.append(df_cond)
# --- resutls with all conditions ---
df_all        = pd.concat(all_results, ignore_index=True)
df_top5_abs = (
    df_all
    .groupby("condition", group_keys=False)
    .apply(lambda x: x.loc[x["t"].abs().nlargest(5).index])
    .reset_index(drop=True)
)
print(df_top5_abs)


# %
# 7. === STEP 7 ===: Statistics: Linear regression
# -----------------------------------------------

all_results_lr = []
# --- Fisher's r = β0 + β1*Grade/RT + e ---
predictors = ["grade", "rt"]
# --- each predictor ---
for pred in predictors:
    # --- each condition ---
    for cond in conditions:
        data = df_clean_dict[cond]
        rows = []
        for roi, g in data.groupby("roi"):
            model = smf.ols(f"Fisher ~ {pred}", data=g).fit()
            rows.append((
                cond,
                roi,
                model.params.get(pred, np.nan),
                model.tvalues.get(pred, np.nan),
                model.pvalues.get(pred, np.nan),
                model.rsquared,
                len(g),
                pred
            ))
        df_lm = pd.DataFrame(
            rows,
            columns = ["condition", "roi", "beta", "t", "p", "r2", "n", "predictor"]
        )
        # --- FDR correction ---
        df_lm["p_fdr"]           = np.nan
        df_lm["significant_fdr"] = False
        valid                    = df_lm["p"].notna()
        if valid.any():
            reject, p_fdr, _, _  = multipletests(
                                                    df_lm.loc[valid, "p"],
                                                    alpha  = 0.05,
                                                    method = "fdr_bh"
                                                )
            df_lm.loc[valid, "p_fdr"]           = p_fdr
            df_lm.loc[valid, "significant_fdr"] = reject
        # --- combine all conditions ---
        all_results_lr.append(df_lm)
# --- combine all predictors ---
df_lm_all = pd.concat(all_results_lr, ignore_index=True)
print(df_lm_all[df_lm_all["significant_fdr"] == True])
# %%
