# %%
# ===  Load modules ===
from pathlib import Path
import templateflow.api as tflow
import nibabel as nib
import seaborn as sns
import numpy as np
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import datasets
from nilearn.image import resample_to_img
from nilearn import image
from nilearn.maskers import NiftiSpheresMasker
from matplotlib import colors
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_roi, show
import ants
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors


# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
TMPL_DIR     = ANALY_DIR / 'templates'

# === Parameters ===
MODEL          = 'glm'
METHOD         = 'parametric' # 'parametric' 'non-parametric'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_pseudo'
GRADES         = ['1', '2', '4'] # 1, 2, 4
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
CORRECTION     = 'fpr' # fdr, fpr, rft
P_CORRECTION   = 0.001 # 0.001, 0.05
CLUSTER_SIZE   = 50
RADIUS         = 6
MNI_VWFA       = [-45, -57, -12]
MASK_TYPE      = 'ventral' # ventral, VWFA, fusiform
EXC_SUBJECTS   = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434']
# Retrieve the T1-weighted template for cohort 4 (7.5-13.5yrs) at 2mm resolution
TEMPLATE       =  tflow.get(
    "MNIPediatricAsym",
    cohort="4",       # cohort 4 = age 7.5-13.5
    resolution=2,     # 2mm isotropic
    suffix="T1w"      # T1-weighted image
)
# List of the contrasts
"""             'images_words-images_pseudo' 
                'audios_words-audios_pseudo' 
                'images_words' 
                'audios_words'
                'images_pseudo'
                'audios_pseudo'
                'images_words+audios_words-images_pseudo+audios_pseudo' """


# %%
# === Compute total volume (mm3) ===
results = []



# Ventral: load masks
if MASK_TYPE == 'ventral':
    left_mask_path = TMPL_DIR / 'mask' / 'left_ventral_pediatric_MNI.nii.gz'
    right_mask_path = TMPL_DIR / 'mask' / 'right_ventral_pediatric_MNI.nii.gz'
elif MASK_TYPE == 'VWFA':
    left_mask_path = TMPL_DIR / 'mask' / 'left_VWFA_pediatric_MNI.nii.gz'
    right_mask_path = TMPL_DIR / 'mask' / 'right_VWFA_pediatric_MNI.nii.gz'
left_mask_img = nib.load(left_mask_path)
right_mask_img = nib.load(right_mask_path)
left_mask = left_mask_img.get_fdata().astype(np.uint8)
right_mask = right_mask_img.get_fdata().astype(np.uint8)



# VWFA: load masks
left_vwfa_mask_path  = TMPL_DIR / 'mask' / 'left_VWFA_pediatric_MNI.nii.gz'
right_vwfa_mask_path = TMPL_DIR / 'mask' / 'right_VWFA_pediatric_MNI.nii.gz'

left_vwfa_mask_img  = nib.load(left_mask_path)
right_vwfa_mask_img = nib.load(right_mask_path)

left_vwfa_mask  = left_vwfa_mask_img.get_fdata().astype(np.uint8)
right_vwfa_mask = right_vwfa_mask_img.get_fdata().astype(np.uint8)


# Get the voxels info
voxel_sizes      = left_mask_img.header.get_zooms()  # should be (2.0, 2.0, 2.0)
voxel_volume     = voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]
left_volume_mm3  = voxel_volume * np.sum(left_mask)
right_volume_mm3 = voxel_volume * np.sum(right_mask)


for GRADE in GRADES:

    # list the beta nii.files
    subjects      = sorted(DERIV_DIR.glob(f"sub-{GRADE}*"))
    exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
    subjects      = [s for s in subjects if s.name not in exclude]
    subject_names = [p.name.replace('sub-', '') for p in subjects]


    for sub in subjects:
        path      = sub / MODEL / SPACE / f"FWHM_{int(FWHM_SMOOTHING)}" 

        # beta maps
        b_path    = next(path.glob(f"{CONTRASTS}_beta.nii.gz"), None)
        b_img     = nib.load(str(b_path))
        b_data    = b_img.get_fdata()

        # z maps
        z_path    = next(path.glob(f"{CONTRASTS}_zscore.nii.gz"), None)
        z_img     = nib.load(str(z_path))
        z_data    = z_img.get_fdata()

        # beta-values within the ventral mask
        b_left  = b_data[left_mask.astype(bool)]
        b_right = b_data[right_mask.astype(bool)]
        # mean
        mean_b_left  = np.nanmean(b_left)
        mean_b_right = np.nanmean(b_right)

        # beta-values within the vwfa mask
        b_left_vwfa  = b_data[left_vwfa_mask.astype(bool)]
        b_right_vwfa = b_data[right_vwfa_mask.astype(bool)]
        # mean
        mean_b_left_vwfa  = np.nanmean(b_left_vwfa)
        mean_b_right_vwfa = np.nanmean(b_right_vwfa)
        # peak beta
        max_b_left_vwfa   = np.nanmax(b_left_vwfa)
        max_b_right_vwfa  = np.nanmax(b_right_vwfa)

        # apply p-value threshold
        z_map_thr, threshold = threshold_stats_img(
            z_img,
            alpha=P_CORRECTION,
            height_control=CORRECTION,
            cluster_threshold=CLUSTER_SIZE,
            two_sided=False,  # using two-sided test
        )
        z_thr_data = z_map_thr.get_fdata()

        # Only keep voxels that are in the mask
        left_significant_mask  = (z_thr_data != 0) & left_mask
        right_significant_mask = (z_thr_data != 0) & right_mask

        # count the significant voxels within the ventral mask
        n_left_sig  = int(left_significant_mask.sum())
        n_right_sig = int(right_significant_mask.sum())

        # total volume computation
        sig_left_volume  = voxel_volume * np.sum(left_significant_mask)
        sig_right_volume = voxel_volume * np.sum(right_significant_mask)

        # percentage of significant volume within the mask
        perc_sig_left = (sig_left_volume / left_volume_mm3) * 100
        perc_sig_right = (sig_right_volume / right_volume_mm3) * 100

        # add a row to the results
        results.append({
            "grade": GRADE,
            "subject": sub.name,
            "n_left_sig": n_left_sig,
            "n_right_sig": n_right_sig,
            "sig_left_volume": sig_left_volume,
            "sig_right_volume": sig_right_volume,
            "perc_sig_left": perc_sig_left,
            "perc_sig_right": perc_sig_right,
            "mean_b_left": mean_b_left,
            "mean_b_right": mean_b_right,
            "mean_b_left_vwfa": mean_b_left_vwfa,
            "mean_b_right_vwfa": mean_b_right_vwfa,
            "max_b_left_vwfa": max_b_left_vwfa,
            "max_b_right_vwfa": max_b_right_vwfa
        })

        print(f"Successful: {sub.name}")

df = pd.DataFrame(results)


# %%
# ======== STATISTICS ========
# ======== VISUALIZATION ========

# ======== Parameters ========
# color specification
grade_colors = {
    "1": "darkgoldenrod", # firebrick
    "2": "darkcyan",
    "4": "firebrick" # darkgoldenrod
}
# directory
dir = FIG_DIR / "evol_vol"
dir.mkdir(parents=True, exist_ok=True)



# ======== Data frame ========
# total volume
df_vol = pd.melt(
    df,
    id_vars=["subject", "grade"],
    value_vars=["sig_left_volume", "sig_right_volume"],
    var_name="hemisphere",
    value_name="total_volume")
# mean in the VWFA
df_mean = pd.melt(
    df,
    id_vars=["subject", "grade"],
    value_vars=["mean_b_left_vwfa", "mean_b_right_vwfa"],
    var_name="hemisphere",
    value_name="mean_b_vwfa")
# max value in the VWFA
df_max = pd.melt(
    df,
    id_vars=["subject", "grade"],
    value_vars=["max_b_left_vwfa", "max_b_right_vwfa"],
    var_name="hemisphere",
    value_name="max_b_vwfa")

# Make hemisphere labels clean
df_vol["hemisphere"] = df_vol["hemisphere"].replace({
    "sig_left_volume": "left",
    "sig_right_volume": "right"
})
df_mean["hemisphere"] = df_mean["hemisphere"].replace({
    "mean_b_left_vwfa": "left",
    "mean_b_right_vwfa": "right"
})
df_max["hemisphere"] = df_max["hemisphere"].replace({
    "max_b_left_vwfa": "left",
    "max_b_right_vwfa": "right"
})

# converge all the dataframe
df_long = df_vol.merge(df_mean, on=["subject", "grade", "hemisphere"])
df_long = df_long.merge(df_max,on=["subject", "grade", "hemisphere"])



# ======== Statistics & Visualization ========
dependents = ["total_volume", "mean_b_vwfa", "max_b_vwfa"]
for dependent in dependents:

    means = df_long.groupby(["grade", "hemisphere"])[dependent].mean().unstack()

    # Compute standard errors
    ses = df_long.groupby(["grade", "hemisphere"])[dependent].sem().unstack()

    print("=========================================")
    print(f"\nDependent variable: {dependent}\n")
    print("mean")
    print(means)
    print("standard error of the mean")
    print(ses)
    print("=========================================")

    # parametric assumption satisfied?
    results = []

    for g in df_long['grade'].unique():
        for h in df_long['hemisphere'].unique():
            subset = df_long[(df_long['grade'] == g) &
                            (df_long['hemisphere'] == h)]

            res = pg.normality(subset[dependent])
            res['grade'] = g
            res['hemisphere'] = h
            results.append(res)

    normality_table = pd.concat(results, ignore_index=True)
    print("=========================================")
    print(f"\nNormality test:\n")
    print(normality_table)
    print("=========================================")

    ### ======= Parametric test ======= 
    # two-way mixed ANOVA
    anova_res = pg.mixed_anova(
        data=df_long,
        dv=dependent,
        within="hemisphere",
        between="grade",
        subject="subject"
    )
    print("=========================================")
    print(f"\nContrast: {CONTRASTS}\n")
    print("\nTwo-way mixed ANOVA\n")
    print(anova_res)
    print("=========================================")


    # pairwise comparisons
    pw_grade = pg.pairwise_tests(
        data=df_long,
        dv=dependent,
        between="grade",
        within="hemisphere",
        subject="subject",
        padjust="fdr_bh",     # Bonferroni correction (or 'fdr_bh')
        effsize="hedges"
    )
    print("=========================================")  
    print("\nParametric\n")
    print("=========================================")  
    print("=========================================") 
    print("Pairwise grade comparisons (within each hemisphere):\n")
    print(pw_grade)
    print("=========================================") 


    ### ======= Non-Parametric test =======
    # Kruskal-Wallis within each hemisphere (one-way ANOVA)
    kw_results = []
    for hemi in df_long['hemisphere'].unique():
        df_h = df_long[df_long['hemisphere'] == hemi]
        kw = pg.kruskal(data=df_h, dv=dependent, between='grade')
        kw['hemisphere'] = hemi
        kw_results.append(kw)

    kw_results = pd.concat(kw_results, ignore_index=True) 

    # Mann-Whitney U test
    results = []
    for hemi in df_long['hemisphere'].unique():
        df_h = df_long[df_long['hemisphere'] == hemi]
        res = pg.pairwise_tests(
            data=df_h,
            dv=dependent,
            between='grade',
            parametric=False,     # <-- rank-based
            padjust='fdr_bh' # Benjamini-Hochberg discovery rate correction within a hemisphere
        )
        res['hemisphere'] = hemi
        results.append(res)

    pw_rank_grade = pd.concat(results, ignore_index=True)
    print("=========================================")  
    print("\nNon-Parametric\n")
    print("=========================================") 
    print("\nKruskal-Wallis (within each hemisphere):\n")
    print(kw_results)
    print("=========================================") 
    print("\nMann-Whitney Pairwise grade comparisons (within each hemisphere):\n")
    print(pw_rank_grade)
    print("=========================================") 


    # === Visualization ===
    # X positions
    x = np.arange(len(means.index))  # number of grade levels

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, hemi, title in zip(
        axes,
        ["left", "right"],
        ["Left hemisphere", "Right hemisphere"]
    ):
        
         # --- violins plot ---
        vp = sns.violinplot(
            data=df_long[df_long["hemisphere"]==hemi], 
            x="grade", 
            y=dependent,
            hue="grade",
            palette=grade_colors,
            inner=None,
            cut=0,
            linewidth=0,
            legend=False,
            width=0.5,
            ax=ax)

        if hemi=="left":
            x_shift = -0.15   # negative = move left, positive = move right
        else:
            x_shift = 0.15
        # --- clip violins to half ---
        for violin in vp.collections:
            violin.set_edgecolor("none")   # remove outline
            violin.set_alpha(0.4)

            # ---- shift the violin ----
            for path in violin.get_paths():
                verts = path.vertices
                verts[:, 0] += x_shift
                path.vertices = verts

            bbox = violin.get_paths()[0].get_extents()
            xmid = (bbox.xmin + bbox.xmax) / 2
            if hemi=="left":
                # left half
                violin.set_clip_path(
                    plt.Rectangle(
                        (bbox.xmin, bbox.ymin),
                        (bbox.xmax + bbox.xmin)/2 - bbox.xmin,
                        bbox.ymax - bbox.ymin,
                        transform=ax.transData
                    )
                )
            else:
                violin.set_clip_path(
                    plt.Rectangle(
                        (xmid, bbox.ymin),   # right half
                        bbox.xmax - xmid,
                        bbox.ymax - bbox.ymin,
                        transform=ax.transData
                    )
                )

        # --- boxplot ---
        bp =sns.boxplot(
            data=df_long[df_long["hemisphere"]==hemi],
            x="grade",
            y=dependent,
            hue="grade",
            width=0.2,                   # narrower box
            showcaps=False,
            boxprops=dict(linewidth=2),
            medianprops=dict(linewidth=4),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            showfliers=False,
            palette=grade_colors,
            ax=ax,
            # positions=positions,
        )
        # ---- force full opacity ----
        for patch in bp.patches:
            face = patch.get_facecolor()
            patch.set_facecolor(face[:3] + (0.1,))  # only face transparent
            patch.set_edgecolor(face[:3] + (1.0,))  # edge fully opaque


        # ---- match all line artists to their corresponding box color ----
        # seaborn draws lines in groups per box: whisker, whisker, median
        lines_per_box = int(len(bp.lines) / len(bp.patches))

        for i, patch in enumerate(bp.patches):
            face = patch.get_facecolor()
            for line in bp.lines[i * lines_per_box : (i + 1) * lines_per_box]:
                line.set_color(face)
                line.set_alpha(1.0)

        # scatter plot
        sp = sns.stripplot(
            data=df_long[df_long["hemisphere"]==hemi], 
            x="grade",
            hue="grade", 
            y=dependent, 
            palette=grade_colors,
            dodge=False, 
            size=8,
            ax=ax
            )
        # decrease alpha for all markers
        for artist in sp.collections:
            facecolors = artist.get_facecolors()
            artist.set_edgecolor(facecolors)
            artist.set_alpha(0.4)  # adjust transparency (0 = fully transparent, 1 = opaque)
            artist.set_linewidth(1.2)
    

        # ---- Clean up the figure ----
        if dependent == "total_volume":
            ax.set_ylim(0,25000)
        elif dependent == "mean_b_vwfa":
            ax.set_ylim(-1,1)
        elif dependent == "max_b_vwfa":
            ax.set_ylim(0,7)
        
        ax.tick_params(axis="both", labelsize=15)
        ax.yaxis.grid(True, linestyle='--', color='gray', linewidth=0.7, alpha=0.8)

        # remove 
        ax.set_title(None)   # completely remove subplot title
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        if hemi=="right":
            ax.spines['left'].set_visible(False)
            # Remove y-tick marks
            ax.tick_params(left=False)


    plt.tight_layout()
    plt.savefig(f"{dir}/{MASK_TYPE}_{CONTRASTS}_{dependent}.png", dpi=300)
    plt.show()

# %%
