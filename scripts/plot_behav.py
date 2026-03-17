# %%
# ===  Load modules ===
from nilearn.plotting import plot_glass_brain
from nilearn import plotting, image
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'

# === Parameters ===
MODALITYS    = ['all', 'visual', 'audio']# all, visual, audio
TARGET       = 'start' # accuracy, RT, dprime, drift, start
CRITERIA     = 'excluded' # all, excluded
EXC_SUBJECTS = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434']

# make a figure directory
OUT_DIR      = ANALY_DIR / 'outputs' / 'behavior' / CRITERIA
FIG_DIR      = ANALY_DIR / 'figures' / 'beheavior' / CRITERIA
FIG_DIR.mkdir(parents=True, exist_ok=True)


# %% Load the Data
for MODALITY in MODALITYS:

    if TARGET in ['drift', 'start']:
        # === DDM ===
        ddm_str         = f"ddm_{MODALITY}"
        path            = OUT_DIR / f"{ddm_str}.csv"
        df              = pd.read_csv(path)
        df['grade']     = (df['subject'] // 100).astype(int)

    elif TARGET in ['accuracy', 'RT', 'dprime']:
        # === Behavior summary ===
        subjects          = sorted([d.name for d in DERIV_DIR.iterdir() if d.is_dir() and d.name.startswith('sub')])
        exclude           = [f"sub-{s}" for s in EXC_SUBJECTS]
        if CRITERIA == 'excluded':
            subjects_filtered = [s for s in subjects if s not in exclude]
        elif CRITERIA == 'all':
            subjects_filtered = subjects

        # List to store each subject's DataFrame
        all_dfs           = []

        for subject in subjects_filtered:

            path          = DERIV_DIR / subject / 'behavior' / CRITERIA / 'accuracy_summary.csv'
            df            = pd.read_csv(path)
            df['subject'] = subject
            all_dfs.append(df)

        # Concatenate all DataFrames into one
        df          = pd.concat(all_dfs, ignore_index=True)
        df['grade'] = df['subject'].str.replace('sub-', '', regex=False).str[0].astype(np.int64)
        df['run']   = df['file_name'].str.extract(r'run-(\d+)_events').astype(int)
        if MODALITY in ['all', 'visual', 'audio']:
            col_name    = f"accuracy_{MODALITY}"
            df          = df[df[col_name] != -1]
            df          = df[df[col_name] != 0]
        else:
            raise ValueError(f"Unknown MODALITY: {MODALITY}")

        # test whether the average is above chance
        acc_str    = f"accuracy_{MODALITY}"
        rt_str     = f"RT_{MODALITY}"
        dprime_str = f"dprime_{MODALITY}"
        c_str      = f"c_{MODALITY}"
        mean       = df.groupby('subject').agg(
            mean_accuracy = (acc_str, 'mean'),
            mean_RT       = (rt_str, 'mean'),
            mean_dprime   = (dprime_str, 'mean'),
            mean_c        = (c_str, 'mean'),
            grade         = ('grade', 'first')
        ).reset_index()

    else:
        raise ValueError(
            f"Unknown Target: {TARGET}\n"
            "Please choose one of the following\n"
            "accuracy, RT, dprime, drift")

    # %
    # === One-sample T-test ===
    if TARGET=='accuracy':
        
        results = []
        # loop over each grade
        for g, group in mean.groupby('grade'):
            t_stat, p_value = stats.ttest_1samp(
                group['mean_accuracy'], 
                50, 
                alternative='greater'  # test if mean > 0.5
            )
            results.append({'grade': g, 't_stat': t_stat, 'p_value': p_value, 'n_subjects': len(group)})

        # convert results to dataframe for easy viewing
        ttest_results = pd.DataFrame(results)
        print(f"Modality: {MODALITY}")
        print(ttest_results)
    else:
        print(f"\n{TARGET}: one-sample ttest is not included")


    # === One-way ANOVA ===
    if TARGET in ['accuracy', 'RT', 'dprime']:
        str = f"mean_{TARGET}"

        groups = [mean.loc[mean['grade'] == g, str] for g in mean['grade'].unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        print("\nOne-way ANOVA")
        print(f"Variable: {TARGET}")
        print(f"Modality: {MODALITY}")
        print(f"F = {f_stat:.3f}, p = {p_value:.4f}")
        tukey = pairwise_tukeyhsd(endog=mean[str], groups=mean['grade'], alpha=0.05)
        print(tukey)

    elif TARGET in ['drift', 'start']:
        # ddm: drift
        groups = [df.loc[df['grade'] == g, TARGET] for g in df['grade'].unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        print("\nOne-way ANOVA")
        print(f"Variable: {TARGET}")
        print(f"Modality: {MODALITY}")
        print(f"F = {f_stat:.3f}, p = {p_value:.4f}")
        tukey = pairwise_tukeyhsd(endog=df['drift'], groups=df['grade'], alpha=0.05)
        print(tukey)

    else:
        raise ValueError(
            f"Unknown Target: {TARGET}\n"
            "Please choose one of the following\n"
            "accuracy, RT, dprime, drift")

    # %
    # === Scatter plot ===
    if TARGET in ['accuracy', 'RT', 'dprime']:
        str = f"{TARGET}_{MODALITY}" 
        # Set a nice theme
        sns.set(style="whitegrid")
        # Bar plot with grade on x-axis, accuracy on y-axis, hue=run
        plt.figure(figsize=(8,6))
        sns.stripplot(
            data=df,
            x='grade', 
            y=str, 
            hue='run',
            jitter=True,       # spreads dots horizontally for visibility
            dodge=True,        # separates runs side-by-side
            palette='Set2',
            size=6,            # size of the dots
        )

        plt.title(f"{TARGET} by grade and run")
        if TARGET=='accuracy':
            plt.ylabel(f"{TARGET} (%)")
        elif TARGET=='RT':
            plt.ylabel(f"{TARGET} (s)")
        elif TARGET=='dprime':
            plt.ylabel(f"{TARGET} (d')") 
        plt.xlabel('grade')
        # plt.ylim(0,100)
        plt.legend(title='run', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.savefig(f"{FIG_DIR}/{str}_scatter.png", dpi=300, bbox_inches='tight')

    else:
        print(f"{TARGET}: scatter plot is not included")

    # %
    # === Violin plot ===
    if TARGET in ['accuracy', 'RT', 'dprime']: 
        str = f"mean_{TARGET}"

        plt.figure(figsize=(8,6))
        sns.violinplot(
            data=mean,
            x='grade',
            y=str,
            inner='quartile',
            split=False,
            palette='Set2',
            cut=0,
            legend=False,
            alpha=0.4
        )

        plt.title(f"{TARGET} distribution by grade")
        plt.xlabel('grade')
        if TARGET=='accuracy':
            plt.ylabel(f"{TARGET} (%)")
            plt.ylim(0, 100)
            plt.yticks([0, 25, 50, 75, 100])
            plt.axhline(y=50, color='red', linestyle='--', linewidth=1.5)  # dashed red line
        elif TARGET=='RT':
            plt.ylabel(f"{TARGET} (s)")
            plt.ylim(0, 4)
            plt.yticks([0, 1, 2, 3, 4])
        elif TARGET=='dprime':
            plt.ylabel(f"{TARGET} (d')") 
        plt.grid(False)
        plt.savefig(f"{FIG_DIR}/{TARGET}_{MODALITY}_violin.png", dpi=300, bbox_inches='tight')
        plt.show()

    elif TARGET in ['drift', 'start']:
        # === DDM: drift ===
        plt.figure(figsize=(8,6))
        sns.violinplot(
            data=df,
            x='grade',
            y=TARGET,
            inner='quartile',
            split=False,
            palette='Set2',
            cut=0,
            legend=False,
            alpha=0.4
        )
        if  TARGET=='drift':
            plt.title(f"{TARGET} rate (v) distribution by grade")
            plt.ylabel(f"{TARGET} rate (v)")
        elif TARGET=='start':
            plt.title("Starting point bias (z) distribution by grade")
            plt.ylabel("Starting point bias (z)")
            plt.ylim(-1, 1)
        plt.grid(False)
        plt.xlabel("grade")
        plt.savefig(f"{FIG_DIR}/{TARGET}_{MODALITY}_violin.png", dpi=300, bbox_inches='tight')
        plt.show()

    else:
        raise ValueError(
            f"Unknown Target: {TARGET}\n"
            "Please choose one of the following\n"
            "accuracy, RT, dprime, drift, start")
# %%
