# %%
# ===  Load modules ===
from nilearn.plotting import plot_glass_brain
from nilearn import plotting, image
from pathlib import Path
import nibabel as nib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from ddm import Model, Fittable
from ddm.models import DriftConstant, NoiseConstant, BoundConstant, ICPoint, OverlayNonDecision
from ddm import Sample
# %%
# ===  Parameters ===
THRESHOLD    = 2
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
SUBJECT      = 'sub-101'
MODEL        = 'glm'
SPACE        = 'MNI152NLin2009cAsym'
MODALITY     = 'audio' # all, visual, audio
EXC_SUBJECTS = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422', '424', '427', '430', '434']

# make a figure directory
FIG_DIR.mkdir(parents=True, exist_ok=True)


# %% Plot
# === Brain ===
""" nii_path = DERIV_DIR / SUBJECT / MODEL / SPACE / 'images_words-images_pseudo_beta.nii.gz'
image    = nib.load(nii_path)

plotting_config = {
    "colorbar": True,
    "cmap": "inferno",
}

display = plotting.plot_glass_brain(
    image,
    title=SUBJECT,
    threshold=THRESHOLD,
    display_mode="lzr",
    **plotting_config,
)
display.savefig(f"{FIG_DIR}/glass_brain.png", dpi=300) """


# %% Plot
# === Behavior ===
subjects = sorted([d.name for d in DERIV_DIR.iterdir() if d.is_dir() and d.name.startswith('sub')])
exclude = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects_filtered = [s for s in subjects if s not in exclude]
# List to store each subject's DataFrame
all_dfs = []

for subject in subjects_filtered:

    path = DERIV_DIR / subject / 'behavior/accuracy_summary.csv'
    df = pd.read_csv(path)
    df['subject'] = subject
    all_dfs.append(df)

# Concatenate all DataFrames into one
df          = pd.concat(all_dfs, ignore_index=True)
df['grade'] = df['subject'].str.replace('sub-','').astype(str).str[0].astype(int)
df['run']   = df['file_name'].str.extract(r'run-(\d+)_events').astype(int)
if MODALITY == 'all':
    df          = df[df['accuracy_all'] != -1]
    df          = df[df['accuracy_all'] != 0]
elif MODALITY == 'visual':
    df          = df[df['accuracy_visual'] != -1]
    df          = df[df['accuracy_visual'] != 0]
elif MODALITY == 'audio':
    df          = df[df['accuracy_audio'] != -1]
    df          = df[df['accuracy_audio'] != 0]
else:
    raise ValueError(f"Unknown MODALITY: {MODALITY}")

# test whether the average is above chance
acc_str = f"accuracy_{MODALITY}"
rt_str = f"RT_{MODALITY}"
dprime_str=f"dprime_{MODALITY}"
c_str=f"c_{MODALITY}"
mean = df.groupby('subject').agg(
    mean_accuracy=(acc_str, 'mean'),
    mean_RT=(rt_str, 'mean'),
    mean_dprime=(dprime_str, 'mean'),
    mean_c=(c_str, 'mean'),
    grade=('grade', 'first')
).reset_index()

# %%
# === One-sample T-test ===
# create an empty list to store results
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

# === One-way ANOVA ===
# accuracy
groups = [mean.loc[mean['grade'] == g, 'mean_accuracy'] for g in mean['grade'].unique()]
f_stat, p_value = stats.f_oneway(*groups)
print("\nOne-way ANOVA")
print("Variable: Accuracy")
print(f"Modality: {MODALITY}")
print(f"F = {f_stat:.3f}, p = {p_value:.4f}")
tukey = pairwise_tukeyhsd(endog=mean['mean_accuracy'], groups=mean['grade'], alpha=0.05)
print(tukey)

# reaction time
groups = [mean.loc[mean['grade'] == g, 'mean_RT'] for g in mean['grade'].unique()]
f_stat, p_value = stats.f_oneway(*groups)
print("\nOne-way ANOVA")
print("Variable: RT")
print(f"Modality: {MODALITY}")
print(f"F = {f_stat:.3f}, p = {p_value:.4f}")
tukey = pairwise_tukeyhsd(endog=mean['mean_RT'], groups=mean['grade'], alpha=0.05)
print(tukey)

# d prime
groups = [mean.loc[mean['grade'] == g, 'mean_dprime'] for g in mean['grade'].unique()]
f_stat, p_value = stats.f_oneway(*groups)
print("\nOne-way ANOVA")
print("Variable: dprime")
print(f"Modality: {MODALITY}")
print(f"F = {f_stat:.3f}, p = {p_value:.4f}")
tukey = pairwise_tukeyhsd(endog=mean['mean_dprime'], groups=mean['grade'], alpha=0.05)
print(tukey)

# %% 
# === Scatter plot ===
# === Accuracy ===
# Set a nice theme
sns.set(style="whitegrid")
# Bar plot with grade on x-axis, accuracy on y-axis, hue=run
plt.figure(figsize=(8,6))
sns.stripplot(
    data=df,
    x='grade', 
    y=acc_str, 
    hue='run',
    jitter=True,       # spreads dots horizontally for visibility
    dodge=True,        # separates runs side-by-side
    palette='Set2',
    size=6,            # size of the dots
)

plt.title('Accuracy by Grade and Run')
plt.ylabel('Accuracy (%)')
plt.xlabel('Grade')
# plt.ylim(0,100)
plt.legend(title='Run', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig(f"{FIG_DIR}/{acc_str}_scatter.png", dpi=300, bbox_inches='tight')

# === RT ===
sns.set(style="whitegrid")
# Bar plot with grade on x-axis, accuracy on y-axis, hue=run
plt.figure(figsize=(8,6))
sns.stripplot(
    data=df,
    x='grade', 
    y=rt_str, 
    hue='run',
    jitter=True,       # spreads dots horizontally for visibility
    dodge=True,        # separates runs side-by-side
    palette='Set2',
    size=6,            # size of the dots
)

plt.title('Reaction Time by Grade and Run')
plt.ylabel('Reaction time (s)')
plt.xlabel('Grade')
# plt.ylim(0,4)
plt.legend(title='Run', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig(f"{FIG_DIR}/{rt_str}_scatter.png", dpi=300, bbox_inches='tight')


# === dprime ===
sns.set(style="whitegrid")
# Bar plot with grade on x-axis, accuracy on y-axis, hue=run
plt.figure(figsize=(8,6))
sns.stripplot(
    data=df,
    x='grade', 
    y=dprime_str, 
    hue='run',
    jitter=True,       # spreads dots horizontally for visibility
    dodge=True,        # separates runs side-by-side
    palette='Set2',
    size=6,            # size of the dots
)

plt.title("Sensitivity (d') by Grade and Run")
plt.ylabel("d'")
plt.xlabel('Grade')
# plt.ylim(0,4)
plt.legend(title='Run', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig(f"{FIG_DIR}/{dprime_str}_scatter.png", dpi=300, bbox_inches='tight')


# %% 
# === Violin plot ===
# === Accuracy ===
plt.figure(figsize=(8,6))
sns.violinplot(
    data=mean,
    x='grade',
    y='mean_accuracy',
    inner='quartile',
    split=False,
    palette='Set2',
    cut=0,
    legend=False,
    alpha=0.4
)

plt.title('Accuracy Distribution by Grade')
plt.xlabel('Grade')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.yticks([0, 25, 50, 75, 100])
# Add horizontal line at y=50
plt.axhline(y=50, color='red', linestyle='--', linewidth=1.5)  # dashed red line
# plt.tight_layout()
plt.grid(False)
plt.savefig(f"{FIG_DIR}/{acc_str}_violin.png", dpi=300, bbox_inches='tight')
plt.show()

# === RT ===
plt.figure(figsize=(8,6))
sns.violinplot(
    data=mean,
    x='grade',
    y='mean_RT',
    inner='quartile',
    split=False,
    palette='Set2',
    cut=0,
    legend=False,
    alpha=0.4
)

plt.title('Reaction Time Distribution by Grade')
plt.xlabel('Grade')
plt.ylabel('Reaction Time (s)')
plt.ylim(0, 4)
plt.yticks([0, 1, 2, 3, 4])
# plt.tight_layout()
plt.grid(False)
plt.savefig(f"{FIG_DIR}/{rt_str}_violin.png", dpi=300, bbox_inches='tight')
plt.show()


# === dprime ===
plt.figure(figsize=(8,6))
sns.violinplot(
    data=mean,
    x='grade',
    y='mean_dprime',
    inner='quartile',
    split=False,
    palette='Set2',
    cut=0,
    legend=False,
    alpha=0.4
)

plt.title("Sensitivity (d') Distribution by Grade")
plt.xlabel('Grade')
plt.ylabel("d'")
# plt.ylim(0, 4)
# plt.tight_layout()
plt.grid(False)
plt.savefig(f"{FIG_DIR}/{dprime_str}_violin.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
