# %%
# ===  Load modules ===
from nilearn.plotting import plot_glass_brain
from nilearn import plotting, image
from pathlib import Path
import nibabel as nib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
# ===  Parameters ===
THRESHOLD = 2
ANALY_DIR = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR = ANALY_DIR / 'derivatives'
SUBJECT   = 'sub-101'
MODEL     = 'glm'
SPACE     = 'MNI152NLin2009cAsym'

# %% Plot
# === Brain ===
nii_path = DERIV_DIR / SUBJECT / MODEL / SPACE / 'images_words-images_pseudo_beta.nii.gz'
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
display.savefig("glass_brain.png", dpi=300)

# %%
# === Behavior ===
subjects = sorted([d.name for d in DERIV_DIR.iterdir() if d.is_dir() and d.name.startswith('sub')])

# List to store each subject's DataFrame
all_dfs = []

for subject in subjects:

    path = DERIV_DIR / subject / 'behavior/accuracy_summary.csv'
    df = pd.read_csv(path)
    df['subject'] = subject
    all_dfs.append(df)

# Concatenate all DataFrames into one
df          = pd.concat(all_dfs, ignore_index=True)
df['grade'] = df['subject'].str.replace('sub-','').astype(str).str[0].astype(int)
df['run']   = df['file_name'].str.extract(r'run-(\d+)_events').astype(int)

# Plot
# Set a nice theme
sns.set(style="whitegrid")
# Bar plot with grade on x-axis, accuracy on y-axis, hue=run
plt.figure(figsize=(8,6))
sns.stripplot(
    data=df,
    x='grade', 
    y='accuracy', 
    hue='run',
    jitter=True,       # spreads dots horizontally for visibility
    dodge=True,        # separates runs side-by-side
    palette='Set2',
    size=6,            # size of the dots
)

plt.title('Accuracy by Grade and Run')
plt.ylabel('Accuracy (%)')
plt.xlabel('Grade')
plt.ylim(0,100)
plt.legend(title='Run', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig('accuracy_by_grade_run.png', dpi=300, bbox_inches='tight')
# %%
