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

# %% 2. === STEP 2 ===: Extract all the availoabel beta maps for each rusubject
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
# for subject in subjects:
subject = subjects[21]
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

roi_data = nib.load(
    MASK_DIR / f"left_{roi_names[7]}_pediatric_MNI.nii.gz"
).get_fdata().astype(bool)

vector = np.vstack([
    nib.load(p).get_fdata()[roi_data].flatten()
    for p in combined
])
stds = np.std(vector, axis=1)
idx  = np.where(stds == 0)[0]
bad_paths = [combined[i] for i in idx]
bad_runs = {p.parts[-2] for p in bad_paths}  # extracts "run-08"
filtered_paths = [
    p for p in combined
    if p.parts[-2] not in bad_runs
]

# %
# 5. === STEP 5 ===: Representational Dissimilarity Matrix (written vs spoken)
corr_matrices = 1 - np.corrcoef(vector)


roi_target = 'Inferior Frontal Gyrus, pars opercularis'
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