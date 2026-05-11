# %% [markdown]
# ## fMRI Group-level Language RDM (2nd-level)
#
# **Pipeline Overview**
# 1. === STEP 1 ===: Install packages
# 2. === STEP 2 ===: Set parameters
# 3. === STEP 3 ===: Compute language RDM for each subject
# 4. === STEP 4 ===: Save the RDMs as csv file



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
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
HEMI           = 'right' 
EXC_SUBJECTS   = [
                '108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', 
                '201', '205', '206', '208', '220', '225', '226', '227', 
                '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434',
                ]
HO_ATLAS_MNI6  = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm') # Harvard-Oxford MNI6Asym
atlas_labels   = HO_ATLAS_MNI6.lut



# %%
# 4. === STEP 4 ===: Compute language RDM for each subject
# -----------------------------------------------
# ROIs
roi_names     = [label for label in atlas_labels['name'] if label != 'Background']

# Subjects
subjects      = sorted(DERIV_DIR.glob(f"sub-*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]
subject_names = [p.name.replace('sub-', '') for p in subjects]

all_results   = {}
# subject loop
for subject in subjects:
    # subject = subjects[0]
    glm_path  = subject / MODEL / SPACE / f'FWHM_{int(FWHM_SMOOTHING)}'
    folders   = [p for p in glm_path.rglob('*') if p.is_dir()]

    # run loop
    written_beta_lists          = []
    spoken_beta_lists           = []
    written_pseudo_beta_lists   = []
    spoken_pseudo_beta_lists    = []
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

    # Pick ROi and vectorize 
    vector             = {}
    for roi_name in roi_names:
        roi_data       = nib.load(
            MASK_DIR / f"{HEMI}_{roi_name}_pediatric_MNI.nii.gz"
        ).get_fdata().astype(bool)

        vec            = np.vstack([
            nib.load(p).get_fdata()[roi_data].flatten()
            for p in combined
        ])
        stds           = np.std(vec, axis=1)
        idx            = np.where(stds==0)[0]
        bad_paths      = [combined[i] for i in idx]
        bad_runs       = {p.parts[-2] for p in bad_paths}  # extracts "run-08"
        filtered_paths = [
            p for p in combined
            if p.parts[-2] not in bad_runs
        ]

        # Skip if nothing left after filtering
        if not filtered_paths:
            print(f"Skipping {roi_name}: no valid paths after filtering")
            continue
        
        vector[roi_name] = np.vstack([
            nib.load(p).get_fdata()[roi_data].flatten()
            for p in filtered_paths
        ])
    
    # Representational Dissimilarity Matrix (written vs spoken)
    corr_matrices = {
        roi: 1 - np.corrcoef(data)
        for roi, data in vector.items()
    }

    # Visualize RDM
    roi_target = roi_names[0]
    corr       = corr_matrices[roi_target]

    plt.imshow(corr, vmin=0, vmax=2)
    cbar       = plt.colorbar(label='Dissimilarity (1 - r)')
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['0', '1', '2'])
    # ticks starting from 1
    n          = corr.shape[0]
    # repeating 1 – max(run) 
    labels = np.tile(np.arange(1, n//len(CONTRASTS) + 1), len(CONTRASTS))
    plt.title(roi_target, fontsize=14)
    plt.xticks(range(n), labels)
    plt.yticks(range(n), labels)
    plt.xlabel('Run', fontsize=11)
    plt.ylabel('Run', fontsize=11)

    """ # save the figure
    roi_path = FIG_DIR / 'multimodal'
    roi_path.mkdir(exist_ok=True, parents=True)
    path     = roi_path / "RDM.pdf"
    plt.savefig(
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
        n_runs            = int(corr.shape[0] / 6)

        word              = range(1, n_runs + 1)
        spoken            = range(n_runs + 1, 2 * n_runs + 1)

        base_multi_index  = []

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


        results[region]    = {
            "word_multi": 1 - np.mean([corr[i, j] for i, j in idx_word_multi]),
            "pseudo_multi": 1 - np.mean([corr[i, j] for i, j in idx_pseudo_multi]),
            "semantic_multi": 1 - np.mean([corr[i, j] for i, j in idx_semantic_multi]),
        }
    print(f"{subject.name} is done")
    all_results[subject.name] = results



# %
# 4. === STEP 4 ===: Save the RDMs as csv file
# -----------------------------------------------

# convert to pandas dataframe
rows        = []
for subject, rois in all_results.items():
    for roi, metrics in rois.items():
        row = {
            "subject": subject,
            "roi": roi
        }
        row.update(metrics)
        rows.append(row)
df          = pd.DataFrame(rows)
df['grade'] = df['subject'].str.extract(r'-(\d+)').astype(int) // 100

# read RT
results       = []
for sub in subjects:
    task_path = sub / 'behavior' / 'accuracy_summary.csv'
    task_df   = pd.read_csv(task_path)
    mean_rt   = task_df['RT_all'].mean()
    # Store results as a dict
    results.append({
        'subject': sub.name,
        'rt': mean_rt,

    })
task_df       = pd.DataFrame(results)

# combine both RDM metrics and RT
df_merged = df.merge(
    task_df[["subject", "rt"]],
    on    = "subject",
    how   = "left"
)

# save it as csv file
path      = OUT_DIR / 'multimodal' / f'{HEMI}_RDM_metrics_supplements.csv'
df_merged.to_csv(path, index=False)

# %%
