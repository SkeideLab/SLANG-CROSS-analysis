# %% [markdown]
# ## fMRI Group-level audiovisual coactivation (2nd-level)
#
# **Pipeline Overview**
# 1. === STEP 1 ===: Install packages
# 2. === STEP 2 ===: Set parameters
# 3. === STEP 3 ===: Compute audiovisual coactivation for each subject
# 4. === STEP 4 ===: Output the results



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
from nilearn.image import load_img, threshold_img
from nilearn.glm import threshold_stats_img


# %%
# 2. === STEP 2 ===: Set parameters
# -----------------------------------------------
MODEL          = 'glm'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = [
                'images_words', 
                'audios_words',
                ]
FWHM_SMOOTHING = 8.0 
HEMI           = 'left'
P_CORRECTION   = 0.05 
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


# Get the list of names with label from the atlas
atlas_img      = HO_ATLAS_MNI6.maps
atlas_labels   = HO_ATLAS_MNI6.labels
label_to_index = {name: i for i, name in enumerate(atlas_labels)}

# Get the label of ROIs
roi_indices = []
for group, regions in ROIs.items():
    for r in regions:
        if r in label_to_index:
            roi_indices.append(label_to_index[r])
roi_indices = np.array(roi_indices)




# %%
# 3. === STEP 3 ===: Compute audiovisual coactivation for each subject
# ---------------------------------------------------------------------
# ROIs
roi_names     = []
for group, regions in ROIs.items():
    for r in regions:
        roi_names.append(r)

# Subjects
subjects      = sorted(DERIV_DIR.glob(f"sub-*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]
subject_names = [p.name.replace('sub-', '') for p in subjects]



all_results   = []
# subject loop
for subject in subjects:
    sub_name  = subject.name
    glm_path  = subject / MODEL / SPACE / f'FWHM_{int(FWHM_SMOOTHING)}'
    folders   = [p for p in glm_path.rglob('*') if p.is_dir()]

    # run loop
    written_thr_lists          = []
    spoken_thr_lists           = []

    for folder in folders:
        written_beta_path          = folder / f'{CONTRASTS[0]}_z.nii.gz'
        spoken_beta_path           = folder / f'{CONTRASTS[1]}_z.nii.gz'

        # load z-map
        z_img_written = nib.load(str(written_beta_path))
        z_img_spoken  = nib.load(str(spoken_beta_path))
        
        # apply p-value threshold
        z_map_thr_written, threshold = threshold_stats_img(
            z_img_written,
            alpha=P_CORRECTION,
            height_control='fpr',
            two_sided=False,  # using two-sided test
        )
        z_map_thr_spoken, threshold = threshold_stats_img(
            z_img_spoken,
            alpha=P_CORRECTION,
            height_control='fpr',
            two_sided=False,  # using two-sided test
        )

        # store thr z-map
        z_thr_written = z_map_thr_written.get_fdata()
        z_thr_spoken  = z_map_thr_spoken.get_fdata()
        written_thr_lists.append(z_thr_written)
        spoken_thr_lists.append(z_thr_spoken)
        

    # roi loop
    for roi_name in roi_names:
        roi_data       = nib.load(
                MASK_DIR / f"{HEMI}_{roi_name}_pediatric_MNI.nii.gz"
            ).get_fdata().astype(bool)

        # count 
        n_runs = len(written_thr_lists)

        overlap_matrix = np.zeros((n_runs, n_runs), dtype=int)

        # Number of significant voxels for each run
        written_n_voxels = np.zeros(n_runs, dtype=int)
        spoken_n_voxels  = np.zeros(n_runs, dtype=int)

        # Count significant voxels within ROI for each run
        for i in range(n_runs):

            # Written
            written_sig = (written_thr_lists[i] > 0) & roi_data
            written_n_voxels[i] = np.sum(written_sig)

            # Spoken
            spoken_sig = (spoken_thr_lists[i] > 0) & roi_data
            spoken_n_voxels[i] = np.sum(spoken_sig)

        for i in range(n_runs):
            # Significant voxels in written map of subject i
            written_sig = (written_thr_lists[i] > 0) & roi_data
            written_n_voxels[i] = np.sum(written_sig)

            for j in range(n_runs):
                # Significant voxels in spoken map of subject j
                spoken_sig = (spoken_thr_lists[j] > 0) & roi_data

                # Number of voxels significant in both
                overlap_matrix[i, j] = np.sum(
                    written_sig & spoken_sig
                )
        # average off-diagonal cells
        off_diagonal = overlap_matrix[~np.eye(overlap_matrix.shape[0], dtype=bool)]
        mean_overlap = round(off_diagonal.mean())
        std_overlap  = off_diagonal.std()

        mean_written = round(written_n_voxels.mean())
        std_written  = written_n_voxels.std()

        mean_spoken = round(spoken_n_voxels.mean())
        std_spoken  = spoken_n_voxels.std()

        n_roi_voxels = np.sum(roi_data)
        percet_overlap = (mean_overlap / n_roi_voxels)*100
        percet_written = (mean_written / n_roi_voxels)*100
        percet_spoken = (mean_spoken / n_roi_voxels)*100

        # Store results
        all_results.append({
            "subject": sub_name,
            "roi": roi_name,

            "mean_overlap": mean_overlap,
            "std_overlap": std_overlap,
            "perc_overlap": percet_overlap,

            # Written activation extent
            "mean_written_voxels": mean_written,
            "std_written_voxels": std_written,
            "perc_written": percet_written,

            # Spoken activation extent
            "mean_spoken_voxels": mean_spoken,
            "std_spoken_voxels": std_spoken,
            "perc_spoken": percet_spoken,
                })

# %%
# 4. === STEP 4 ===: Output the results
# -----------------------------------
results_df = pd.DataFrame(all_results)
# save it as csv file
path      = OUT_DIR / 'multimodal' / f'{HEMI}_Coactivation_metrics_FWHM_{int(FWHM_SMOOTHING)}.csv'
results_df.to_csv(path, index=False)

