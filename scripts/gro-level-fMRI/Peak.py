# %% [markdown]
# ## fMRI Group-level Language RDM (2nd-level)
#
# **Pipeline Overview**
# 1. === STEP 1 ===: Install packages
# 2. === STEP 2 ===: Set parameters
# 3. === STEP 3 ===: visualize ROIs
# 4. === STEP 4 ===: Compute language RDM for each subject
# 5. === STEP 5 ===: Save the RDMs as csv file



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
FWHM_SMOOTHING = 5.0 # 6.0, 9.0, 12.0
HEMI           = 'left'
P_CORRECTION   = 0.01 
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

# get the ROI path
path     = MASK_DIR / f"{HEMI}_Temporal Occipital Fusiform Cortex_pediatric_MNI.nii.gz"
roi_img  = nib.load(path)
roi_mask = roi_img.get_fdata().astype(bool)

# %%
# 6. === STEP 6 ===: compute beta for each y-axis
# -----------------------------------------------
# Get voxel indices where mask is True
ijk_left = np.array(np.where(roi_mask)).T

# Convert voxel indices to MNI coordinates
xyz_left = nib.affines.apply_affine(roi_img.affine, ijk_left)

# get the y-axis
arr_sorted     = ijk_left[np.argsort(ijk_left[:, 1])]
y_axes         = np.unique(arr_sorted[:,1])
y_pairs        = y_axes.reshape(-1, 2)

arr_sorted_mni  = xyz_left[np.argsort(xyz_left[:, 1])]
y_axes_mni      = np.unique(arr_sorted_mni[:,1])
y_axes_combined = y_axes_mni.reshape(-1, 2).mean(axis=1)

all_results   = []

# %%
# subject loop
for subject in subjects:
    sub_name  = subject.name
    glm_path  = subject / MODEL / SPACE / f'FWHM_{int(FWHM_SMOOTHING)}'
    folders   = [p for p in glm_path.rglob('*') if p.is_dir()]

    written_beta_lists          = []
    spoken_beta_lists           = []

    for folder in folders:
        written_beta_path          = folder / f'{CONTRASTS[0]}_beta.nii.gz'
        spoken_beta_path           = folder / f'{CONTRASTS[1]}_beta.nii.gz'

        written_beta_lists.append(written_beta_path)
        spoken_beta_lists.append(spoken_beta_path)

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

        for i in range(n_runs):
            # Significant voxels in written map of subject i
            written_sig = (written_thr_lists[i] > 0) & roi_data

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
        # diagonal = np.diag(overlap_matrix)
        # mean_overlap = diagonal.mean()
        # std_overlap  = diagonal.std()
        

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        all_results.append({
            "subject": sub_name,
            "roi": roi_name,
            "mean_overlap": mean_overlap,
            "std_overlap": std_overlap,
        })

# %
# ----------------------------------------------------
# Convert to DataFrame and save
# ----------------------------------------------------
results_df = pd.DataFrame(all_results)

# save it as csv file
path      = OUT_DIR / 'multimodal' / f'{HEMI}_Coactivation_metrics_FWHM_{int(FWHM_SMOOTHING)}.csv'
results_df.to_csv(path, index=False)

# %%
