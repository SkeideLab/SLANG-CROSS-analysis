#!/usr/bin/env python
# %% [markdown]
# ## fMRI univariate analysis
#
# Classical GLM analysis in SPM using Nipype for an event-related design
# with 4 conditions: visual word, visual pseudoword, audio word, audio pseudoword.

# %%
# ===  Load modules ===
import pandas as pd
from pathlib import Path
from nilearn.glm.first_level import (FirstLevelModel,
                                    make_first_level_design_matrix)
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn import plotting
from nilearn.image import load_img, mean_img, binarize_img
from nilearn.plotting import plot_design_matrix, plot_stat_map
from bids import BIDSLayout
import nibabel as nib
import re
import os
import numpy as np


# %%
# === Paths ===
BIDS_DIR       = Path('/ptmp/kazma/SLANG-CROSS-conversion')
ANALY_DIR      = Path('/ptmp/kazma/SLANG-CROSS-analysis')
FMRIPRE_DIR    = BIDS_DIR / 'derivatives/fmriprep'
DERIV_DIR      = ANALY_DIR / 'derivatives'  # where to store outputs
PYBIDS_DIR     = DERIV_DIR / 'pybids'

# === Inpupt parameters: First-level GLM ===
SPACE          = 'MNI152NLin2009cAsym' # T1w, fsnative, MNI152NLin2009cAsym
SUBJECT        = '101'  # subject label
SESSION        = '01'
TASK           = 'language'
BLOCKWISE      = False
TR             = 2.0
SMOOTHING_FWHM = 5.0
HRF_MODEL      = 'spm'
DRIFT_MODEL    = 'cosine'
HIGH_PASS      = 0.01
STANDARDIZE    = 'psc' 
SAVE_RESIDUALS = True
N_JOBS         = 8
CONDITION      = ['images_words', 'images_pseudo', 'audios_words', 'audios_pseudo']
MOV_COLS       = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
COMPCOR_COLS   = ['a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02',
                    'a_comp_cor_03', 'a_comp_cor_04', 'a_comp_cor_05']
CONTRASTS      = ['images_words - images_pseudo', 
                  'audios_words - audios_pseudo', 
                  'images_words', 
                  'audios_words',
                  'images_pseudo',
                  'audios_pseudo',
                  '(images_words + audios_words) - (images_pseudo + audios_pseudo)']
EXC_SUBJECTS   = ['108', '113', '116', '120', '122', '125', '126', '206', '220', '227', '405', '410', '421','424', '427', '430']

# %%
# === import user-defined functions ====
def load_mask_img(layout, subject, session, task, space):
    """Loads the preprocessed brain mask for a given subject and session."""

    mask_files = layout.get(subject=subject, session=session, task=task,
                            space=space, desc='brain', suffix='mask',
                            extension='nii.gz')
    assert len(mask_files) == 1
    mask_file = mask_files[0]

    return load_img(mask_file)


def load_events(layout, subject, session, task):
    """Loads task events for a given subject and session as a DataFrame."""

    events_files = layout.get(subject=subject, session=session, task=task,
                              suffix='events', extension='tsv')
    assert len(events_files) == 1
    events_file = events_files[0]

    return pd.read_csv(events_file, sep='\t')


def load_func_img(layout, subject, session, task, space):
    """Loads preprocessed fMRI data for a given subject and session."""

    func_files = layout.get(subject=subject, session=session, task=task,
                            space=space, desc='preproc', suffix='bold',
                            extension='nii.gz')
    assert len(func_files) == 1
    func_file = func_files[0]

    return load_img(func_file)


def make_frame_times(layout, func_img):
    """Creates a list of frame times for a given fMRI image."""

    n_scans = func_img.shape[-1]
    tr = layout.get_tr()

    return tr * (np.arange(n_scans) + 0.5)


def get_confounds(layout, subject, session, task, fd_threshold=0.5):
    """Loads a common set of confounds for a given subject and session.

    These are:
    * Six head motion parameters (translations and rotations)
    * Six top aCompCor components
    * Cosine regressors for high-pass filtering
    * Spike regressors for non-steady-state volumes at the beginning of scans
    * Spike regressors for outlier volumes based on framewise displacement
    """

    confounds_files = layout.get(subject=subject, session=session, task=task,
                                 desc='confounds', suffix='timeseries',
                                 extension='tsv')
    assert len(confounds_files) == 1
    confounds_file = confounds_files[0]
    confounds = pd.read_csv(confounds_file, sep='\t')

    hmp_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

    compcor_cols = ['a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02',
                    'a_comp_cor_03', 'a_comp_cor_04', 'a_comp_cor_05']

    cosine_cols = [col for col in confounds if col.startswith('cosine')]

    non_steady_cols = [col for col in confounds
                       if col.startswith('non_steady_state_outlier')]
    n_non_steady = len(non_steady_cols)
    n_volumes = len(confounds)
    perc_non_steady = n_non_steady / n_volumes
    print(f'Found {n_non_steady} non-steady-state volumes ' +
          f'({perc_non_steady * 100:.1f}%) for subject {subject}, session {session}')

    confounds, outlier_cols = add_outlier_regressors(confounds, fd_threshold)

    n_outliers = len(outlier_cols)
    perc_outliers = n_outliers / n_volumes
    print(f'Found {n_outliers} outlier volumes ({perc_outliers * 100:.1f}%) ' +
          f'for subject {subject}, session {session}')

    cols = hmp_cols + compcor_cols + cosine_cols + non_steady_cols + outlier_cols
    confounds = confounds[cols]

    return confounds, perc_non_steady, perc_outliers


def add_outlier_regressors(confounds, fd_threshold=0.5):
    """Adds outlier regressors based on framewise displacement to confounds."""

    fd = confounds['framewise_displacement']
    outlier_ixs = np.where(fd > fd_threshold)[0]
    outliers = np.zeros((len(fd), len(outlier_ixs)))
    outliers[outlier_ixs, np.arange(len(outlier_ixs))] = 1
    outlier_cols = [f'fd_outlier{i}' for i in range(len(outlier_ixs))]
    outliers = pd.DataFrame(outliers, columns=outlier_cols)
    confounds = pd.concat([confounds, outliers], axis=1)

    return confounds, outlier_cols

def combine_save_mask_imgs(mask_imgs, output_dir, task, space,
                           perc_threshold=0.5):
    """Combines brain masks across subjects and sessions and saves the result.

    Only voxels that are present in at least `perc_threshold` of all masks are
    included in the final mask."""

    mask_img = combine_mask_imgs(mask_imgs, perc_threshold)

    mask_file = save_img(mask_img, output_dir, task, space,
                         desc='brain', suffix='mask')

    return mask_img, mask_file


def combine_mask_imgs(mask_imgs, perc_threshold=0.5):
    """Combines brain masks across subjects and sessions.

    Only voxels that are present in at least `perc_threshold` of all masks are
    included in the final mask."""

    return binarize_img(mean_img(mask_imgs), threshold=perc_threshold)

def save_img(img, output_dir, task, space, desc, suffix,
             subject=None, session=None):
    """Saves a NIfTI image to a file in the output directory."""

    filename = f'task-{task}_space-{space}_desc-{desc}_{suffix}.nii.gz'

    if session:
        filename = f'ses-{session}_{filename}'

    if subject:
        filename = f'sub-{subject}_{filename}'

    file = output_dir / filename
    img.to_filename(file)

    return file


# %%
## === Prepare First-level GLM ===
layout   = BIDSLayout(BIDS_DIR, derivatives=FMRIPRE_DIR, database_path=PYBIDS_DIR, validate=True, reset_database=True)

# Get all subject list
subjects = layout.get_subjects()  # returns a list like ['01', '02', '03', ...]

for subject in subjects:
    if subject in EXC_SUBJECTS:
        continue
    
    print(f"\nProcessing subject {subject}...\n")

    # Event files
    events_files = layout.get(subject=subject, session=SESSION, task=TASK,
                                suffix='events', extension='tsv')
    # Mask files
    mask_files   = layout.get(subject=subject, session=SESSION, task=TASK,
                                space=SPACE, desc='brain', suffix='mask',
                                extension='nii.gz')
    # Functional runs
    func_files   = layout.get(subject=subject, session=SESSION, task=TASK,
                                space=SPACE, desc='preproc', suffix='bold',
                                extension='nii.gz')

    # confounds
    con_files    = layout.get(subject=subject, session=SESSION, task=TASK,
                                    desc='confounds', suffix='timeseries',
                                    extension='tsv')

    # Create output folder
    out_dir = Path(f"{DERIV_DIR}/sub-{subject}/glm/{SPACE}")
    os.makedirs(out_dir, exist_ok=True)

    # create a brain mask combined across runs
    mask_imgs = [f.path for f in mask_files]
    x = combine_save_mask_imgs(mask_imgs, out_dir, TASK, SPACE,
                            perc_threshold=0.5)
    masks_sum = load_img(str(x[1]))

    # === Prepare First-level GLM for a run ===
    # contrast = 'images_words - images_pseudo'
    for contrast in CONTRASTS: 
        print(f"\nTarget contrast: {contrast}...\n")
        contrast_imgs = [] # store effect size for each run
        variance_imgs = [] # store effect variance for each run

        # n_run = 0
        for n_run in range(len(func_files)):
            print(f"\nProcessing run {n_run+1}/{len(func_files)}...\n")

            # functional file
            func_img    = load_img(func_files[n_run].path)

            # frame time
            frame_times = make_frame_times(layout, func_img)

            # event file
            events = pd.read_csv(events_files[n_run].path, sep='\t')
            trial_types_to_model = CONDITION # Keep only the trial types you want as regressors
            events_filtered = events[events['trial_type'].isin(trial_types_to_model)]

            # confound file
            confounds = pd.read_csv(con_files[n_run].path, sep='\t')
            non_steady_cols = [col for col in confounds
                                if col.startswith('non_steady_state_outlier')]
            outlier_cols    = [col for col in confounds
                                if col.startswith('motion_outlier')]
            comp_cols       = [col for col in confounds
                                if col.startswith('a_comp')]
            comp_cols       = comp_cols[:len(COMPCOR_COLS)]
            if len(comp_cols) < len(COMPCOR_COLS):
                print(f"\nOnly found {len(comp_cols)} a_comp columns, expected 6. Continuing anyway.\n")

            cols = MOV_COLS + comp_cols + non_steady_cols + outlier_cols
            confounds = confounds[cols]

            # mask file
            mask_img = load_img(mask_files[n_run].path)


            # generate design matrix
            design_matrix = make_first_level_design_matrix(frame_times,
                                                            events=events_filtered,
                                                            drift_model=DRIFT_MODEL, # high-pass filter (~128s)
                                                            high_pass=HIGH_PASS, 
                                                            hrf_model=HRF_MODEL,   
                                                            add_regs=confounds,
                                                            add_reg_names=confounds.columns.tolist()) # extra regressors

            # save the figure of design matrix
            """ ax  = plot_design_matrix(design_matrix)
            fig = ax.get_figure()
            fig.savefig('design_matrix.png', dpi=300)  # PNG with 300 dpi """

            # === Initialize First-level GLM for a run ===
            fmri_glm = FirstLevelModel(  
                standardize=STANDARDIZE, # percent signal change
                smoothing_fwhm=SMOOTHING_FWHM,
                n_jobs=N_JOBS,
                mask_img=mask_img,
                minimize_memory=False
            )


            # 
            # === Run GLM ===
            fmri_glm.fit(run_imgs=func_img, design_matrices=[design_matrix])

            # 
            # === Summary of the GLM ===
            summary_stats = fmri_glm.compute_contrast(contrast, output_type='all')
            contrast_imgs.append(summary_stats["effect_size"])
            variance_imgs.append(summary_stats["effect_variance"])

            # LOOP END

        #
        # === subject level effect size ===
        fixed_fx_contrast, fixed_fx_variance, fixed_fx_stat = compute_fixed_effects(
            contrast_imgs, variance_imgs, mask=masks_sum)

        # save in the outdir
        contrast_no_space = contrast.replace(' ', '').replace('(', '').replace(')', '')
        output_beta       = Path(f"{out_dir}/{contrast_no_space}_beta.nii.gz")
        output_tstat      = Path(f"{out_dir}/{contrast_no_space}_tstat.nii.gz")
        output_var        = Path(f"{out_dir}/{contrast_no_space}_var.nii.gz")

        fixed_fx_contrast.to_filename(output_beta)
        fixed_fx_variance.to_filename(output_tstat)
        fixed_fx_stat.to_filename(output_var)


