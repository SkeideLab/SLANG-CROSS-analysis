# %%
# ===  Load modules ===
# Load modules
import json
import os
import re
import zipfile
from pathlib import Path
from sys import executable
from helpers import submit_job
# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
DEMO_DIR     = ANALY_DIR / 'demographics'
SCRIPT_DIR   = ANALY_DIR / 'scripts'
LOG_DIR      = ANALY_DIR / 'logs'

# === Parameters ===
MODEL          = 'glm'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_words-images_pseudo'
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
FACTOR         = 3
EXC_SUBJECTS   = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434']
# List of the contrasts
"""             'images_words-images_pseudo' 
                'audios_words-audios_pseudo' 
                'images_words' 
                'audios_words'
                'images_pseudo'
                'audios_pseudo'
                'images_words+audios_words-images_pseudo+audios_pseudo' """

# %%
# Make sure that containers are available for the batch jobs
containers_prefix = ANALY_DIR / 'containers'

containers_dict = {
    'afni': containers_prefix / 'afni-25.3.02.sif'
    }

# Check that container files exist
for name, path in containers_dict.items():
    if not path.exists():
        raise FileNotFoundError(f"Container '{name}' not found at {path}")
    else:
        print(f"Found container '{name}' at {path}")

# %% list the beta nii.files
subjects      = sorted(DERIV_DIR.glob(f"sub-*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]
subject_names = [p.name.replace('sub-', '') for p in subjects]

# get the beta paths
beta_lists = []
afni_lists = []
for sub in subjects:
    path = sub / 'glm' / SPACE / f"FWHM_{int(FWHM_SMOOTHING)}" 
    beta_path = next(path.glob(f"{CONTRASTS}_beta.nii.gz"), None)
    afni_name = beta_path.name.replace('beta.nii.gz', 'beta_afni')
    afni_path = beta_path.parent / afni_name
    beta_lists.append(beta_path)
    afni_lists.append(afni_path)
beta_paths = [str(p) for p in beta_lists]
afni_paths = [str(p) for p in afni_lists]

# %% convert nii → afni dataset (.BRICK and .HEAD)
script = SCRIPT_DIR / 'anova_convert.sh'
script.chmod(script.stat().st_mode | 0o111) # allow permission
for beta_path, afni_path in zip(beta_paths, afni_paths):
    args = [script, ANALY_DIR, CONTRASTS, beta_path, afni_path]
    job_id = submit_job(args, cpus=1, mem=3200, job_name='anova', log_dir=LOG_DIR)

# %% conpute anova
script = SCRIPT_DIR / 'anova_oneway.sh'
script.chmod(script.stat().st_mode | 0o111) # allow permission
sub    = [p.name for p in subjects]
grade1 = [s for s in sub if s.startswith('sub-1')]
grade1_arg = ' '.join(grade1)
grade2 = [s for s in sub if s.startswith('sub-2')]
grade2_arg = ' '.join(grade2)
grade4 = [s for s in sub if s.startswith('sub-4')]
grade4_arg = ' '.join(grade4)

args   = [script, ANALY_DIR, CONTRASTS, SPACE, int(FWHM_SMOOTHING), {grade1_arg}, {grade2_arg}, {grade4_arg}, FACTOR]
job_id = submit_job(args, cpus=1, mem=3200, job_name='anova', log_dir=LOG_DIR)

# %%
