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
from nilearn import plotting
import nibabel as nib
import numpy as np
import templateflow.api as tflow
from scipy.stats import f
from nilearn import image
from nilearn.glm import cluster_level_inference, threshold_stats_img
# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
DEMO_DIR     = ANALY_DIR / 'demographics'
SCRIPT_DIR   = ANALY_DIR / 'scripts'
TEMP_DIR     = ANALY_DIR / 'templates'
LOG_DIR      = ANALY_DIR / 'logs'

# === Parameters ===
MODEL          = 'anova'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_pseudo'
MASK           = 'ventral' # none, ventral, fusiform, VWFA
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
FACTOR         = 3
P_CORRECTION   = 0.05
CLUSTER_SIZE   = 0
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
# === Make sure that containers are available for the batch jobs === 
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

# %% 
# === list the beta nii.files === 
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


# %% 
# === convert nii → afni dataset (.BRICK and .HEAD) === 
script = SCRIPT_DIR / 'nifti-afni.sh'
script.chmod(script.stat().st_mode | 0o111) # allow permission
for beta_path, afni_path in zip(beta_paths, afni_paths):
    args = [script, ANALY_DIR, CONTRASTS, beta_path, afni_path]
    job_id = submit_job(args, cpus=1, mem=32000, job_name='nifti-afni', log_dir=LOG_DIR)

# %% 
# === evaluate left-right flip === 
# script = SCRIPT_DIR / 'left_right_flip.sh'
# script.chmod(script.stat().st_mode | 0o111) # allow permission

# top_dir = Path('/ptmp/kazma/SLANG-CROSS-conversion')
# top_wri = OUT_DIR / 'flip'
# # for sub in subject_names:
#     # subject    = f"sub-{sub}"
# subject = "sub-117"
# args   = [script, ANALY_DIR, subject, top_dir, top_wri]
# job_id = submit_job(args, cpus=1, mem=32000, job_name='flipcheck', log_dir=LOG_DIR)

# %% 
# === compute 3dANOVA === 
script = SCRIPT_DIR / '3dANOVA.sh'
script.chmod(script.stat().st_mode | 0o111) # allow permission
outdir_anova = OUT_DIR / 'anova' / CONTRASTS
outdir_anova.mkdir(parents=True, exist_ok=True)

smoothing = f"FWHM_{int(FWHM_SMOOTHING)}" 
args   = [script, ANALY_DIR, CONTRASTS, outdir_anova, DERIV_DIR, SPACE, smoothing]
job_id = submit_job(args, cpus=1, mem=3200, job_name='3dANOVA', log_dir=LOG_DIR)


# %% 
# === convert afni → nii dataset (nii.gz) === 
script = SCRIPT_DIR / 'afni-nifti.sh'
f_name = 'onewayANOVA_Fstat.nii.gz'
script.chmod(script.stat().st_mode | 0o111) # allow permission
args   = [script, ANALY_DIR, CONTRASTS, outdir_anova, f_name]
job_id = submit_job(args, cpus=1, mem=3200, job_name='afni-nifti', log_dir=LOG_DIR)



# %% 
# === Thresholded f-map === 
outdir_anova = OUT_DIR / 'anova' / CONTRASTS
f_name = 'onewayANOVA_Fstat.nii.gz'
path  = outdir_anova / f_name
f_img = nib.load(path)
data  = f_img.get_fdata()

# compute the critical f-value
alpha = P_CORRECTION
df_n  = FACTOR - 1  # (3 groups - 1)
# count the sample size
n1    = sum(s.startswith("1") for s in subject_names)
n2    = sum(s.startswith("2") for s in subject_names)
n4    = sum(s.startswith("4") for s in subject_names)
df_d  = (n1 + n2 + n4) - FACTOR  # (Total subjects - 3 groups)
critical_f = f.ppf(q=1 - alpha, dfn=df_n, dfd=df_d)
print(f"The critical F-value for alpha {alpha} is: {critical_f:.4f}")

thresholded_f_map, threshold = threshold_stats_img(
    f_img,
    alpha=None,
    threshold=critical_f,
    height_control=None,
    cluster_threshold=CLUSTER_SIZE,
    two_sided=False
)

# save the map 
file_name   = f"p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_f-map.nii.gz"
output_path = outdir_anova / file_name
thresholded_f_map.to_filename(output_path)



# %%
# === Visualize the thresholded f-map ===
if '-' in CONTRASTS:
    contrast = CONTRASTS.replace('-', ' > ')
else:
    contrast = CONTRASTS

if MASK=='ventral':
    left_mask_fn = TEMP_DIR / 'mask' / 'left_ventral_mask.nii.gz'
    right_mask_fn = TEMP_DIR / 'mask' / 'right_ventral_mask.nii.gz'
    left_mask_img = nib.load(left_mask_fn)
    right_mask_img = nib.load(right_mask_fn)
    # Combine masks (union)
    ventral_mask_data = (
        (left_mask_img.get_fdata() == 1) |
        (right_mask_img.get_fdata() == 1)
    ).astype(np.uint8)
    ventral_mask_img = nib.Nifti1Image(
        ventral_mask_data,
        left_mask_img.affine
    )
    thresholded_f_map = image.math_img(
        "stat * mask",
        stat=thresholded_f_map,
        mask=ventral_mask_img
    )
    # Count non-zero, finite voxels
    xx = thresholded_f_map.get_fdata()
    n_voxels = np.sum((xx != 0) & np.isfinite(xx))
    print(f"Number of suprathreshold voxels (ventral mask): {n_voxels}")



# === glass plot === 
display = plotting.plot_glass_brain(
    thresholded_f_map,
    title=f"{contrast} (cluster-thr > {CLUSTER_SIZE}, p < {P_CORRECTION})",
    threshold=threshold,
    colorbar=True,
    display_mode='lyrz',  # show left, sagittal, coronal, axial views
    plot_abs=False,     # keep sign info if you have positive/negative values
    black_bg=False,
    vmin=0
)
display.add_contours(
    thresholded_f_map, 
    levels=[threshold],       # level(s) to contour
    colors='black',             # color of the outline
    linewidths=0.5
)

# create output dir
dir = f"{FIG_DIR}/{MODEL}"
os.makedirs(dir, exist_ok=True)

display.savefig(f"{dir}/f-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}_cls>{CLUSTER_SIZE}_glass_{MASK}.png", dpi=300)
plotting.show()


# === MNI-pediatric === 
# Z-coordinates you want to visualize
if 'images' in CONTRASTS and 'audios' in CONTRASTS:
    z_coords = np.arange(0, 20, 2)
elif 'images' in CONTRASTS:
    z_coords = np.arange(-15, -4, 1)  # from -15 to -5 inclusive
elif 'audios' in CONTRASTS:
    z_coords = np.arange(-5, 10, 2) # from -5 to 5

# configure the figure
plotting_config = {
    "display_mode": "z",
    "cut_coords": z_coords,
    "draw_cross": False,
    "vmax": 8,
    "vmin": 0,
    "cmap": "hot",
}
# plot the uncorrrected p-value figure 
display = plotting.plot_stat_map(
    thresholded_f_map,
    title=f"{contrast} (cluster-thr > {CLUSTER_SIZE}, p < {P_CORRECTION})",
    bg_img=TEMPLATE,
    threshold=threshold,
    **plotting_config,
)
display.savefig(f"{dir}/f-maps_{CONTRASTS}_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}_cls>{CLUSTER_SIZE}_MNI-ped_{MASK}.png", dpi=300)

# %%
