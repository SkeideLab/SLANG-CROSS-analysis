# %% ===  Load modules ===
from nilearn.plotting import plot_glass_brain, plot_stat_map, show
from nilearn.plotting import plot_stat_map
from nilearn import plotting, image, datasets
from pathlib import Path
import templateflow.api as tflow
import os

# Limit threads to avoid OpenBLAS / kernel crashes
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"
import ants
import numpy as np
import nibabel as nib
from scipy.ndimage import label, find_objects
import pandas as pd

# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
DEMO_DIR     = ANALY_DIR / 'demographics'
TEMP_DIR     = ANALY_DIR / 'templates'
# === Parameters ===
MODEL          = 'glm'
METHOD         = 'parametric' # 'parametric' 'non-parametric'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_pseudo'
GRADE          = '4' # 1, 2, 4 or all
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
CORRECTION     = 'fdr' # fdr, fpr 
P_CORRECTION   = 0.05 # 0.001, 0.05
CLUSTER_SIZE   = 100
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

""" # convert the VWFA coordinate in adult MNI for pediatrci MNI
adultMNI  = np.array(MNI_VWFA)
scale     = np.array([1.21988, 1.23510, 1.28654])
ped_coord = adultMNI / scale """

# %% =========== convert Pediatric to Adult
# adult MNI
MNI_2mm        = datasets.load_mni152_template(resolution=2)
MNI_path       = ANALY_DIR / 'templates' / 'adult_MNI_2mm'
MNI_path.mkdir(parents=True, exist_ok=True)
MNI_filepath = MNI_path / 'MNI152_2mm.nii.gz'
MNI_2mm.to_filename(MNI_filepath)

fixed = ants.image_read(str(MNI_filepath))

# T1w pediatric space
moving = ants.image_read(str(TEMPLATE))

# nonlinear registration
reg = ants.registration(
    fixed  = fixed,
    moving = moving,
    type_of_transform = "SyN"   # recommended
)

if MODEL=='glm':
    path      = OUT_DIR / MODEL / SPACE / CONTRASTS
    file_name = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
elif MODEL=='anova':
    path      = OUT_DIR / MODEL / CONTRASTS
    file_name = f"p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_f-map.nii.gz"

filepath  = path / file_name
data_img  = nib.load(filepath)
data      = data_img.get_fdata()

# Find the minimum positive value
min_positive = data[data > 0].min()

# find the MNI coordinates of a cluster
cluster_mask = data != 0

# Label connected clusters
labeled_array, num_clusters = label(cluster_mask)
print(f"Found {num_clusters} clusters")

# glass brain
display = plotting.plot_glass_brain(
    data_img,
    title=f"Grade-{GRADE} (cluster-thr < {CLUSTER_SIZE}, pFDR<{P_CORRECTION})",
    threshold=min_positive,
    colorbar=True,
    display_mode='lyrz',  # show left, sagittal, coronal, axial views
    plot_abs=False,     # keep sign info if you have positive/negative values
    black_bg=False,
    vmin=0,
)
plotting.show()


# % =============================
# atlas
# aal = datasets.fetch_atlas_aal(version='SPM12')
AAL_DIR = TEMP_DIR / 'aal'

atlas_img = nib.load(f"{AAL_DIR}/ROI_MNI_V4.nii")
# atlas maps and labels
# atlas_img = nib.load(aal['maps'])
atlas_data = atlas_img.get_fdata()
# atlas_labels = aal['labels'] 
# Load the labels from the txt file
atlas_labels = pd.read_csv(
    f"{AAL_DIR}/ROI_MNI_V4.txt",
    sep="\t",
    header=None,
    names=["index", "label"]
) 
atlas_affine = atlas_img.affine

# Loop over clusters
for i in range(1, num_clusters + 1):

    cluster_data = data * (labeled_array == i)
    

    max_val = np.max(cluster_data)
    min_val = np.min(cluster_data)
    if max_val > 0:
    # Find the voxel of the maximum value in this cluster
        voxel_index = np.unravel_index(
            np.argmax(cluster_data), 
            cluster_data.shape
        )
    elif min_val < 0:
        # Find the voxel of the minimum value in this cluster
        voxel_index = np.unravel_index(
            np.argmin(cluster_data), 
            cluster_data.shape
        )

    # voxel index → pediatric world (mm)
    ped_affine = data_img.affine
    ped_coords = nib.affines.apply_affine(ped_affine, voxel_index)
    
    pts = pd.DataFrame([{
        "x": ped_coords[0],
        "y": ped_coords[1],
        "z": ped_coords[2]}])

    peak_mni_adult = ants.apply_transforms_to_points(
        dim=3,
        points=pts,
        transformlist=reg['fwdtransforms']
    )               
    adult_coords = peak_mni_adult[['x','y','z']].values[0]
    print("\n=========================")
    print(f"Cluster {i}:")
    print("Adult MNI:", np.round(adult_coords).astype(int))

    # MNI coords → voxel index in AAL
    voxel = nib.affines.apply_affine(
        np.linalg.inv(atlas_affine),
        adult_coords
    )

    # round & cast to int
    voxel = np.round(voxel).astype(int)

    if np.any(voxel < 0) or np.any(voxel >= atlas_data.shape):
        print("Outside AAL atlas")
        label_index = 0
    else:
        label_index = int(atlas_data[tuple(voxel)])
        print("AAL index:", label_index)

    if label_index == 0:
        print("Region: no region")
        print("Finding the nearest label...")

        found = False

        for r in range(1, 101):
            if found:
                break
            # iterate voxels in the shell (cube)
            i0, j0, k0 = voxel
            imin, imax = i0 - r, i0 + r
            jmin, jmax = j0 - r, j0 + r
            kmin, kmax = k0 - r, k0 + r

            # iterate through the shell
            candidates = []
            for s in range(imin, imax + 1):
                if found:
                    break
                if s < 0 or s >= atlas_data.shape[0]:
                    continue

                for j in range(jmin, jmax + 1):
                    if found:
                        break
                    if j < 0 or j >= atlas_data.shape[1]:
                        continue

                    for k in range(kmin, kmax + 1):
                        if k < 0 or k >= atlas_data.shape[2]:
                            continue

                        lbl = int(atlas_data[s, j, k])
                        if lbl != 0:
                            index = lbl
                            region_name = atlas_labels[atlas_labels['label'] == index]
                            print("Nearest Region:", region_name['index'].values[0])
                                
                            found = True
                            break
    else:
        region_name = atlas_labels[atlas_labels['label'] == label_index]  # AAL labels start at 1
        print("Region:", region_name['index'].values[0])

    # Optional: get the cluster size in voxels
    cluster_size = np.sum(labeled_array == i)
    
    print(f"Peak value: {cluster_data[voxel_index]:.2f}, "
          f"Size: {cluster_size} voxels")


# %%
