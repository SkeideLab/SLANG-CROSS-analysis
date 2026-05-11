# %% [markdown]
# ## Generate ROI masks in the pediatric MNI space
#
# **Pipeline Overview**
# 1. === STEP 1 ===: Fetch templates and atlases
# 2. === STEP 2 ===: ROIs creation in adult MNI space
# 3. === STEP 3 ===: Nonlinear transformation
# 4. === STEP 4 ===: ROIs transformed in Pediatric MNI space 
# 5. === STEP 5 ===: Supplemnetary Figure creation


# %%
# === Packages ===
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import templateflow.api as tflow
from nilearn.image import resample_to_img
from matplotlib import colors
from nilearn.plotting import plot_glass_brain, plot_stat_map, show
from nilearn.plotting import plot_stat_map
from nilearn import plotting, image, datasets
import ants
import string
print(ants.__version__)


# %%
# ===  Parameters ===
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
TMPL_DIR     = ANALY_DIR / 'templates'
MASK_DIR     = TMPL_DIR / 'mask'
MASK_DIR.mkdir(exist_ok=True, parents=True)




# %%
# ===========================================
# === STEP 1 ===: Fetch templates and atlases
# ===========================================

# pediatric MNI T1w
Pediatric_MNI_T1 =  tflow.get(
    "MNIPediatricAsym",
    cohort="4",       # cohort 4 = age 7.5-13.5
    resolution=2,     # 2mm isotropic
    suffix="T1w")      # T1-weighted image

# adult MNI T1w
Adult_MNI_T1 = tflow.get(
    'MNI152NLin2009cAsym', 
    resolution=2,
    desc='brain', 
    suffix='T1w')

# Harvard-Oxford atlas
ATLAS = tflow.get(
    'MNI152NLin2009cAsym', 
    atlas='HOCPAL', 
    desc='th25', 
    resolution=2)

# Pediatric MNI GM probablity Mask
Pediatric_GM = tflow.get(
    "MNIPediatricAsym",
    cohort="4",       # cohort 4 = age 7.5-13.5
    resolution=2,     # 2mm isotropic
    suffix="probseg",
    desc=None,
    label="GM")

# adult MNI6Asym T1w
Adult_MNI6_T1 = tflow.get(
    'MNI152NLin6Asym', 
    resolution=2,
    desc='brain', 
    suffix='T1w')

# Harvard-Oxford MNI6Asym
HO_ATLAS_MNI6 = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

# VWFA parameters
RADIUS      = 6  # mm
VWFA_CENTER = np.array([-45, -57, -12])  # MNI coordinates (left VWFA)

# Ventral mask parameters
# Define your MNI coordinate for left hemisphere
x_min, x_max = -70, -20
y_min, y_max = -82, -33
z_min, z_max = -28, 4

# Load adult MNI template
mni_img = nib.load(Adult_MNI_T1)
affine  = mni_img.affine
shape   = mni_img.shape

# Load Harvard-Oxford atlas
HO_img  = nib.load(ATLAS)
HO_data = HO_img.get_fdata()
HO_mask = (HO_data > 0)



# %%
# ================================================
# === STEP 2 ===: ROIs creation in adult MNI space
# ================================================

# === VWFA mask ===
# Generate voxel coordinates
ijk            = np.indices(shape).reshape(3, -1)
ijk_h          = np.vstack([ijk, np.ones((1, ijk.shape[1]))])
xyz            = (affine @ ijk_h)[:3, :].T
# Compute distances from VWFA center
dist           = np.linalg.norm(xyz - VWFA_CENTER, axis=1)
# Create spherical mask
mask_left      = (dist <= RADIUS).astype(np.uint8).reshape(shape)
# right spherical mask
mask_right     = np.flip(mask_left, axis=0)
# Save as NIfTI
# left VWFA
vwfa_mask_img  = nib.Nifti1Image(mask_left, affine)
vwfa_mask_path = MASK_DIR / "left_VWFA_adult_MNI.nii.gz"
nib.save(vwfa_mask_img, vwfa_mask_path)
print(f"VWFA mask saved to: {vwfa_mask_path}")
# right VWFA
vwfa_mask_img  = nib.Nifti1Image(mask_right, affine)
vwfa_mask_path = MASK_DIR / "right_VWFA_adult_MNI.nii.gz"
nib.save(vwfa_mask_img, vwfa_mask_path)
print(f"VWFA mask saved to: {vwfa_mask_path}")


# === ATLAS mask in MNI6Asym 2mm ===
atlas_mni6   = HO_ATLAS_MNI6.maps
atlas_data   = atlas_mni6.get_fdata()
# ATLAS Labels
atlas_labels = HO_ATLAS_MNI6.lut
roi_dict     = {}
for name in atlas_labels['name']:
    if name == 'Background':
        continue
    roi_dict.setdefault(name, []).append(name)

for roi_name, names in roi_dict.items():
    # Get the index and Mask the regions out
    audiotry_ind = atlas_labels.loc[
        atlas_labels['name'].isin(names),
        'index'
        ].values
    audiotry_mask_data     = np.where(np.isin(atlas_data, audiotry_ind), atlas_data, 0)
    # Split the hemisphere
    left_mask              = audiotry_mask_data.copy()
    right_mask             = audiotry_mask_data.copy()
    # get the middle x-coordinate
    mid_x                  = atlas_data.shape[0] // 2
    # assign zero 
    left_mask[mid_x:,:,:]  = 0
    right_mask[:mid_x,:,:] = 0
    # Save as NIfTI
    stg_left_mask_nifti    = nib.Nifti1Image(left_mask, atlas_mni6.affine)
    stg_right_mask_nifti   = nib.Nifti1Image(right_mask, atlas_mni6.affine)
    # path
    left_auditory_path     = MASK_DIR / f"left_{roi_name}_adult_MNI.nii.gz"
    right_auditory_path    = MASK_DIR / f"right_{roi_name}_adult_MNI.nii.gz"
    # save
    nib.save(stg_left_mask_nifti, left_auditory_path)
    nib.save(stg_right_mask_nifti, right_auditory_path)
    print(f"left {roi_name} mask saved to: {left_auditory_path}")
    print(f"right {roi_name} mask saved to: {right_auditory_path}")


# === Ventral mask in adult MNI ===
mask = np.zeros(shape, dtype=np.uint8)
# Loop through all voxels and check if they fall within the boundaries
for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            # Convert voxel coordinates to MNI coordinates
            voxel_coords = np.array([i, j, k, 1])
            mni_coords   = affine @ voxel_coords
            x, y, z      = mni_coords[:3]
            # Check if coordinates are within boundaries
            if (x_min <= x <= x_max and 
                y_min <= y <= y_max and 
                z_min <= z <= z_max):
                mask[i, j, k] = 1
# right ventral mask
mask_right = np.flip(mask, axis=0)
# subtract background
mask       = mask * HO_mask.astype(np.uint8)
mask_right = mask_right * HO_mask.astype(np.uint8)
# Save as NIfTI 
# left ventral mask
left_ventral_img   = nib.Nifti1Image(mask, affine)
left_ventral_path  = MASK_DIR / "left_ventral_adult_MNI.nii.gz"
nib.save(left_ventral_img, left_ventral_path)
print(f"left Ventral mask saved to: {left_ventral_path}")
# right ventral mask
right_ventral_img  = nib.Nifti1Image(mask_right, affine)
right_ventral_path = MASK_DIR / "right_ventral_adult_MNI.nii.gz"
nib.save(right_ventral_img, right_ventral_path)
print(f"right Ventral mask saved to: {right_ventral_path}")




# %%
# =========================================
## === STEP 3 ===: Nonlinear transformation 
# =========================================

# nonlinear transfomration files 
# adult MNI152 T1w
moving = ants.image_read(str(Adult_MNI_T1))
# T1w pediatric space
fixed  = ants.image_read(str(Pediatric_MNI_T1))
# nonlinear registration
out    = str(MASK_DIR / 'MNI_to_Pediatric_')
reg    = ants.registration(
    fixed             = fixed, # pediatric space
    moving            = moving, # adult space
    type_of_transform = "SyN",  
    outprefix         = out
)
print(f"Transforms saved to {MASK_DIR}")

# nonlinear transfomration files
# adult MNI6 T1w
moving = ants.image_read(str(Adult_MNI6_T1))
# T1w pediatric space
fixed  = ants.image_read(str(Pediatric_MNI_T1))
# nonlinear registration
out    = str(MASK_DIR / 'MNI6_to_Pediatric_')
reg    = ants.registration(
    fixed             = fixed, # pediatric space
    moving            = moving, # adult space
    type_of_transform = "SyN", 
    outprefix         = out
)
print(f"Transforms saved to {MASK_DIR}")




# %%
# ========================================================
## === STEP 4 ===: ROIs transformed in Pediatric MNI space 
# ========================================================
# target masks
rois = ['ventral', 'VWFA'] + list(roi_dict.keys())

for roi in rois:
    # path
    left_path               = MASK_DIR / f"left_{roi}_adult_MNI.nii.gz"
    right_path              = MASK_DIR / f"right_{roi}_adult_MNI.nii.gz"
    # Apply to your mask
    pediatric_img           = ants.image_read(str(Pediatric_MNI_T1))
    left_ventral_mask       = ants.image_read(str(left_path))
    right_ventral_mask      = ants.image_read(str(right_path))
    # Get transform file
    if roi in roi_dict.keys():
        warp_path           = str(MASK_DIR / 'MNI6_to_Pediatric_1Warp.nii.gz')
        affine_path         = str(MASK_DIR / 'MNI6_to_Pediatric_0GenericAffine.mat')
    else:
        warp_path           = str(MASK_DIR / 'MNI_to_Pediatric_1Warp.nii.gz')
        affine_path         = str(MASK_DIR / 'MNI_to_Pediatric_0GenericAffine.mat')
    forward_transforms      = [warp_path, affine_path]
    # Transform
    pediatric_ventral_left  = ants.apply_transforms(
        fixed               = pediatric_img,
        moving              = left_ventral_mask,
        transformlist       = forward_transforms,
        interpolator        = 'genericLabel' 
    )
    pediatric_ventral_right = ants.apply_transforms(
        fixed               = pediatric_img,
        moving              = right_ventral_mask,
        transformlist       = forward_transforms,
        interpolator        = 'genericLabel' 
    )
    if roi in roi_dict.keys():
        left_ventral_data   = pediatric_ventral_left.numpy().astype(np.int16)
        right_ventral_data  = pediatric_ventral_right.numpy().astype(np.int16)
    else:
        left_ventral_data   = pediatric_ventral_left.numpy().astype(bool)
        right_ventral_data  = pediatric_ventral_right.numpy().astype(bool)
    # GM probability mask 
    GM_img                  = nib.load(Pediatric_GM)
    GM_data                 = GM_img.get_fdata()
    # GM probability threshold
    GM_THRESH               = 0.4
    # remove not GM regions
    if roi in roi_dict.keys():
        # keep labels where GM > threshold, else 0
        left_ventral_clean                        = left_ventral_data.copy()
        left_ventral_clean[GM_data <= GM_THRESH]  = 0

        right_ventral_clean                       = right_ventral_data.copy()
        right_ventral_clean[GM_data <= GM_THRESH] = 0
    else:
        left_ventral_clean                        = left_ventral_data  & (GM_data > GM_THRESH) 
        right_ventral_clean                       = right_ventral_data  & (GM_data > GM_THRESH) 
    # Describe the characteristic of mask
    voxel_sizes      = GM_img.header.get_zooms()  # should be (2.0, 2.0, 2.0)
    voxel_volume     = voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]
    if roi in roi_dict.keys():
        n_vox_left   = np.sum(left_ventral_clean > 0)
        n_vox_right  = np.sum(right_ventral_clean > 0)
    else:
        n_vox_left   = np.sum(left_ventral_clean)
        n_vox_right  = np.sum(right_ventral_clean)
    left_volume_mm3  = voxel_volume * n_vox_left
    right_volume_mm3 = voxel_volume * n_vox_right

    # print basic info
    print(f"\n{roi}: Left hemisphere")
    print(f"Mask created with shape: {left_ventral_clean.shape}")
    print(f"Number of voxels in mask: {n_vox_left}")
    print(f"Total volume: {left_volume_mm3} mm³")
    print(f"\n{roi}: Right hemisphere")
    print(f"Mask created with shape: {right_ventral_clean.shape}")
    print(f"Number of voxels in mask: {n_vox_right}")
    print(f"Total volume: {right_volume_mm3} mm³")

    # Save as NifTi
    # Pediatric T1w
    ped_img            = nib.load(Pediatric_MNI_T1)
    left_ventral_img   = nib.Nifti1Image(
        left_ventral_clean.astype(np.int16),
        affine         = ped_img.affine
    )
    right_ventral_img  = nib.Nifti1Image(
        right_ventral_clean.astype(np.int16),
        affine         = ped_img.affine
    )
    # path
    left_ventral_path  = MASK_DIR / f"left_{roi}_pediatric_MNI.nii.gz"
    right_ventral_path = MASK_DIR / f"right_{roi}_pediatric_MNI.nii.gz"
    # save
    nib.save(left_ventral_img, left_ventral_path)
    nib.save(right_ventral_img, right_ventral_path)




# %%
# ==============================================
## === STEP 5 ===: Supplementary Figure creation 
# ==============================================

# Figure S1. Ventral and VWFA masks in the pediatric MNI space.pdf
rois       = ['ventral', 'VWFA']
letters    = ['A', 'B']
fig, axes  = plt.subplots(len(rois), 1, figsize=(5, 5))

# coordinates
z_coord    = -12 
y_coord    = -57 
x_coord    = -45 
green_cmap = colors.ListedColormap(['springgreen'])
blue_cmap  = colors.ListedColormap(['dodgerblue'])

# configure the figure
plotting_config = {
    "display_mode": "ortho",
    "cut_coords": (x_coord, y_coord, z_coord),
    "draw_cross": False,
    "cmap": green_cmap,
    "bg_img": Pediatric_MNI_T1,
}

for i, roi in enumerate(rois):
    # Load images
    left_path    = MASK_DIR / f"left_{roi}_pediatric_MNI.nii.gz"
    right_path   = MASK_DIR / f"right_{roi}_pediatric_MNI.nii.gz"
    left_img     = nib.load(left_path)
    right_img    = nib.load(right_path)
    # Plot on the specific axis for this ROI
    display      = plotting.plot_roi(
        left_img,
        axes     = axes[i],  
        colorbar = False,
        **plotting_config
    )
    # Overlay the right side
    display.add_overlay(
        right_img,
        colorbar = False,
        cmap     = blue_cmap
    )
    # Add a title 
    axes[i].set_title(f"{letters[i]}", loc='left', fontsize=12, fontweight='bold')

# save the figure
roi_path = FIG_DIR / 'roi'
roi_path.mkdir(exist_ok=True, parents=True)
path     = roi_path / "Figure_S1.pdf"
plt.savefig(
    path,
    format      = 'pdf',
    dpi         = 300,
    transparent = True,
    bbox_inches = 'tight',
    pad_inches  = 0.3
)
print(f"\nSuccessful: Figure S1 is saved ")

# Figure S2. Harvard-Oxford cortical atlas in the pediatric MNI space.pdf
# Get all 48 left ROI files
left_rois_all   = sorted(list(MASK_DIR.glob("left*pediatric*")))
# filter out 'ventral' and 'VWFA'
left_rois       = sorted([
    f for f in left_rois_all 
    if 'ventral' not in f.name.lower() and 'vwfa' not in f.name.lower()
])
n_rois          = len(left_rois)
# Generate a large set of distinct colors using a colormap
colormap_base   = plt.cm.get_cmap('nipy_spectral', n_rois)
roi_colors      = [colormap_base(i) for i in range(n_rois)]
# Initialize the figure
fig, ax         = plt.subplots(figsize=(4, 2))
plotting_config = {
    "display_mode": "ortho",
    "cut_coords": (x_coord, y_coord, z_coord),
    "draw_cross": False,
    "bg_img": Pediatric_MNI_T1,
}
display = None
for i, left_path in enumerate(left_rois):
    roi_name     = left_path.name.replace('left_', '')
    right_path   = MASK_DIR / f"right_{roi_name}"
    # Create a solid colormap for this specific ROI pair
    current_cmap = colors.ListedColormap([roi_colors[i]])
    # Load images
    l_img        = nib.load(left_path)
    r_img        = nib.load(right_path)
    if display is None:
        # Initialize display
        display      = plotting.plot_roi(
            l_img,
            axes     = ax,
            alpha    = 0.8,
            cmap     = current_cmap,
            colorbar = False,
            **plotting_config
        )
        display.add_overlay(r_img, cmap=current_cmap, alpha=0.8)
    else:
        display.add_overlay(l_img, cmap=current_cmap, alpha=0.8)
        display.add_overlay(r_img, cmap=current_cmap, alpha=0.8)

# save the figure
roi_path = FIG_DIR / 'roi'
roi_path.mkdir(exist_ok=True, parents=True)
path     = roi_path / "Figure_S2.pdf"
display.savefig(
    path,
    dpi         = 300,
    format      = 'pdf',
    transparent = True,
    bbox_inches = 'tight'
)
print(f"\nSuccessful: Figure S2 is saved ")


# %%
