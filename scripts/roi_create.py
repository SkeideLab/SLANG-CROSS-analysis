
# -------------------------------
# %% === Libraries ===
# -------------------------------
import numpy as np
import nibabel as nib
from pathlib import Path
import templateflow.api as tflow
from nilearn.image import resample_to_img
from matplotlib import colors
from nilearn.plotting import plot_glass_brain, plot_stat_map, show
from nilearn.plotting import plot_stat_map
from nilearn import plotting, image, datasets
import ants

# -------------------------------
# %% === Paths & parameters ===
# -------------------------------
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
TMPL_DIR     = ANALY_DIR / 'templates'
MASK_DIR     = TMPL_DIR / "mask"
MASK_DIR.mkdir(exist_ok=True, parents=True)

# === VWFA parameters ===
RADIUS = 6  # mm
VWFA_CENTER = np.array([-45, -57, -12])  # MNI coordinates (left VWFA)

# === Ventral mask parameters ===
# Define your MNI coordinate for left hemisphere
x_min, x_max = -70, -20
y_min, y_max = -82, -33
z_min, z_max = -28, 4



# -------------------------------
# %% === Load MNI template in adult MNI ===
# -------------------------------
MNI_filepath = TMPL_DIR / "adult_MNI_2mm" / "MNI152_2mm.nii.gz"
mni_img = nib.load(MNI_filepath)
affine = mni_img.affine
shape = mni_img.shape

# % === Load cerebellum template in adult MNI ===
cer_path = TMPL_DIR / 'cerebellar_atlases-master' / 'tpl-MNI152NLin2009cSymC' / 'tpl-MNI152NLin2009cSymC_desc-cereb_mask.nii'
cer_mask = nib.load(cer_path)

# -----------
# --------------------
# %% === Create spherical VWFA mask ===
# -------------------------------
# Generate voxel coordinates
ijk = np.indices(shape).reshape(3, -1)
ijk_h = np.vstack([ijk, np.ones((1, ijk.shape[1]))])
xyz = (affine @ ijk_h)[:3, :].T

# Compute distances from VWFA center
dist = np.linalg.norm(xyz - VWFA_CENTER, axis=1)

# Create spherical mask
mask_data = (dist <= RADIUS).astype(np.uint8).reshape(shape)

# % === Save as NIfTI ===
# -------------------------------
vwfa_mask_img = nib.Nifti1Image(mask_data, affine)
vwfa_mask_path = MASK_DIR / "left_VWFA_adult_MNI.nii.gz"
nib.save(vwfa_mask_img, vwfa_mask_path)

print(f"VWFA mask saved to: {vwfa_mask_path}")



# -------------------------------
# %% === Create Ventral mask in adult MNI ===
# -------------------------------
mask = np.zeros(shape, dtype=np.uint8)

# Loop through all voxels and check if they fall within the boundaries
for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            # Convert voxel coordinates to MNI coordinates
            voxel_coords = np.array([i, j, k, 1])
            mni_coords = affine @ voxel_coords
            
            x, y, z = mni_coords[:3]
            
            # Check if coordinates are within boundaries
            if (x_min <= x <= x_max and 
                y_min <= y <= y_max and 
                z_min <= z <= z_max):
                mask[i, j, k] = 1

# % === Save as NIfTI ===
# -------------------------------
left_ventral_img  = nib.Nifti1Image(mask, affine)
left_ventral_path = MASK_DIR / "left_ventral_adult_MNI.nii.gz"
nib.save(left_ventral_img, left_ventral_path)

print(f"left Ventral mask saved to: {left_ventral_path}")



# -------------------------------
# %% === Conversion to pediatric space ===
# -------------------------------

# Pediatric MNI template (2mm)
TEMPLATE =  tflow.get(
    "MNIPediatricAsym",
    cohort="4",       # cohort 4 = age 7.5-13.5
    resolution=2,     # 2mm isotropic
    suffix="T1w"      # T1-weighted image
)

# === Pediatric MNI GM probablity Mask ===
PROGS = tflow.get(
    "MNIPediatricAsym",
    cohort="4",       # cohort 4 = age 7.5-13.5
    resolution=2,     # 2mm isotropic
    suffix="probseg",
    desc=None,     
)
GM_mask = PROGS[1]
GM_img  = nib.load(GM_mask)
GM_data = GM_img.get_fdata()


# === nonlinear transfomration ===
# adult MNI
fixed = ants.image_read(str(MNI_filepath))
# T1w pediatric space
moving = ants.image_read(str(TEMPLATE))
# nonlinear registration
reg = ants.registration(
    fixed  = fixed, # adult MNI
    moving = moving, # pediatric space
    type_of_transform = "SyN"   # recommended
)

# === VWFA: adult MNI space → Pediatric ===
adult_vwfa_mask = ants.image_read(str(vwfa_mask_path))
# convert adult MNI → pediatric MNI
pediatric_vwfa_mask = ants.apply_transforms(
    fixed=fixed,  # pediatric template
    moving=adult_vwfa_mask,                # adult ROI
    transformlist=reg['invtransforms'],    # adult → pediatric
    interpolator='nearestNeighbor'         # preserve binary mask
)
left_vwfa_data  = pediatric_vwfa_mask.numpy().astype(bool)
right_vwfa_data = np.flip(left_vwfa_data, axis=0)


# === Ventral: adult MNI space → Pediatric  ===
adult_ventral_mask = ants.image_read(str(left_ventral_path))
# convert adult MNI → pediatric MNI
pediatric_ventral_mask = ants.apply_transforms(
    fixed=fixed,  # pediatric template
    moving=adult_ventral_mask,                # adult ROI
    transformlist=reg['invtransforms'],    # adult → pediatric
    interpolator='nearestNeighbor'         # preserve binary mask
)
left_ventral_data  = pediatric_ventral_mask.numpy().astype(bool)
right_ventral_data = np.flip(left_ventral_data, axis=0)


# === cerebellum: adult MNI space → Pediatric  ===
cer_path = TMPL_DIR / 'cerebellar_atlases-master' / 'tpl-MNI152NLin2009cSymC' / 'tpl-MNI152NLin2009cSymC_desc-cereb_mask.nii'
adult_cerebellum_mask = ants.image_read(str(cer_path))
pediatric_cerebellum_mask = ants.apply_transforms(
    fixed=fixed,  # pediatric template
    moving=adult_cerebellum_mask,          # adult ROI
    transformlist=reg['invtransforms'],    # adult → pediatric
    interpolator='nearestNeighbor'         # preserve binary mask
)
pediatric_cerebellum_data  = pediatric_cerebellum_mask.numpy().astype(bool)


# === Clean GM probability and cerebellum  ===
GM_THRESH = 0.4

# VWFA
left_vwfa_clean  = left_vwfa_data  & (GM_data > GM_THRESH) & (~pediatric_cerebellum_data)
right_vwfa_clean = right_vwfa_data  & (GM_data > GM_THRESH) & (~pediatric_cerebellum_data)

# Cerebellum
left_ventral_clean  = left_ventral_data  & (GM_data > GM_THRESH) & (~pediatric_cerebellum_data)
right_ventral_clean = right_ventral_data  & (GM_data > GM_THRESH) & (~pediatric_cerebellum_data)


# ===  VWFA: describe the characteristic === 
voxel_sizes      = GM_img.header.get_zooms()  # should be (2.0, 2.0, 2.0)
voxel_volume     = voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]
left_volume_mm3  = voxel_volume * np.sum(left_vwfa_clean)
right_volume_mm3 = voxel_volume * np.sum(right_vwfa_clean)

# ===  ventral: describe the characteristic === 
ventral_left_volume_mm3  = voxel_volume * np.sum(left_ventral_clean)
ventral_right_volume_mm3 = voxel_volume * np.sum(right_ventral_clean)

# VWFA: print basic info
print("\nVWFA: Left hemisphere")
print(f"Mask created with shape: {left_vwfa_clean.shape}")
print(f"Number of voxels in mask: {np.sum(left_vwfa_clean)}")
print(f"Total volume: {left_volume_mm3} mm³")
print("\nVWFA: Right hemisphere")
print(f"Mask created with shape: {right_vwfa_clean.shape}")
print(f"Number of voxels in mask: {np.sum(right_vwfa_clean)}")
print(f"Total volume: {right_volume_mm3} mm³")
# ventral: print basic info
print("\nventral: Left hemisphere")
print(f"Mask created with shape: {left_ventral_clean.shape}")
print(f"Number of voxels in mask: {np.sum(left_ventral_clean)}")
print(f"Total volume: {ventral_left_volume_mm3} mm³")
print("\nventral: Right hemisphere")
print(f"Mask created with shape: {right_ventral_clean.shape}")
print(f"Number of voxels in mask: {np.sum(right_ventral_clean)}")
print(f"Total volume: {ventral_right_volume_mm3} mm³")

# -------------------------------
# %% === Save as NifTi ===
# -------------------------------
# VWFA
left_vwfa_img = nib.Nifti1Image(
    left_vwfa_clean.astype(np.uint8),
    affine=GM_img.affine
)
right_vwfa_img = nib.Nifti1Image(
    right_vwfa_clean.astype(np.uint8),
    affine=GM_img.affine
)

left_vwfa_path  = MASK_DIR / "left_VWFA_pediatric_MNI.nii.gz"
right_vwfa_path = MASK_DIR / "right_VWFA_pediatric_MNI.nii.gz"
nib.save(left_vwfa_img, left_vwfa_path)
nib.save(right_vwfa_img, right_vwfa_path)

# Ventral
left_ventral_img = nib.Nifti1Image(
    left_ventral_clean.astype(np.uint8),
    affine=GM_img.affine
)
right_ventral_img = nib.Nifti1Image(
    right_ventral_clean.astype(np.uint8),
    affine=GM_img.affine
)

left_ventral_path  = MASK_DIR / "left_ventral_pediatric_MNI.nii.gz"
right_ventral_path = MASK_DIR / "right_ventral_pediatric_MNI.nii.gz"

nib.save(left_ventral_img, left_ventral_path)
nib.save(right_ventral_img, right_ventral_path)


# -------------------------------
# %% === Visualization ===
# -------------------------------

# VWFA
left_vwfa_img  = nib.load(left_vwfa_path)
right_vwfa_img = nib.load(right_vwfa_path)

# ventral
left_ventral_img  = nib.load(left_ventral_path)
right_ventral_img = nib.load(right_ventral_path)

# Define single coordinates for each view
z_coord = -12  # One axial slice
y_coord = -57  # One coronal slice (middle of your range)
x_coord = -45  # One sagittal slice (middle of your range)


green_cmap = colors.ListedColormap(['springgreen'])
blue_cmap = colors.ListedColormap(['dodgerblue'])
# configure the figure
plotting_config = {
    "display_mode": "ortho",
    "cut_coords": (x_coord, y_coord, z_coord),
    "draw_cross": False,
    "cmap": green_cmap,
    "bg_img": TEMPLATE,
}

# VWFA: plot the mask (left) figure 
display_vwfa = plotting.plot_roi(
    left_vwfa_img,
    # title="Ventral Mask",
    colorbar=False,
    **plotting_config,
)
# VWFA: Second mask (right) added to the same figure
display_vwfa.add_overlay(
    right_vwfa_img,
    colorbar=False,
    cmap=blue_cmap,
)
plotting.show()

# VWFA: plot the mask (left) figure 
display_ventral = plotting.plot_roi(
    left_ventral_img,
    # title="Ventral Mask",
    colorbar=False,
    **plotting_config,
)
# VWFA: Second mask (right) added to the same figure
display_ventral.add_overlay(
    right_ventral_img,
    colorbar=False,
    cmap=blue_cmap,
)
plotting.show()

# -------------------------------
# % === save the figure ===
# -------------------------------
roi_path = FIG_DIR / 'roi'
roi_path.mkdir(exist_ok=True, parents=True)

# VWFA
display_vwfa.savefig(f"{roi_path}/VWFA_pediatric_mask.png", dpi=300)
print(f"\nSuccessful: Figure of the VWFA mask is saved ")
# ventral
display_ventral.savefig(f"{roi_path}/ventral_pediatric_mask.png", dpi=300)
print(f"\nSuccessful: Figure of the ventral mask is saved ")
#  -------------------------------


