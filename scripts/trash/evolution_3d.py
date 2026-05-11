# %%
# ===  Load modules ===
from pathlib import Path
import nibabel as nib
from nilearn.plotting import plot_glass_brain
from nilearn import plotting, image, datasets
import templateflow.api as tflow
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours
from skimage import measure  # to find contours
# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
TEMP_DIR     = ANALY_DIR / 'templates'
MASK_DIR     = TEMP_DIR / 'mask'

# === Parameters ===
MODEL          = 'glm'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_pseudo'
MASK           = 'ventral' # none, ventral, WVFA, 'auditory'
GRADE          = '1' # 1, 2, 4 or all
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
TEMPLATE       =  tflow.get(
    "MNIPediatricAsym",
    cohort="4",       # cohort 4 = age 7.5-13.5
    resolution=2,     # 2mm isotropic
    suffix="T1w"      # T1-weighted image
)


# %%
# =========================
# ===  Load the dataset ===
# =========================

# ===  Load the z-map ===
fn   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_{MASK}_z-map.nii.gz"
path = OUT_DIR / MODEL / SPACE / CONTRASTS / fn

z_map  = nib.load(path)
z_data = z_map.get_fdata()
affine = z_map.affine

# ===  load the left VWFA mask ===
fn   = f"left_VWFA_pediatric_MNI.nii.gz"
path = MASK_DIR / fn
left_vwfa = nib.load(path)
left_vwfa = left_vwfa.get_fdata().astype(np.int16)



# ==========================
# ===  Clean the dataset ===
# ==========================

# assign zero to negative z-score
copy_zmap  = z_data.copy()
copy_zmap[copy_zmap < 0] = 0

# create new NIfTI
masked_img = nib.Nifti1Image(copy_zmap, affine)

# load the data
data       = masked_img.get_fdata()
affine     = masked_img.affine

# Target z-coordinate pediatric MNI
z_mni      = -12 # mm coordinate

# Convert MNI coordinate (0,0,z_mni) to voxel space
voxel_coo  = nib.affines.apply_affine(
    np.linalg.inv(affine),
    [0, 0, z_mni]
)
z_idx      = int(round(voxel_coo[2]))
print("voxel slice index:", z_idx)

# sliced data with based on the z-coordinate
slice_data = data[:, :, z_idx].copy()  # shape (99, 117)

# mask of valid voxels (adjust condition if needed)
valid      = slice_data != 0

rows       = np.any(valid, axis=1)
cols       = np.any(valid, axis=0)

# obtain the range of x and y-axis
if GRADE == '4' and CONTRASTS == 'images_words':
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
else:
    rmin, rmax = 15, 84
    cmin, cmax = 27, 51

# remove unncessary brain regions
slice_crop = slice_data[1+rmin:rmax+1, 1+cmin:cmax+1]

# left VWFA contour
vwfa_data  = left_vwfa[:, :, z_idx].copy()  # shape (99, 117)
slice_vwfa = vwfa_data[1+rmin:rmax+1, 1+cmin:cmax+1]
ZI_vwfa    = slice_vwfa.T




# ====================
# === 3D plot prep ===
# ====================

# range of the each axis
xi        = np.arange(slice_crop.shape[0]) + rmin 
yi        = np.arange(slice_crop.shape[1]) + cmin 

# create the mesh grid
# Z score
ZI        = slice_crop.T  # transpose to match X=rows, Y=columns

# Smooth the Z grid
ZI_smooth = gaussian_filter(ZI, sigma=1)  # adjust sigma for more/less smoothing

# X and Y axis
XI, YI    = np.meshgrid(xi, yi)

# extract the MNI coordinates
ZI_vox    = np.full_like(XI, z_idx)
coords    = np.stack([XI, YI, ZI_vox], axis=-1)   # (ny, nx, 3)
coords_mm = nib.affines.apply_affine(affine, coords)
Xmm       = coords_mm[..., 0]
Ymm       = coords_mm[..., 1]
Zmm       = coords_mm[..., 2]  




# ===============
# === 3D plot ===
# ===============

fig   = plt.figure(figsize=(10,8))
ax    = fig.add_subplot(111, projection='3d')

# surface plot
color = 'viridis'
surf  = ax.plot_surface(
    Xmm, Ymm, ZI_smooth,
    edgecolor='white', 
    lw=0.2, rstride=1, cstride=1,
    alpha=0.9, antialiased=True, 
    cmap=color
)


# contour projections
# desired alpha
alpha_val = 0.6

# create RGBA array: only gold where ROI > 0
facecolors = np.zeros(ZI_vwfa.shape + (4,))        # fully transparent
facecolors[..., 0] = 1.0                           # red channel
facecolors[..., 1] = 0.65                        # green channel
facecolors[..., 2] = 0.0                           # blue channel
facecolors[..., 3] = (ZI_vwfa > 0) * alpha_val    # alpha channel


ax.contourf(Xmm, Ymm, ZI_smooth, zdir='z', offset=-3, cmap=color)
# plot thin red ROI surface at z=-2.5
ax.plot_surface(
    Xmm, Ymm, np.full_like(ZI_vwfa, -2.9),
    facecolors=facecolors,
    linewidth=0,
    lw=0.1, rstride=1, cstride=1,
    antialiased=True
)
ax.contourf(Xmm, Ymm, ZI_smooth, zdir='y', offset=-35, cmap=color)


# labels and limits
ax.set(xlim=(np.min(Xmm)-2, np.max(Xmm)+2), ylim=(np.min(Ymm)-2, np.max(Ymm)+2), zlim=(-3, 3.5))
ax.set_xlabel('Pediatric MNI X (mm)', fontsize=12)
ax.set_ylabel('Pediatric MNI Y (mm)', fontsize=12)
ax.set_zlabel('Z-score', fontsize=12)

# change the view point
ax.view_init(elev=30, azim=-90)

# save figure
path  = FIG_DIR / 'evol_3d'
path.mkdir(exist_ok=True, parents=True)
plt.tight_layout() # adjusts spacing
plt.savefig(f'{path}/GRADE-{GRADE}_{CONTRASTS}_MNI_z={z_mni}mm.png', dpi=600, bbox_inches='tight')  # high-res PNG

# show figure
plt.show()

# %%
# visualize anterior shift
# ===============
# === 3D plot ===
# ===============
ZI_new = ZI_smooth.copy()
ZI_new[Xmm > 0] = 0

fig   = plt.figure(figsize=(10,8))
ax    = fig.add_subplot(111, projection='3d')

# surface plot
color = 'viridis'
surf  = ax.plot_surface(
    Xmm, Ymm, ZI_new,
    edgecolor='white', 
    lw=0.2, rstride=1, cstride=1,
    alpha=0.9, antialiased=True, 
    cmap=color
)


# contour projections
# desired alpha
alpha_val = 0.6

# create RGBA array: only gold where ROI > 0
facecolors = np.zeros(ZI_vwfa.shape + (4,))        # fully transparent
facecolors[..., 0] = 1.0                           # red channel
facecolors[..., 1] = 0.65                          # green channel
facecolors[..., 2] = 0.0                           # blue channel
facecolors[..., 3] = (ZI_vwfa > 0) * alpha_val     # alpha channel


ax.contourf(Xmm, Ymm, ZI_smooth, zdir='z', offset=-3, cmap=color)
# plot thin red ROI surface at z=-2.5
ax.plot_surface(
    Xmm, Ymm, np.full_like(ZI_vwfa, -2.9),
    facecolors=facecolors,
    linewidth=0,
    lw=0.1, rstride=1, cstride=1,
    antialiased=True
)
# ax.contourf(Xmm, Ymm, ZI_smooth, zdir='y', offset=-35, cmap=color)


# labels and limits
ax.set(xlim=(np.min(Xmm)-2, np.max(Xmm)+2), ylim=(np.min(Ymm)-2, np.max(Ymm)+2), zlim=(-3, 3.5))
ax.set_xlabel('Pediatric MNI X (mm)', fontsize=12)
ax.set_ylabel('Pediatric MNI Y (mm)', fontsize=12)
ax.set_zlabel('Z-score', fontsize=12)

# change the view point
ax.view_init(elev=20, azim=5)

# save figure
path  = FIG_DIR / 'evol_3d'
path.mkdir(exist_ok=True, parents=True)
plt.tight_layout() # adjusts spacing
plt.savefig(f'{path}/GRADE-{GRADE}_{CONTRASTS}_MNI_z={z_mni}mm_anterior.png', dpi=600, bbox_inches='tight')  # high-res PNG

# show figure
plt.show()




# %%
# ===  Visualization ===
display = plotting.plot_glass_brain(
    masked_img,
    title=f"z-map",
    colorbar=True,
    display_mode='lyrz',  # show left, sagittal, coronal, axial views
    plot_abs=False,     # keep sign info if you have positive/negative values
    black_bg=False,
)
plotting.show()
z_coords = np.arange(-16, -7, 1) # from -5 to 5
# configure the figure
plotting_config = {
    "display_mode": "z",
    "cut_coords": z_coords,
    "draw_cross": False,
    "vmax": 5,
    "vmin": -5,
    "cmap": "coolwarm",
}
display_FPR = plotting.plot_stat_map(
    masked_img,
    title=f"z-map",
    bg_img=TEMPLATE,
    **plotting_config,
)