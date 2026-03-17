# %%
# ===  Load modules ===
from nilearn import plotting, surface, datasets
from nilearn.plotting import plot_img_on_surf
from pathlib import Path
import numpy as np
import templateflow.api as tflow
import nibabel as nib
import ants
import matplotlib.pyplot as plt
from scipy import ndimage
import plotly.graph_objects as go
from scipy import ndimage
from scipy.ndimage import label, distance_transform_edt
from matplotlib.colors import ListedColormap, to_rgba

# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
TMPL_DIR     = ANALY_DIR / 'templates'
MASK_DIR     = TMPL_DIR / 'mask'
# === Parameters ===
MODEL          = 'glm'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
GRADES         = '4' # 1, 2, 4
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
P_CORRECTION   = 0.005 # 0.001, 0.05
CLUSTER_SIZE   = 50
MASK_TYPE      = 'STG-MTG' # ventral, VWFA, A1, IFG, MTG, STG, STG-MTG, AG-SMG
image_cont     = 'images_words'
audio_cont     = 'audios_words'
# List of the contrasts
"""             'images_words-images_pseudo' 
                'audios_words-audios_pseudo' 
                'images_words' 
                'audios_words'
                'images_pseudo'
                'audios_pseudo'
                'images_words+audios_words-images_pseudo+audios_pseudo' """




# %%
# ===  Load the thresholded z-map ===
# z-map in pediatric MNI space


f_name         =  f"GRADE-{GRADES}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
nib_path_image = OUT_DIR / MODEL / SPACE / image_cont / MASK_TYPE / f_name
z_img_image    = nib.load(nib_path_image)
z_data_image   = z_img_image.get_fdata().astype(dtype=np.float32)

f_name         =  f"GRADE-{GRADES}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
nib_path_audio = OUT_DIR / MODEL / SPACE / audio_cont / MASK_TYPE / f_name
z_img_audio    = nib.load(nib_path_audio)
z_data_audio   = z_img_audio.get_fdata().astype(dtype=np.float32)

# MNI152 adult
Adult_MNI_T1 = tflow.get(
    'MNI152NLin2009cAsym', 
    resolution=2,
    desc='brain', 
    suffix='T1w')
# MNI Pediatric
TEMPLATE     =  tflow.get(
    "MNIPediatricAsym",
    cohort="4",       # cohort 4 = age 7.5-13.5
    resolution=2,     # 2mm isotropic
    suffix="T1w"      # T1-weighted image
)
# VWFA mask
left_path = MASK_DIR / "left_VWFA_adult_MNI.nii.gz"
left_vwfa = nib.load(left_path)
right_path = MASK_DIR / "right_VWFA_adult_MNI.nii.gz"
right_vwfa = nib.load(right_path)

# MASK_TYPE mask
left_path = MASK_DIR / f"left_{MASK_TYPE}_adult_MNI.nii.gz"
left_mask = nib.load(left_path)
right_path = MASK_DIR / f"right_{MASK_TYPE}_adult_MNI.nii.gz"
right_mask = nib.load(right_path)

# %%

# === Overlap ===
overlap_data = ((z_data_image > 0) & (z_data_audio > 0)).astype(np.uint8)

# --- label connected clusters ---
labeled_array, num_features = ndimage.label(overlap_data)

# compute cluster sizes
cluster_sizes = ndimage.sum(
    overlap_data,
    labeled_array,
    index=range(1, num_features + 1)
)

# create empty mask
filtered_overlap = np.zeros_like(overlap_data, dtype=np.uint8)

# keep only clusters >= 50 voxels
for i, size in enumerate(cluster_sizes, start=1):
    if size >= 0:
        filtered_overlap[labeled_array == i] = 1

# create nifti
overlap_img = nib.Nifti1Image(
    filtered_overlap,
    affine=z_img_image.affine,
    header=z_img_image.header
)

purple_cmap = ListedColormap(["white", "purple"])
display = plotting.plot_glass_brain(
    overlap_img,
    title='Overlap',
    threshold=0,
    colorbar=False,
    display_mode='lyrz',
    plot_abs=False,
    black_bg=False,
    vmin=0,
    vmax=1,
    cmap=purple_cmap
)
display.add_contours(
    overlap_img, 
    levels=1,       # level(s) to contour
    colors='black',             # color of the outline
    linewidths=0.5
)

# save the figure
path = FIG_DIR / 'overlap' / image_cont / MASK_TYPE
path.mkdir(exist_ok=True, parents=True)
display.savefig(f"{path}/Grade_{GRADES}_z-maps_FWHM{int(FWHM_SMOOTHING)}_p-val<{P_CORRECTION}_cls>{CLUSTER_SIZE}.png", dpi=300)
plotting.show()

# save
path = OUT_DIR /'overlap' / SPACE / image_cont / MASK_TYPE
path.mkdir(exist_ok=True, parents=True)
overlap_path = path / f"GRADE-{GRADES}_overlap_images_audio.nii.gz"

nib.save(overlap_img, overlap_path)






# %%
# =========================================================
# ===  Brain regions specification with assiged cluster ===
# =========================================================

# === specify the target grade === 
tar_grade = GRADES


# Fetch Harvard-Oxford cortical + subcortical atlas
harvard_oxford_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
# Load the labels from the txt file
atlas_labels         = harvard_oxford_atlas.lut
# Path to atlas NIfTI file
atlas_img_nib        = harvard_oxford_atlas['maps']  # or .maps
# Save to temp file
atlas_path           = '/tmp/harvard_oxford_atlas.nii.gz'
atlas_img_nib.to_filename(atlas_path)


atlas_img = ants.image_read(atlas_path)
ped_img   = ants.image_read(str(TEMPLATE))
ped_nifti = nib.load(TEMPLATE)

warp_path   = str(MASK_DIR / 'MNI6_to_Pediatric_1Warp.nii.gz')
affine_path = str(MASK_DIR / 'MNI6_to_Pediatric_0GenericAffine.mat')

forward_transforms = [
    warp_path, 
    affine_path
]

# Transform
pediatric_atlas_ants = ants.apply_transforms(
    fixed=ped_img,
    moving=atlas_img,
    transformlist=forward_transforms,
    interpolator='genericLabel' 
)
pediatric_atlas_data = pediatric_atlas_ants.numpy().astype(np.int16)
# Convert pediatric atlas data back to NIfTI
pediatric_atlas      = nib.Nifti1Image(pediatric_atlas_data, ped_nifti.affine)


# % === load the z-map ===
data_img = nib.load(overlap_path)
data = data_img.get_fdata()

# Find connected clusters
cluster_mask = data != 0
labeled_array, num_clusters = label(cluster_mask)
print(f"Found {num_clusters} clusters")

# Precompute nearest labeled atlas voxel for out-of-atlas voxels
labeled_mask = pediatric_atlas_data > 0
_, nearest_indices = distance_transform_edt(
    ~labeled_mask,  # True where atlas is background
    return_indices=True
)

for i in range(1, num_clusters + 1):
    cmask = (labeled_array == i)
    cluster_voxels = np.where(cmask)  # tuple of arrays (x,y,z)
    cluster_size = cmask.sum()

    # one voxel only (first voxel in cluster)
    coords = np.argwhere(cmask)      # Nx3 voxel indices for this cluster
    voxel_index = tuple(coords[0])   # pick the first one

    # voxel -> MNI (mm)
    mni_mm = nib.affines.apply_affine(data_img.affine, voxel_index)

    # Atlas indices at cluster voxels
    atlas_vals = pediatric_atlas_data[cluster_voxels].astype(int)

    # Replace zeros (outside atlas) with nearest atlas label
    zero_mask = (atlas_vals == 0)
    if np.any(zero_mask):
        x = cluster_voxels[0][zero_mask]
        y = cluster_voxels[1][zero_mask]
        z = cluster_voxels[2][zero_mask]
        nx = nearest_indices[0, x, y, z]
        ny = nearest_indices[1, x, y, z]
        nz = nearest_indices[2, x, y, z]
        atlas_vals[zero_mask] = pediatric_atlas_data[nx, ny, nz].astype(int)

    # Count label frequencies
    atlas_vals = atlas_vals[atlas_vals > 0]  # safety
    uniq, cnt = np.unique(atlas_vals, return_counts=True)
    order = np.argsort(cnt)[::-1]  # descending by count

    print("============================================")
    print(f"One MNI coordinate: ({mni_mm[0]:.1f}, {mni_mm[1]:.1f}, {mni_mm[2]:.1f})")
    print(f"Cluster-{i} (size={cluster_size} voxels)")
    for idx, c in zip(uniq[order], cnt[order]):
        row = atlas_labels.loc[atlas_labels["index"] == idx]
        name = row.iloc[0]["name"] if len(row) else f"index={idx}"
        pct = 100 * c / cluster_size
        print(f"{name}: {c} voxels ({pct:.1f}%)")
    print("============================================")







# %%
# =============================
# ===  Plot on pial surface ===
# =============================

# overlap map
z_ants_overlap   = ants.image_read(str(overlap_path))
# visual map
z_ants_visual   = ants.image_read(str(nib_path_image))
# auditory map
z_ants_audio   = ants.image_read(str(nib_path_audio))


# MNI adults as ANTS
adult_mni       = ants.image_read(str(Adult_MNI_T1))
adult_mni_nifti = nib.load(Adult_MNI_T1)

# conversion file
warp   = str(MASK_DIR / 'MNI_to_Pediatric_1InverseWarp.nii.gz')
affine = str(MASK_DIR / 'MNI_to_Pediatric_0GenericAffine.mat')

inverse_transforms = [
    warp, 
    affine 
]

# overlap
# convert from Pediatric to Adult
z_adult = ants.apply_transforms(
    fixed=adult_mni,
    moving=z_ants_overlap,
    transformlist=inverse_transforms,
    whichtoinvert=[False, True],
    interpolator='genericLabel' 
)
# convert from ants to nifti
z_adult_data = z_adult.numpy().astype(np.float32)
z_adult = nib.Nifti1Image(z_adult_data, adult_mni_nifti.affine)


# visual
# convert from Pediatric to Adult
z_adult_visual = ants.apply_transforms(
    fixed=adult_mni,
    moving=z_ants_visual,
    transformlist=inverse_transforms,
    whichtoinvert=[False, True],
    interpolator='genericLabel' 
)
# convert from ants to nifti
z_adult_data_visual = z_adult_visual.numpy().astype(np.float32)
z_adult_visual = nib.Nifti1Image(z_adult_data_visual, adult_mni_nifti.affine)


# audio
# convert from Pediatric to Adult
z_adult_audio = ants.apply_transforms(
    fixed=adult_mni,
    moving=z_ants_audio,
    transformlist=inverse_transforms,
    whichtoinvert=[False, True],
    interpolator='genericLabel' 
)
# convert from ants to nifti
z_adult_data_audio = z_adult_audio.numpy().astype(np.float32)
z_adult_audio = nib.Nifti1Image(z_adult_data_audio, adult_mni_nifti.affine)


# fsaverage surface
fsaverage = datasets.fetch_surf_fsaverage()


hemis = ['right', 'left']
for hemi in hemis:
    if hemi == 'right':
        #  mesh and project volumetric data
        mesh  = surface.load_surf_mesh(fsaverage.pial_right)
        white_mesh = surface.load_surf_mesh(fsaverage.white_right)
        sulc  = fsaverage.sulc_right
        color = 'dodgerblue'
    else:
        mesh  = surface.load_surf_mesh(fsaverage.pial_left)
        white_mesh = surface.load_surf_mesh(fsaverage.white_left)
        sulc  = fsaverage.sulc_left
        color = 'springgreen'
    map_data = surface.vol_to_surf(
        z_adult,
        surf_mesh=mesh,
        inner_mesh=white_mesh,
        interpolation='linear',
        n_samples=20   # increase sampling density
    )
    map_data = (map_data > 0).astype(int)

    map_data_visual = surface.vol_to_surf(
        z_adult_visual,
        surf_mesh=mesh,
        inner_mesh=white_mesh,
        interpolation='linear',
        n_samples=20   # increase sampling density
    )
    map_data_visual = (map_data_visual > 0).astype(int) * 2

    map_data_audio = surface.vol_to_surf(
        z_adult_audio,
        surf_mesh=mesh,
        inner_mesh=white_mesh,
        interpolation='linear',
        n_samples=20   # increase sampling density
    )
    map_data_audio = (map_data_audio > 0).astype(int) * 3

    map_data = np.where(
        map_data != 0, map_data,
        np.where(map_data_visual != 0, map_data_visual, map_data_audio)
    )

    roi = map_data.astype(float).copy()
    # keep integer labels for contours
    roi_labels = map_data.astype(int)   # 0,1,2,3
    roi[roi == 0] = np.nan   # transparent outside mask

    roi_cmap = ListedColormap([
        (0, 0, 0, 0),            # 0: transparent
        to_rgba("darkviolet", 0.9), # 1
        to_rgba("steelblue", 0.8),   # 2
        to_rgba("indianred", 0.8)   # 3
    ])


    # VWFA map
    if hemi == 'right':
        vwfa_surf = surface.vol_to_surf(
            right_vwfa,
            surf_mesh=mesh,
            inner_mesh=white_mesh,
            interpolation='linear',
            n_samples=20   # increase sampling density
        )
        mask_surf = surface.vol_to_surf(
            right_mask,
            surf_mesh=mesh,
            inner_mesh=white_mesh,
            interpolation='linear',
            n_samples=20   # increase sampling density
        )
    else:
        vwfa_surf = surface.vol_to_surf(
            left_vwfa,
            surf_mesh=mesh,
            inner_mesh=white_mesh,
            interpolation='linear',
            n_samples=20   # increase sampling density
        )
        mask_surf = surface.vol_to_surf(
            left_mask,
            surf_mesh=mesh,
            inner_mesh=white_mesh,
            interpolation='linear',
            n_samples=20   # increase sampling density
        )
    vwfa_mask = (vwfa_surf > 0).astype(int)
    mask_mask = (mask_surf > 0).astype(int)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    if  MASK_TYPE == 'ventral':
        view="ventral"
    elif MASK_TYPE in ['STG', 'STG-MTG','AG-SMG']:
        view="lateral"

    plotting.plot_surf_roi(
        surf_mesh=mesh, roi_map=roi, hemi=hemi, view=view,
        bg_map=sulc, cmap=roi_cmap, vmin=0, vmax=3, colorbar=False,
        axes=ax, figure=fig, darkness=0.4, alpha=1
    )

    # change camera here
    if hemi == 'right':
        if MASK_TYPE == 'ventral':
            ax.view_init(elev=295, azim=0)
        elif MASK_TYPE == 'IFG':
            ax.view_init(elev=0, azim=25)
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=mask_mask,
                levels=[1],
                colors =['gold'],
                linewidths=0,
                axes=ax
            )
        elif MASK_TYPE == 'AG-SMG':
            ax.view_init(elev=20, azim=-20)
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=mask_mask,
                levels=[1],
                colors =['gold'],
                linewidths=0,
                axes=ax
            ) 
        elif MASK_TYPE in ['STG', 'STG-MTG']:
            # ax.view_init(elev=0, azim=155)
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=mask_mask,
                levels=[1],
                colors =['gold'],
                linewidths=0,
                axes=ax
            )
    else: 
        if MASK_TYPE == 'ventral':
            ax.view_init(elev=245, azim=0)
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=vwfa_mask,
                levels=[1],
                colors =['gold'],
                linewidths=0,
                axes=ax
            )
        elif MASK_TYPE == 'IFG':
            ax.view_init(elev=0, azim=155)
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=mask_mask,
                levels=[1],
                colors =['gold'],
                linewidths=0,
                axes=ax
            )
        elif MASK_TYPE == 'AG-SMG':
            ax.view_init(elev=20, azim=200)
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=mask_mask,
                levels=[1],
                colors =['gold'],
                linewidths=0,
                axes=ax
            ) 
        elif MASK_TYPE in ['STG', 'STG-MTG']: 
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=mask_mask,
                levels=[1],
                colors =['gold'],
                linewidths=0,
                axes=ax
            )
        

    """     fig = plotting.plot_surf_roi(
        surf_mesh=mesh,
        roi_map=roi,
        hemi=hemi,
        view="ventral",
        bg_map=sulc,
        cmap=roi_cmap,
        vmin=0,
        vmax=3,
        colorbar=False,
        engine="plotly",
        darkness=None
    )

    fig.show() """

    plt.tight_layout()
    # save first
    path = FIG_DIR / 'overlap' / image_cont / MASK_TYPE / f"Grade_{GRADES}_{hemi}_surface.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")

    plt.show() 



# %%
