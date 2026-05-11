# %% [markdown]
# ## fMRI Group-level: Orthographic processing in the ventral viusal pathway

# **Pipeline Overview**
# 1. === STEP 1 ===: Voxel-wise one-sample t-tests (pFDR < 0.05, cluster size > 50)
# 2. === STEP 2 ===: Anatomical label on peak z-value within clusters
# 3. === STEP 3 ===: Count the significant voxels within VWFA mask
# 4. === STEP 4 ===: 3D topographic surface plots of z-maps (z = -12 mm)
# 5. === STEP 5 ===: Figure 1. Recognition of written words in the ventral visual pathway
# 6. === STEP 6 ===: Figure 2. Emergence of the VWFA


# %%
# %% 
# === Packages ===

# --- Standard library ---
import os
from pathlib import Path
from collections import Counter

# --- Scientific computing ---
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.ndimage import label, find_objects, distance_transform_edt, gaussian_filter

# --- Neuroimaging I/O ---
import nibabel as nib
import ants
import templateflow.api as tflow

# --- Nilearn (fMRI analysis) ---
from nilearn import plotting, image, datasets, surface
from nilearn.plotting import plot_glass_brain, plot_stat_map, show

from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import cluster_level_inference, threshold_stats_img

from nilearn.mass_univariate import permuted_ols

from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_multi_epi_mask

# --- Atlases ---
from nilearn.datasets import (
    fetch_atlas_harvard_oxford,
    fetch_atlas_aal,
)

# --- FIgure ---
from matplotlib.colors import ListedColormap, to_rgba
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import matplotlib.font_manager as fm
from matplotlib import font_manager

# %%
# ===  FIXED: Parameters ===
ANALY_DIR       = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
DERIV_DIR       = ANALY_DIR / 'derivatives'
FIG_DIR         = ANALY_DIR / 'figures'
OUT_DIR         = ANALY_DIR / 'outputs'
DEMO_DIR        = ANALY_DIR / 'demographics'
TEMP_DIR        = ANALY_DIR / 'templates'
MASK_DIR        = TEMP_DIR / 'mask'

# === Parameters ===
MODEL          = 'glm'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS       = ['images_words', 'images_pseudo', 'audios_words', 'audios_pseudo']
MASK           = 'ventral' # 
GRADE          = '1' # 1, 2, 4
FWHM_SMOOTHING = 9.0 
CORRECTION     = 'fdr' # fdr, fpr, bonferoni 
P_CORRECTION   = 0.05 
CLUSTER_SIZE   = 50
EXC_SUBJECTS   = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434']
TEMPLATE       = tflow.get("MNIPediatricAsym", cohort="4",resolution=2,suffix="T1w") # T1w template for cohort 4 (7.5-13.5yrs) at 2mm resolution
Adult_MNI_T1   = tflow.get('MNI152NLin2009cAsym', resolution=2,desc='brain', suffix='T1w') # MNI152 adult
# adult VWFA mask
left_path      = MASK_DIR / "left_VWFA_adult_MNI.nii.gz"
left_vwfa_adu  = nib.load(left_path)
right_path     = MASK_DIR / "right_VWFA_adult_MNI.nii.gz"
right_vwfa_adu = nib.load(right_path)
# adult ventral mask
left_path = MASK_DIR / f"left_{MASK}_adult_MNI.nii.gz"
left_mask = nib.load(left_path)
right_path = MASK_DIR / f"right_{MASK}_adult_MNI.nii.gz"
right_mask = nib.load(right_path)




# %%
for CONTRAST in CONTRASTS:
    print(f"Analyses on {CONTRAST}")
    # %
    # =================================================================================
    # 1. === STEP 1 ===: Voxel-wise one-sample t-tests (pFDR < 0.05, cluster size > 50)
    # =================================================================================

    # list the beta nii.files
    subjects      = sorted(DERIV_DIR.glob(f"sub-{GRADE}*"))
    exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
    subjects      = [s for s in subjects if s.name not in exclude]
    subject_names = [p.name.replace('sub-', '') for p in subjects]

    # get the beta paths
    beta_lists = []
    for sub in subjects:
        path = sub / 'glm' / SPACE / f"FWHM_{int(FWHM_SMOOTHING)}" 
        beta_path = next(path.glob(f"{CONTRAST}_beta.nii.gz"), None)
        beta_lists.append(beta_path)
    beta_paths = [str(p) for p in beta_lists]

    # get the behavioral results
    acc_lists = []
    rt_lists  = []
    for sub in subjects:
        path     = sub / 'behavior' / 'excluded' / 'accuracy_summary.csv'
        df       = pd.read_csv(path)
        acc_mean = float(round(df['accuracy_visual'].mean(), 1))
        rt_mean  = float(round(df['RT_visual'].mean(), 1))
        acc_lists.append(acc_mean)
        rt_lists.append(rt_mean)


    # GLM design
    design_matrix = pd.DataFrame({
        "intercept": [1] * len(beta_paths),
    })

    # specify the model
    second_level_model = SecondLevelModel(
        smoothing_fwhm=None, 
        n_jobs=2
        )

    # Parametric: One-sample t-test
    second_level_model = second_level_model.fit(
        beta_paths,
        design_matrix=design_matrix,
    )

    # z-map 
    z_map = second_level_model.compute_contrast(
        second_level_contrast="intercept",
        output_type="z_score",
    )

    # ventral mask
    left_mask_fn  = MASK_DIR / f'left_{MASK}_pediatric_MNI.nii.gz'
    right_mask_fn = MASK_DIR / f'right_{MASK}_pediatric_MNI.nii.gz'

    left_mask_img = nib.load(left_mask_fn)
    right_mask_img = nib.load(right_mask_fn)

    # Combine masks (union)
    ventral_mask_data = (
            (left_mask_img.get_fdata() > 0) |
            (right_mask_img.get_fdata() > 0)
        ).astype(np.uint8)
        
    ventral_mask_img = nib.Nifti1Image(
            ventral_mask_data,
            left_mask_img.affine
        )

    # z-map in the ventral  
    z_map_ventral = image.math_img(
            "stat * mask",
            stat=z_map,
            mask=ventral_mask_img
        )

    # save the z-map before applying threshold
    path        = OUT_DIR / 'orthography' / SPACE / CONTRAST / MASK
    path.mkdir(parents=True, exist_ok=True)
    file_name   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_{MASK}_z-map.nii.gz"
    output_path_unth =  path / file_name
    z_map_ventral.to_filename(output_path_unth)

    # apply statistical threshold   
    thresholded_map, threshold = threshold_stats_img(
        z_map_ventral,
        two_sided=True,
        alpha=P_CORRECTION,
        cluster_threshold=CLUSTER_SIZE,
        height_control=CORRECTION,
    )

    # save the thresholded z-map
    path = OUT_DIR / 'orthography' / SPACE / CONTRAST / MASK
    path.mkdir(parents=True, exist_ok=True)

    file_name   = f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
    output_path = path / file_name
    thresholded_map.to_filename(output_path)

    # print the job
    print("# 1. Voxel-wise one-sample t-tests (pFDR < 0.05, cluster size > 50)")
    print(f"thresholded z-map is saved as {output_path}")





    # %
    # ===================================================================
    # 2. === STEP 2 ===: Anatomical label on peak z-value within clusters
    # ===================================================================

    # Harvard-Oxford cortical atlas
    harvard_oxford_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_labels         = harvard_oxford_atlas.lut # Load the labels from the txt file
    atlas_img_nib        = harvard_oxford_atlas['maps'] # Path to atlas NIfTI file
    atlas_path           = '/tmp/harvard_oxford_atlas.nii.gz'
    atlas_img_nib.to_filename(atlas_path) # Save to temp file

    # read it as ants
    atlas_img = ants.image_read(atlas_path)
    ped_img   = ants.image_read(str(TEMPLATE))
    ped_nifti = nib.load(TEMPLATE)

    # get transformation file
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
    pediatric_atlas      = nib.Nifti1Image(pediatric_atlas_data, ped_nifti.affine)

    # load the thresholded z-map === 
    data_img  = nib.load(output_path)
    data      = data_img.get_fdata()

    # Find the minimum positive value
    #  min_positive = data[data > 0].min()

    # find the MNI coordinates of a cluster
    cluster_mask = data != 0

    # Label connected clusters
    labeled_array, num_clusters = label(cluster_mask)

    # for every voxels, givs indices of nearest labeled voxel
    labeled_mask = pediatric_atlas_data > 0
    distances, nearest_indices = distance_transform_edt(
        ~labeled_mask,            # background voxels
        return_indices=True
    )

    # Loop over clusters
    for i in range(1, num_clusters + 1):

        cluster_data = data * (labeled_array == i)
        max_val = np.max(cluster_data)

        # Only proceed if peak z is positive
        if max_val <= 0:
            continue

        # Find voxel of maximum value
        voxel_index = np.unravel_index(
            np.argmax(cluster_data),
            cluster_data.shape
        )

        # get the label value
        atlas_index = pediatric_atlas_data[voxel_index]

        if atlas_index != 0:
            outside = False
            atlas_label_row = atlas_labels.loc[atlas_labels['index'] == atlas_index]
        else:
            outside = True
            nearest_voxel = tuple(nearest_indices[:, voxel_index[0], voxel_index[1], voxel_index[2]])
            nearest_label_value = pediatric_atlas_data[nearest_voxel]
            atlas_label_row = atlas_labels.loc[atlas_labels['index'] == nearest_label_value]

        # Voxel -> pediatric MNI world coordinates
        voxel_homogeneous = np.array(voxel_index + (1,))
        mni_mm = ped_nifti.affine.dot(voxel_homogeneous)[:3]

        # cluster size
        cluster_size = np.sum(labeled_array == i)

        print("============================================")
        print(f"Cluster-{i}")
        if outside:
            print("nearest label")
        print(atlas_label_row)
        print(f"MNI (mm) {mni_mm}")
        print(f"Peak value: {cluster_data[voxel_index]:.2f}, Size: {cluster_size} voxels")
        print("============================================")

    # print the job
    print("\n2. Anatomical label on peak z-value within clusters")
    print(f"Completed")




    # %
    # ================================================================
    # 3. === STEP 3 ===: Count the significant voxels within VWFA mask
    # ================================================================

    # load the thresholded z-map === 
    thresholded_map  = nib.load(output_path)

    # Load VWFA masks
    vwfa_lh_path = TEMP_DIR / 'mask' / 'left_VWFA_pediatric_MNI.nii.gz'
    vwfa_rh_path = TEMP_DIR / 'mask' / 'right_VWFA_pediatric_MNI.nii.gz'
    vwfa_lh = image.load_img(vwfa_lh_path)
    vwfa_rh = image.load_img(vwfa_rh_path)
    vwfa_lh_data = vwfa_lh.get_fdata()
    vwfa_rh_data = vwfa_rh.get_fdata()
    n_lh = np.count_nonzero(vwfa_lh_data)
    n_rh = np.count_nonzero(vwfa_rh_data) 

    # count the overlapping significant voxels
    overlap_lh = image.math_img(
        "(img1 > 0) & (img2 > 0)",
        img1=thresholded_map,
        img2=vwfa_lh
    )

    overlap_rh = image.math_img(
        "(img1 > 0) & (img2 > 0)",
        img1=thresholded_map,
        img2=vwfa_rh
    )
    n_vox_lh = int(np.sum(overlap_lh.get_fdata()))
    n_vox_rh = int(np.sum(overlap_rh.get_fdata()))

    print(f"-------{CONTRAST}-------")
    print(f"VWFA LH overlapping significant voxels: {n_vox_lh}")
    print(f"VWFA LH overlapping volume %: {(n_vox_lh/n_lh)*100:.2f}%")
    print(f"VWFA RH overlapping significant voxels: {n_vox_rh}")
    print(f"VWFA RH overlapping volume %: {(n_vox_rh/n_rh)*100:.2f}%")

    # print the job
    print("\n3. Count the significant voxels within VWFA mask")
    print(f"Completed")




    # %
    # ========================================================================
    # # 4. === STEP 4 ===: 3D topographic surface plots of z-maps (z = -12 mm)
    # ========================================================================

    # ===  Load the z-map ===
    z_map  = nib.load(output_path_unth)
    z_data = z_map.get_fdata()
    affine = z_map.affine

    # ===  load the left VWFA mask ===
    left_vwfa = nib.load(vwfa_lh_path)
    left_vwfa = left_vwfa.get_fdata().astype(np.int16)

    # ===  Clean the dataset ===
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

    # sliced data with based on the z-coordinate
    slice_data = data[:, :, z_idx].copy()  # shape (99, 117)

    # mask of valid voxels (adjust condition if needed)
    valid      = slice_data != 0

    rows       = np.any(valid, axis=1)
    cols       = np.any(valid, axis=0)

    # obtain the range of x and y-axis
    rmin, rmax = 15, 84
    cmin, cmax = 27, 51

    # remove unncessary brain regions
    slice_crop = slice_data[1+rmin:rmax+1, 1+cmin:cmax+1]

    # left VWFA contour
    vwfa_data  = left_vwfa[:, :, z_idx].copy()  # shape (99, 117)
    slice_vwfa = vwfa_data[1+rmin:rmax+1, 1+cmin:cmax+1]
    ZI_vwfa    = slice_vwfa.T


    # === 3D plot prep ===

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



    # === 3D plot ===
    fig   = plt.figure(figsize=(6,4))
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
    ax.set_xlabel('MNI X (mm)', fontsize=10)
    ax.set_ylabel('MNI Y (mm)', fontsize=10)
    ax.set_zlabel('Z-score', fontsize=10)

    # change the view point
    ax.view_init(elev=30, azim=-90)

    # save figure
    path  = FIG_DIR / 'orthography' / MASK
    path.mkdir(exist_ok=True, parents=True)
    plt.tight_layout() # adjusts spacing
    plt.savefig(f'{path}/GRADE-{GRADE}_{CONTRAST}_MNI_z={z_mni}mm.pdf', bbox_inches='tight', transparent=True)  

    # show figure
    plt.show()



    # === visualize anterior shift ===
    ZI_new = ZI_smooth.copy()
    ZI_new[Xmm > 0] = 0

    fig   = plt.figure(figsize=(6,4))
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

    # labels and limits
    ax.set(xlim=(np.min(Xmm)-2, np.max(Xmm)+2), ylim=(np.min(Ymm)-2, np.max(Ymm)+2), zlim=(-3, 3.5))
    ax.set_xlabel('MNI X (mm)', fontsize=10)
    ax.set_ylabel('MNI Y (mm)', fontsize=10)
    ax.set_zlabel('Z-score', fontsize=10)

    # change the view point
    ax.view_init(elev=20, azim=5)

    # save figure
    path  = FIG_DIR / 'orthography' / MASK
    path.mkdir(exist_ok=True, parents=True)
    plt.tight_layout() # adjusts spacing
    plt.savefig(f'{path}/GRADE-{GRADE}_{CONTRAST}_MNI_z={z_mni}mm_anterior.pdf', bbox_inches="tight",transparent=True)  # high-res PNG

    # show figure
    plt.show()






# %%
# =======================================================================
# 5. === STEP 5 ===: Figure 1. recognition of written words in the ventral visual pathway
# =======================================================================

# surface plots
dir = OUT_DIR / 'orthography' / SPACE

conditions = ['word', 'pseudo']
for cond in conditions:
    if cond == 'word':
        nib_path_image = dir / 'images_words' / MASK / f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
        nib_path_audio = dir / 'audios_words' / MASK / f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
    else:
        nib_path_image = dir / 'images_pseudo' / MASK / f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
        nib_path_audio = dir / 'audios_pseudo' / MASK / f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
 

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

        map_data_visual = surface.vol_to_surf(
            z_adult_visual,
            surf_mesh=mesh,
            inner_mesh=white_mesh,
            interpolation='linear',
            n_samples=1   # increase sampling density
        )
        map_data_visual = (map_data_visual > 0).astype(int) * 1

        map_data_audio = surface.vol_to_surf(
            z_adult_audio,
            surf_mesh=mesh,
            inner_mesh=white_mesh,
            interpolation='linear',
            n_samples=1   # increase sampling density
        )
        map_data_audio = (map_data_audio > 0).astype(int) * 2

        # Sum them up
        map_data = map_data_visual + map_data_audio

        roi = map_data.astype(float).copy()
        # keep integer labels for contours
        roi_labels = map_data.astype(int)   # 0,1,2,
        roi[roi == 0] = np.nan   # transparent outside mask

        roi_cmap = ListedColormap([
            (0, 0, 0, 0),            # 0: transparent
            to_rgba("cyan", 0.8),   # 1
            to_rgba("fuchsia", 0.8)   # 2
        ])


        # VWFA map
        if hemi == 'right':
            vwfa_surf = surface.vol_to_surf(
                right_vwfa_adu,
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
                left_vwfa_adu,
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

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        view="ventral"

        plotting.plot_surf_roi(
            surf_mesh=mesh, roi_map=roi, hemi=hemi, view=view,
            bg_map=sulc, cmap=roi_cmap, vmin=0, vmax=2, colorbar=False,
            axes=ax, figure=fig, darkness=None, alpha=1
        )

        # change camera here
        if hemi == 'right':
            ax.view_init(elev=300, azim=0)
        else: 
            ax.view_init(elev=240, azim=0)
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=vwfa_mask,
                levels=[1],
                colors =['gold'],
                linewidths=0,
                axes=ax
            )
            
        plt.tight_layout()
        # save first
        path = FIG_DIR / 'orthography' /  MASK / f"Grade_{GRADE}_{hemi}_surface_{cond}.pdf"
        fig.savefig(path, bbox_inches="tight",transparent=True)

        plt.show() 



# %%
# ==================================================
# 6. === STEP 6 ===: Figure 2. Emergence of the VWFA
# ==================================================
MASK      = 'ventral' #
contrasts = ['images_words', 'images_pseudo', 'audios_words', 'audios_pseudo']
grades    = ['1', '2', '4']
dir       = OUT_DIR / 'orthography' / SPACE

# Load Ventral masks
ventral_left_path  = MASK_DIR / f'left_{MASK}_pediatric_MNI.nii.gz'
ventral_right_path = MASK_DIR / f'right_{MASK}_pediatric_MNI.nii.gz'
ventral_left  = nib.load(ventral_left_path)
ventral_right = nib.load(ventral_right_path)

# Load VWFA masks
vwfa_lh_path = MASK_DIR / 'left_VWFA_pediatric_MNI.nii.gz'
vwfa_rh_path = MASK_DIR / 'right_VWFA_pediatric_MNI.nii.gz'
vwfa_lh      = nib.load(vwfa_lh_path)
vwfa_rh      = nib.load(vwfa_rh_path)
vwfa_lh_data = vwfa_lh.get_fdata()
vwfa_rh_data = vwfa_rh.get_fdata()
n_lh         = np.count_nonzero(vwfa_lh_data)
n_rh         = np.count_nonzero(vwfa_rh_data) 

# 1. Initialize an empty list to store row data
results_data = []

for contrast in contrasts:
    for grade in grades:
        nib_path = dir / contrast / MASK / f"GRADE-{grade}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
        z_map    = nib.load(nib_path)

        # count the overlapping significant voxels in ventral mask
        overlap_lh_ventral = image.math_img(
            "(img1 > 0) & (img2 > 0)",
            img1=z_map,
            img2=ventral_left
        )
        overlap_rh_ventral = image.math_img(
            "(img1 > 0) & (img2 > 0)",
            img1=z_map,
            img2=ventral_right
        )

        # count the overlapping significant voxels in VWFA masks
        overlap_lh = image.math_img(
            "(img1 > 0) & (img2 > 0)",
            img1=z_map,
            img2=vwfa_lh
        )
        overlap_rh = image.math_img(
            "(img1 > 0) & (img2 > 0)",
            img1=z_map,
            img2=vwfa_rh
        )
        # significant voxels with the VWFA mask
        n_vox_lh = int(np.sum(overlap_lh.get_fdata()))
        n_vox_rh = int(np.sum(overlap_rh.get_fdata()))

        # significant voxels with the VWFA mask
        perc_lh = (n_vox_lh/n_lh)*100
        perc_rh = (n_vox_rh/n_rh)*100

        # Efficiency
        total_ventral_lh = int(np.sum(overlap_lh_ventral.get_fdata()))
        total_ventral_rh = int(np.sum(overlap_rh_ventral.get_fdata()))

        # Safe calculation: value if denominator > 0 else 0
        eff_lh = (n_vox_lh / total_ventral_lh * 100) if total_ventral_lh > 0 else 0.0
        eff_rh = (n_vox_rh / total_ventral_rh * 100) if total_ventral_rh > 0 else 0.0

        # Append a dictionary for each iteration
        results_data.append({
            'contrast': contrast,
            'grade': grade,
            'left_sig_voxels': n_vox_lh,
            'right_sig_voxels': n_vox_rh,
            'left_perc_voxels': round(perc_lh, 2),
            'right_perc_voxels': round(perc_rh, 2),
            'left_eff_voxels': round(eff_lh, 2),
            'right_eff_voxels': round(eff_rh, 2)

        })
df = pd.DataFrame(results_data)

# %%
# Grade order setup
grade_order = ['1', '2', '4']
df['grade'] = pd.Categorical(df['grade'], categories=grade_order, ordered=True)
df = df.sort_values('grade')

# We will loop through these to create the two subplots
metrics = [
    {'col_prefix': 'perc_voxels', 'label': 'VWFA overlap (%)', 'yticks': [0, 25, 50, 75, 100], 'ylim': (-5, 105)},
    {'col_prefix': 'eff_voxels',  'label': 'VWFA efficiency (%)',  'yticks': [0, 10, 20, 30], 'ylim': (-5, 35)}
]

# Style Dictionaries
line_styles = {'images_words': '-', 'images_pseudo': '--'}
alphas = {'images_words': 1.0, 'images_pseudo': 0.3}
hemi_colors = {
    'left': 'crimson',
    'right': 'royalblue'
}
# labels = ['A', 'B']

# Create Figure with 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, metric in enumerate(metrics):
    ax = axes[i]
    
    for contrast in ['images_words', 'images_pseudo']:
        subset = df[df['contrast'] == contrast]
        ls = line_styles[contrast]
        a = alphas[contrast]
        name = "Word" if contrast == "images_words" else "Pseudo"
        
        # Determine column names dynamically (e.g., 'left_perc_voxels' vs 'left_efficiency')
        left_col = f"left_{metric['col_prefix']}"
        right_col = f"right_{metric['col_prefix']}"

        # Plot Left Hemisphere
        ax.plot(subset['grade'], subset[left_col],
                marker='o', markersize=8, linewidth=2.5, 
                color=hemi_colors['left'], linestyle=ls, alpha=a, 
                label=f'{name} (left)')
        
        # Plot Right Hemisphere
        ax.plot(subset['grade'], subset[right_col],
                marker='o', markersize=8, linewidth=2.5, 
                color=hemi_colors['right'], linestyle=ls, alpha=a, 
                label=f'{name} (right)')

    # --- AXIS STYLING ---
    ax.set_xlabel('Grade', fontsize=18, labelpad=10)
    ax.set_ylabel(metric['label'], fontsize=18, labelpad=10)
    
    # Custom Ticks for Overlap (Axes[0]), Automatic for Efficiency (Axes[1])
    if metric['yticks']:
        ax.set_yticks(metric['yticks'])
    if metric['ylim']:
        ax.set_ylim(metric['ylim'])

    ax.set_xticks(range(len(grade_order)))
    ax.set_xticklabels(grade_order)
    
    # Physical Ticks
    ax.tick_params(axis='both', which='major', labelsize=11, direction='out', 
                   length=6, width=1.5, bottom=True, left=True)
    
    """     ax.text(
        -0.15, 1, labels[i],            # Position: slightly left and above the axes
        transform=ax.transAxes,         # Use relative axis coordinates
        fontsize=20, fontweight='bold',   # Large, bold font
        va='top', ha='right'              # Alignment
    ) """
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

# We anchor it to the right of the second subplot (axes[1])
# --- LEGEND STYLING: CENTER RIGHT ---
axes[1].legend(
    loc='center left',        # Tells Matplotlib to align the center of the legend...
    bbox_to_anchor=(1.05, 0.5), # ...to the middle (0.5) and right (1.05) of the plot
    fontsize=15,               
    frameon=False              
)
plt.tight_layout()

# --- SAVE THE COMBINED FIGURE ---
path = FIG_DIR / 'orthography' / MASK / "Emergence_VWFA.pdf"
plt.savefig(path, format='pdf', transparent=True, bbox_inches='tight')

plt.show()
# %%
