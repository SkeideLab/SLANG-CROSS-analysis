# %%
# ===  Load modules ===
from nilearn import plotting, surface, datasets
from pathlib import Path
import numpy as np
import templateflow.api as tflow
import nibabel as nib
import ants
import matplotlib.pyplot as plt
from scipy import ndimage
import plotly.graph_objects as go

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
CONTRASTS      = 'audios_words'
GRADES         = '4' # 1, 2, 4
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
P_CORRECTION   = 0.05 # 0.001, 0.05
CLUSTER_SIZE   = 15
MASK_TYPE      = 'STG' # ventral, VWFA, A1, IFG, MTG, STG

# %%
# ===  Datasets ===

# pediatric MNI 4 (7.5-13.5yrs) at 2mm resolution
TEMPLATE     =  tflow.get(
    "MNIPediatricAsym",
    cohort="4",       # cohort 4 = age 7.5-13.5
    resolution=2,     # 2mm isotropic
    suffix="T1w"      # T1-weighted image
)
# adult MNI T1w
Adult_MNI_T1 = tflow.get(
    'MNI152NLin2009cAsym', 
    resolution=2,
    desc='brain', 
    suffix='T1w')

# z-map in pediatric MNI space
f_name   =  f"GRADE-{GRADES}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
nib_path = OUT_DIR / MODEL / SPACE / CONTRASTS / MASK_TYPE / f_name

z_img    = nib.load(nib_path)
z_data   = z_img.get_fdata().astype(dtype=np.float32)



# %%
# ===  Peak coordinates ===
peak_coords = {
    "STG": {
        "1": [
            {"hemi": "left",  "coord": (-68, -10, 1.5)},
            {"hemi": "right", "coord": (71, -32, 13)}
        ],
        "2": [
            {"hemi": "left",  "coord": (-66, -36, 3)},
            {"hemi": "right", "coord": (69, -18, 5)}
        ],
        "4": [
            {"hemi": "left",  "coord": (-58, -24, 1)},
            {"hemi": "right", "coord": (63, -22, 3)}
        ],
    },
}



# %%
# ===  Plot on Pediatric MNI ===
for grade, peaks in peak_coords[MASK_TYPE].items():

    if grade != GRADES:
        continue

    left_coord  = next(p["coord"] for p in peaks if p["hemi"] == "left")
    right_coord = next(p["coord"] for p in peaks if p["hemi"] == "right")

    marker_coords = {
        "left": left_coord,
        "right": right_coord,
    }
    for hemi, marker_coord in marker_coords.items():
        plotting_config = {
            "display_mode": "ortho",
            "cut_coords": marker_coord,
            "draw_cross": True,
            "vmax": 5,
            "vmin": 0,
            "cmap": "hot",
        }

        display = plotting.plot_stat_map(
            z_img,
            bg_img=TEMPLATE,
            **plotting_config,
            # title="peak"
        )
        # Resize colorbar
        if display._colorbar_ax is not None:
            pos = display._colorbar_ax.get_position()
            display._colorbar_ax.set_position([
                pos.x0, 
                (pos.y0 + pos.x0) / 4, 
                pos.width * 0.6,   # shrink width
                pos.height * 0.6   # shrink height
            ])

        # Add peak markers
        display.add_markers(
            marker_coords=[left_coord],
            marker_color='springgreen',
            marker_size=20
        )
        # Add peak markers
        display.add_markers(
            marker_coords=[right_coord],
            marker_color='dodgerblue',
            marker_size=20
        )
        # show
        plotting.show()

        # % === save the figure ===
        # -------------------------------
        path = FIG_DIR / MODEL / MASK_TYPE 
        path.mkdir(exist_ok=True, parents=True)

        display.savefig(f"{path}/Grade_{grade}_{CONTRASTS}_{hemi}_peak.png", dpi=300)
        print(f"\nSuccessful: The {hemi} hemisphere figure is saved ")




# %%
# ===  Plot on pial surface MNI ===
z_ants   = ants.image_read(str(nib_path))

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

# convert from Pediatric to Adult
z_adult = ants.apply_transforms(
    fixed=adult_mni,
    moving=z_ants,
    transformlist=inverse_transforms,
    whichtoinvert=[False, True],
    interpolator='linear' 
)
# convert from ants to nifti
z_adult_data = z_adult.numpy().astype(np.float32)
z_adult = nib.Nifti1Image(z_adult_data, adult_mni_nifti.affine)


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

    # Find the vertex with the maximum value
    max_vertex = np.argmax(map_data)
    max_coord  = mesh.coordinates[max_vertex]  # mesh.coordinates from load_surf_mesh

    # surface plot
    surf_fig = plotting.plot_surf_stat_map(
        mesh, map_data, hemi=hemi,
        view='lateral',
        threshold=1,
        colorbar=False,
        bg_map=sulc,
        engine='plotly',
        cmap='hot',
        vmin=0,
        vmax=4
    )

    # marker plot
    surf_fig.figure.add_trace(go.Scatter3d(
        x=[max_coord[0]], 
        y=[max_coord[1]], 
        z=[max_coord[2]],
        mode='markers',
        marker=dict(color=color, size=15, opacity=0.9,line=dict(
                color='black',       # border color
                width=2              # border thickness
            )),
        name='Maximum'
    ))

    # show
    surf_fig.show()





# %% ===  Plot only peaks on pial surface MNI ===
# ===  Plot on pial surface MNI ===

GRADES = ['1', '2', '4']

# List to store results
z_adults_list = []

for GRADE in GRADES:
    f_name   =  f"GRADE-{GRADE}_FWHM-{int(FWHM_SMOOTHING)}_p<{P_CORRECTION}_cls>{CLUSTER_SIZE}_z-map.nii.gz"
    nib_path = OUT_DIR / MODEL / SPACE / CONTRASTS / MASK_TYPE / f_name

    z_ants   = ants.image_read(str(nib_path))

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

    # convert from Pediatric to Adult
    z_adult = ants.apply_transforms(
        fixed=adult_mni,
        moving=z_ants,
        transformlist=inverse_transforms,
        whichtoinvert=[False, True],
        interpolator='linear' 
    )
    # convert from ants to nifti
    z_adult_data = z_adult.numpy().astype(np.float32)
    z_adult_nifti = nib.Nifti1Image(z_adult_data, adult_mni_nifti.affine)

    # Store in list
    z_adults_list.append(z_adult_nifti)


hemis = ['right','left']
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
    map_data_1 = surface.vol_to_surf(
        z_adults_list[0],
        surf_mesh=mesh,
        inner_mesh=white_mesh,
        interpolation='linear',
        n_samples=20   # increase sampling density
    )
    map_data_2 = surface.vol_to_surf(
        z_adults_list[1],
        surf_mesh=mesh,
        inner_mesh=white_mesh,
        interpolation='linear',
        n_samples=20   # increase sampling density
    )
    map_data_4 = surface.vol_to_surf(
        z_adults_list[2],
        surf_mesh=mesh,
        inner_mesh=white_mesh,
        interpolation='linear',
        n_samples=20   # increase sampling density
    )

    # Find the vertex with the maximum value
    max_vertex_1 = np.argmax(map_data_1)
    max_coord_1  = mesh.coordinates[max_vertex_1]  # mesh.coordinates from load_surf_mesh

    # Find the vertex with the maximum value
    max_vertex_2 = np.argmax(map_data_2)
    max_coord_2  = mesh.coordinates[max_vertex_2]  # mesh.coordinates from load_surf_mesh

    # Find the vertex with the maximum value
    max_vertex_4 = np.argmax(map_data_4)
    max_coord_4  = mesh.coordinates[max_vertex_4]  # mesh.coordinates from load_surf_mesh

    # surface plot 
    surf_fig = plotting.plot_surf(
        mesh, 
        bg_map=sulc,
        hemi=hemi,
        view='lateral',
        engine='plotly',
        colorbar=False,
        darkness=0.5  
    )

    # marker plot
    surf_fig.figure.add_trace(go.Scatter3d(
        x=[max_coord_1[0]], 
        y=[max_coord_1[1]], 
        z=[max_coord_1[2]],
        mode='markers',
        showlegend=False,
        marker=dict(color="darkgoldenrod", size=20, opacity=1,line=dict(
                color='black',       # border color
                width=2              # border thickness
            ))
    ))
    # marker plot
    surf_fig.figure.add_trace(go.Scatter3d(
        x=[max_coord_2[0]], 
        y=[max_coord_2[1]], 
        z=[max_coord_2[2]],
        mode='markers',
        showlegend=False,
        marker=dict(color="darkcyan", size=20, opacity=1,line=dict(
                color='black',       # border color
                width=2              # border thickness
            ))
    ))
    # marker plot
    surf_fig.figure.add_trace(go.Scatter3d(
        x=[max_coord_4[0]], 
        y=[max_coord_4[1]], 
        z=[max_coord_4[2]],
        mode='markers',
        showlegend=False,
        marker=dict(color="firebrick", size=20, opacity=1,line=dict(
                color='black',       # border color
                width=2              # border thickness
            ))
    ))


    # show
    surf_fig.show()

# %%






""" # %% ===  Plot on inflated surface MNI ===

hemis = ['right', 'left']
for hemi in hemis:

    if hemi == 'right':
        proj_mesh = surface.load_surf_mesh(fsaverage.pial_right)   # projection mesh
        white_mesh = surface.load_surf_mesh(fsaverage.white_right)
        plot_mesh = surface.load_surf_mesh(fsaverage.infl_right)   # display mesh
        sulc = fsaverage.sulc_right
        color = 'dodgerblue'
    else:
        proj_mesh  = surface.load_surf_mesh(fsaverage.pial_left)
        white_mesh = surface.load_surf_mesh(fsaverage.white_left)
        plot_mesh = surface.load_surf_mesh(fsaverage.infl_left)
        sulc = fsaverage.sulc_left
        color = 'springgreen'

    # project volumetric data (IMPORTANT: use pial mesh)
    map_data = surface.vol_to_surf(z_adult, surf_mesh= proj_mesh, inner_mesh=white_mesh)

    # vertex of maximum
    max_vertex = np.argmax(map_data)
    max_coord = plot_mesh.coordinates[max_vertex]

    # maximize the contrast
    bg_data = surface.load_surf_data(sulc)
    binary_bg = np.where(bg_data > 0, 1, -1) # 凸凹を1と-1に固定

    # plot on inflated surface
    surf_fig = plotting.plot_surf_stat_map(
        plot_mesh, map_data,
        hemi=hemi,
        view='lateral',
        threshold=1,
        bg_map=binary_bg,
        engine='plotly',
        cmap='hot',
        vmin=0,
        vmax=4,
        colorbar=False,
        darkness=0.4
    )

    surf_fig.figure.add_trace(go.Scatter3d(
        x=[max_coord[0]],
        y=[max_coord[1]],
        z=[max_coord[2]],
        mode='markers',
        marker=dict(color=color, size=25, opacity=0.9,
                    line=dict(color='black', width=2))
    ))

    surf_fig.show()

 """