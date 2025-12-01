# %%
# ===  Load modules ===
from pathlib import Path
import templateflow.api as tflow
import nibabel as nib
import seaborn as sns
import numpy as np
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import datasets
from nilearn.image import resample_to_img
from nilearn import image
from nilearn.maskers import NiftiSpheresMasker
from matplotlib import colors
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_roi, show


# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
TMPL_DIR     = ANALY_DIR / 'templates'

# === Parameters ===
MODEL          = 'glm'
METHOD         = 'parametric' # 'parametric' 'non-parametric'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = 'images_words'
GRADES         = ['1', '2', '4'] # 1, 2, 4
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
CORRECTION     = 'fpr' # fdr, fpr, rft
P_CORRECTION   = 0.001 # 0.001, 0.05
CLUSTER_SIZE   = 10
RADIUS         = 12
MNI_PEAK       = [55.5, -40.5, -14.5]
MASK_TYPE      = 'ventral' # ventral, VWFA, fusiform
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
# === Brain Template ===
# Retrieve the T1-weighted template for cohort 4 (7.5-13.5yrs) at 2mm resolution
TEMPLATE =  tflow.get(
    "MNIPediatricAsym",
    cohort="4",       # cohort 4 = age 7.5-13.5
    resolution=2,     # 2mm isotropic
    suffix="T1w"      # T1-weighted image
)

# === Brain Mask ===
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

# Cerebellum mask
cer_path = TMPL_DIR / 'cerebellar_atlases-master' / 'tpl-MNI152NLin2009cSymC' / 'tpl-MNI152NLin2009cSymC_desc-cereb_mask.nii'
cer_mask = nib.load(cer_path)
# resample cerebellum mask to TEMPLATE space
cer_resampled = resample_to_img(
    source_img=cer_mask,
    target_img=TEMPLATE,
    interpolation='nearest',
    copy_header=True,
    force_resample=True
)
cer_data = cer_resampled.get_fdata().astype(bool)



# === Ventral Mask ===
# Load the reference image
ref_img = nib.load(TEMPLATE)
ref_data = ref_img.get_fdata()
affine = ref_img.affine

# Get the shape of the image
shape = ref_data.shape

# Create an empty mask
mask = np.zeros(shape, dtype=np.uint8)

if MASK_TYPE == 'VWFA':

    # VWFA MNI coordinate
    center = [-45, -57, -12]
    radius = RADIUS  # mm

    ijk = np.indices(shape).reshape(3, -1)
    ijk_h = np.vstack([ijk, np.ones((1, ijk.shape[1]))])
    xyz = (affine @ ijk_h)[:3, :].T

    center = np.asarray(center).reshape(1, 3)
    dist = np.linalg.norm(xyz - center, axis=1)

    mask = (dist <= radius).astype(np.uint8).reshape(shape)

    # Save the mask
    left_mask_img = nib.Nifti1Image(mask, affine)


    # create right hemisphere mask
    left_mask_data  = left_mask_img.get_fdata()
    right_mask_data = np.flip(left_mask_data, axis=0)
    right_mask_img  = nib.Nifti1Image(right_mask_data.astype(np.uint8), affine)

    # remove voxels outside of GM
    left_mask  = (left_mask_data == 1) & (GM_data >= 0.8)
    left_mask  = left_mask.astype(np.uint8)
    right_mask = (right_mask_data == 1) & (GM_data >= 0.8)
    right_mask = right_mask.astype(np.uint8)

    # Save the masks
    left_mask_img  = nib.Nifti1Image(left_mask, affine)
    right_mask_img = nib.Nifti1Image(right_mask, affine)

elif MASK_TYPE == 'fusiform':

    # data
    dataset_ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_ho_filename = dataset_ho.filename
    atlas_img = nib.load(atlas_ho_filename)
    atlas_data = atlas_img.get_fdata().astype(int)
    atlas_affine = atlas_img.affine
    atlas_shape = atlas_img.shape

    # Left and right fusiform label indices in HO atlas
    fusiform_label_anterior  = dataset_ho.labels.index('Temporal Fusiform Cortex, anterior division') 
    fusiform_label_posterior = dataset_ho.labels.index('Temporal Fusiform Cortex, posterior division') 
    fusiform_label_tempoocc  = dataset_ho.labels.index('Temporal Occipital Fusiform Cortex') 
    fusiform_label_infetempo = dataset_ho.labels.index('Inferior Temporal Gyrus, temporooccipital part') 

    # Create a binary fusiform mask (left, right, or bilateral)
    fusiform_mask = (
        (atlas_data == fusiform_label_anterior) | 
        (atlas_data == fusiform_label_posterior) | 
        (atlas_data == fusiform_label_tempoocc)  |
        (atlas_data == fusiform_label_infetempo)
    ) 
    fusiform_mask      = fusiform_mask.astype(np.uint8)
    fusiform_mask_flat = fusiform_mask.flatten()

    ijk = np.indices(atlas_shape).reshape(3, -1)
    ijk_h = np.vstack([ijk, np.ones((1, ijk.shape[1]))])
    xyz = (affine @ ijk_h)[:3, :].T

    # Left hemisphere: x < 0 (negative MNI x)
    left_mask_flat = fusiform_mask_flat & (xyz[:, 0] < 0)

    # Right hemisphere: x > 0 (positive MNI x)
    right_mask_flat = fusiform_mask_flat & (xyz[:, 0] > 0)

    # Reshape back to original volume
    left_mask = left_mask_flat.reshape(atlas_shape).astype(np.uint8)
    right_mask = right_mask_flat.reshape(atlas_shape).astype(np.uint8)

    # left fusiform mask
    left_mask_img = resample_to_img(
        nib.Nifti1Image(left_mask, atlas_affine),
        ref_img,
        interpolation='nearest',
        copy_header=True,
        force_resample=True
    )
    left_mask = left_mask_img.get_fdata().astype(np.uint8)

    # left fusiform mask
    right_mask_img = resample_to_img(
        nib.Nifti1Image(right_mask, atlas_affine),
        ref_img,
        interpolation='nearest',
        copy_header=True,
        force_resample=True
    )
    right_mask = right_mask_img.get_fdata().astype(np.uint8)



elif MASK_TYPE == 'ventral':
    # Define your MNI coordinate for left hemisphere
    x_min, x_max = -70, -20
    y_min, y_max = -82, -33
    z_min, z_max = -28, 4

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

    # Save the mask
    left_mask_img = nib.Nifti1Image(mask, affine)


    # create right hemisphere mask
    left_mask_data  = left_mask_img.get_fdata()
    right_mask_data = np.flip(left_mask_data, axis=0)
    right_mask_img  = nib.Nifti1Image(right_mask_data.astype(np.uint8), affine)

    # remove voxels outside of GM
    left_mask  = (left_mask_data == 1) & (GM_data >= 0.8) & (~cer_data)
    left_mask  = left_mask.astype(np.uint8)
    right_mask = (right_mask_data == 1) & (GM_data >= 0.8) & (~cer_data)
    right_mask = right_mask.astype(np.uint8)

    # Save the masks
    left_mask_img  = nib.Nifti1Image(left_mask, affine)
    right_mask_img = nib.Nifti1Image(right_mask, affine)

# count the total volume
voxel_sizes      = GM_img.header.get_zooms()  # should be (2.0, 2.0, 2.0)
voxel_volume     = voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]
left_volume_mm3  = voxel_volume * np.sum(left_mask)
right_volume_mm3 = voxel_volume * np.sum(right_mask)

# print basic info
print("\nLeft hemisphere")
print(f"Mask created with shape: {left_mask.shape}")
print(f"Number of voxels in mask: {np.sum(left_mask)}")
print(f"Total volume: {left_volume_mm3} mm³")
print("\nRight hemisphere")
print(f"Mask created with shape: {right_mask.shape}")
print(f"Number of voxels in mask: {np.sum(right_mask)}")
print(f"Total volume: {right_volume_mm3} mm³")




# %%
# === Visualize Mask ===

# directory
dir = f"{FIG_DIR}/{MODEL}"

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

# plot the uncorrrected p-value figure 
display = plotting.plot_roi(
    right_mask_img,
    # title="Ventral Mask",
    **plotting_config,
)
# Second mask (right) added to the same figure
display.add_overlay(
    left_mask_img,
    cmap=blue_cmap,
)
plotting.show()
display.savefig(f"{dir}/{MASK_TYPE}_mask.png", dpi=300)
print(f"\nSuccessful: Figure of the {MASK_TYPE} mask is saved ")


# %%
# === Compute total volume (mm3) ===
results = []

for GRADE in GRADES:

    # list the beta nii.files
    subjects      = sorted(DERIV_DIR.glob(f"sub-{GRADE}*"))
    exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
    subjects      = [s for s in subjects if s.name not in exclude]
    subject_names = [p.name.replace('sub-', '') for p in subjects]


    for sub in subjects:
        path      = sub / 'glm' / SPACE / f"FWHM_{int(FWHM_SMOOTHING)}" 

        # beta maps
        b_path    = next(path.glob(f"{CONTRASTS}_beta.nii.gz"), None)
        b_img     = nib.load(str(b_path))
        b_data    = b_img.get_fdata()

        # z maps
        z_path    = next(path.glob(f"{CONTRASTS}_zscore.nii.gz"), None)
        z_img     = nib.load(str(z_path))
        z_data    = z_img.get_fdata()

        # z-value in the peak MNI coordinates
        # --- Convert MNI coordinate -> voxel coordinate ---
        affine     = b_img.affine
        inv_affine = np.linalg.inv(affine)
        # Add homogeneous coordinate 1
        voxel_coord = inv_affine @ np.append(MNI_PEAK, 1)
        # Take first 3 elements, round to nearest voxel
        i, j, k = np.round(voxel_coord[:3]).astype(int)
        # Safety check: ensure voxel is inside bounds
        if (0 <= i < b_data.shape[0] and
            0 <= j < b_data.shape[1] and
            0 <= k < b_data.shape[2]):
            b_value = b_data[i, j, k]
        else:
            b_value = np.nan
            print(f"Warning: voxel out of bounds for subject {sub}")

        print("beta =", b_value)

        # average beta-values within the mask
        b_left = b_data[left_mask.astype(bool)]
        b_right = b_data[right_mask.astype(bool)]
        mean_b_left = np.mean(b_left)
        mean_b_right = np.mean(b_right)

        # apply p-value threshold
        z_map_thr, threshold = threshold_stats_img(
            z_img,
            alpha=P_CORRECTION,
            height_control=CORRECTION,
            cluster_threshold=CLUSTER_SIZE,
            two_sided=False,  # using a one-sided test
        )
        z_thr_data = z_map_thr.get_fdata()

        # Only keep voxels that are in the mask
        left_significant_mask  = (z_thr_data != 0) & left_mask
        right_significant_mask = (z_thr_data != 0) & right_mask

        # count the significant voxels within the ventral mask
        n_left_sig  = int(left_significant_mask.sum())
        n_right_sig = int(right_significant_mask.sum())

        # total volume computation
        sig_left_volume  = voxel_volume * np.sum(left_significant_mask)
        sig_right_volume = voxel_volume * np.sum(right_significant_mask)

        # percentage of significant volume within the mask
        perc_sig_left = (sig_left_volume / left_volume_mm3) * 100
        perc_sig_right = (sig_right_volume / right_volume_mm3) * 100

        # add a row to the results
        results.append({
            "grade": GRADE,
            "subject": sub.name,
            "n_left_sig": n_left_sig,
            "n_right_sig": n_right_sig,
            "sig_left_volume": sig_left_volume,
            "sig_right_volume": sig_right_volume,
            "perc_sig_left": perc_sig_left,
            "perc_sig_right": perc_sig_right,
            "mean_b_left": mean_b_left,
            "mean_b_right": mean_b_right,
            "peak_b": b_value,
        })

        print(f"Successful: {sub.name}")

df = pd.DataFrame(results)


# %%
# ======== STATISTICS ========

# ======== ANOVA one-way =========
# ======== Peak coordinates =========

df_peak = df[["grade", "subject", "peak_b"]]
means = df_peak.groupby(["grade"])["peak_b"].mean()
ses = df_peak.groupby(["grade"])["peak_b"].sem()
print("=========================================")
print(f"\nDependent variable: beta-value in the peak MNI cooedinates\n")
print("mean")
print(means)
print("\nstandard error of the mean")
print(ses)
print("=========================================")

# parametric assumption satisfied?
results = []
for g in df_peak['grade'].unique():
    subset = df_peak[df_peak['grade'] == g]
    res = pg.normality(subset["peak_b"])
    res['grade'] = g
    results.append(res)
normality_table = pd.concat(results, ignore_index=True)
print("=========================================")
print(f"\nNormality test:\n")
print(normality_table)
print("=========================================")

### ======= Parametric test ======= 
# one-way ANOVA
anova_res = pg.anova(
    data=df_peak,
    dv="peak_b",
    between="grade",
    detailed=True
)
print("=========================================")
print(f"\nContrast: {CONTRASTS}\n")
print("\nOne-way ANOVA\n")
print(anova_res)
print("=========================================")

# pairwise comparisons
pw_grade = pg.pairwise_tests(
    data=df_peak,
    dv="peak_b",
    between="grade",
    padjust="fdr_bh",     # Bonferroni correction (or 'fdr_bh')
    effsize="hedges"
)
print("=========================================")  
print("\nParametric\n")
print("=========================================")  
print("=========================================") 
print("Pairwise grade comparisons:\n")
print(pw_grade)
print("=========================================") 

# === Visualization ===
# X positions
x = np.arange(len(means.index))  # number of grade levels

fig, ax = plt.subplots(figsize=(8,6))
sns.violinplot(
    data=df_peak, 
    x="grade", 
    y="peak_b", 
    inner=None,
    linewidth=1.2,
    color="gold",
    edgecolor="black", 
    ax=ax
    )
sp = sns.stripplot(
    data=df_peak, 
    x="grade", 
    y="peak_b", 
    color = "white",
    edgecolor="black",
    dodge=False, 
    size=10,
    ax=ax
    )

# Remove fill color from all stripplot markers
for artist in sp.collections:
    artist.set_facecolor("gold")   # <-- hollow markers
    artist.set_linewidth(1.2)
    offsets = artist.get_offsets()
    offsets[:, 0] += 0.15 
    artist.set_offsets(offsets)

ax.errorbar(
    x,
    means,
    yerr=ses,
    marker="o",
    linestyle="-",
    color="black"
)

ax.set_title(f"MNI coordinates: {MNI_PEAK}", size=15)
ax.set_ylim(-2,2)
ax.set_ylabel("Effect size (β-value)", size=13)
ax.set_xlabel("Grade", size=13)
plt.tight_layout()
plt.savefig(f"{dir}/{MASK_TYPE}_peak-b.png", dpi=300)
plt.show()



# ======== ANOVA two-way =========
# ======== average and volume =========
df_vol = pd.melt(
    df,
    id_vars=["subject", "grade"],
    value_vars=["sig_left_volume", "sig_right_volume"],
    var_name="hemisphere",
    value_name="total_volume")

# Make hemisphere labels clean
df_vol["hemisphere"] = df_vol["hemisphere"].replace({
    "sig_left_volume": "left",
    "sig_right_volume": "right"
})
df_z = pd.melt(
    df,
    id_vars=["subject", "grade"],
    value_vars=["mean_b_left", "mean_b_right"],
    var_name="hemisphere",
    value_name="mean_b")

# Make hemisphere labels clean
df_z["hemisphere"] = df_z["hemisphere"].replace({
    "mean_b_left": "left",
    "mean_b_right": "right"
})
df_long = df_vol.merge(df_z, on=["subject", "grade", "hemisphere"])


dependents = ["total_volume", "mean_b"]
for dependent in dependents:

    means = df_long.groupby(["grade", "hemisphere"])[dependent].mean().unstack()

    # Compute standard errors
    ses = df_long.groupby(["grade", "hemisphere"])[dependent].sem().unstack()

    print("=========================================")
    print(f"\nDependent variable: {dependent}\n")
    print("mean")
    print(means)
    print("standard error of the mean")
    print(ses)
    print("=========================================")

    # parametric assumption satisfied?
    results = []

    for g in df_long['grade'].unique():
        for h in df_long['hemisphere'].unique():
            subset = df_long[(df_long['grade'] == g) &
                            (df_long['hemisphere'] == h)]

            res = pg.normality(subset[dependent])
            res['grade'] = g
            res['hemisphere'] = h
            results.append(res)

    normality_table = pd.concat(results, ignore_index=True)
    print("=========================================")
    print(f"\nNormality test:\n")
    print(normality_table)
    print("=========================================")

    ### ======= Parametric test ======= 
    # two-way mixed ANOVA
    anova_res = pg.mixed_anova(
        data=df_long,
        dv=dependent,
        within="hemisphere",
        between="grade",
        subject="subject"
    )
    print("=========================================")
    print(f"\nContrast: {CONTRASTS}\n")
    print("\nTwo-way mixed ANOVA\n")
    print(anova_res)
    print("=========================================")


    # pairwise comparisons
    pw_grade = pg.pairwise_tests(
        data=df_long,
        dv=dependent,
        between="grade",
        within="hemisphere",
        subject="subject",
        padjust="fdr_bh",     # Bonferroni correction (or 'fdr_bh')
        effsize="hedges"
    )
    print("=========================================")  
    print("\nParametric\n")
    print("=========================================")  
    print("=========================================") 
    print("Pairwise grade comparisons (within each hemisphere):\n")
    print(pw_grade)
    print("=========================================") 


    ### ======= Non-Parametric test ======= 
    # Mann-Whitney U test
    results = []
    for hemi in df_long['hemisphere'].unique():
        df_h = df_long[df_long['hemisphere'] == hemi]
        res = pg.pairwise_tests(
            data=df_h,
            dv=dependent,
            between='grade',
            parametric=False,     # <-- rank-based
            padjust='fdr_bh'
        )
        res['hemisphere'] = hemi
        results.append(res)

    pw_rank_grade = pd.concat(results, ignore_index=True)
    print("=========================================")  
    print("\nNon-Parametric\n")
    print("=========================================") 
    print("\nMann-Whitney Pairwise grade comparisons (within each hemisphere):\n")
    print(pw_rank_grade)
    print("=========================================") 

    """     # Wilkoxon paired test
    results = []
    for grade in df_long['grade'].unique():
        df_h = df_long[df_long['grade'] == grade]
        res = pg.pairwise_tests(
            data=df_h,
            dv=dependent,
            within='hemisphere',
            parametric=False,     # <-- rank-based
            subject='subject',
            padjust="fdr_bh"
        )
        res['grade'] = grade
        results.append(res)

    pw_rank_grade = pd.concat(results, ignore_index=True)
    print("=========================================") 
    print("\nWilcoxon Pairwise  hemispheric comparisons (within each grade):\n")
    print(pw_rank_grade)
    print("=========================================") 
    """

    # === Visualization ===
    # X positions
    x = np.arange(len(means.index))  # number of grade levels

    # color specification
    hemi_colors = {
        "left": 'springgreen',  
        "right":'dodgerblue'   # marine blue
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, hemi, title in zip(
        axes,
        ["left", "right"],
        ["Left hemisphere", "Right hemisphere"]
    ):
        sns.violinplot(
            data=df_long[df_long["hemisphere"]==hemi], 
            x="grade", 
            y=dependent, 
            inner=None,
            linewidth=1.2,
            color=hemi_colors[hemi],
            edgecolor="black",
            ax=ax)
        sp = sns.stripplot(
            data=df_long[df_long["hemisphere"]==hemi], 
            x="grade", 
            y=dependent, 
            color = "white",
            edgecolor="black",
            dodge=False, 
            size=10,
            ax=ax
            )
        # Remove fill color from all stripplot markers
        for artist in sp.collections:
            artist.set_facecolor(hemi_colors[hemi])   # <-- hollow markers
            artist.set_linewidth(1.2)
            offsets = artist.get_offsets()
            offsets[:, 0] += 0.15 
            artist.set_offsets(offsets)
        
        ax.errorbar(
            x,
            means[hemi],
            yerr=ses[hemi],
            marker="o",
            linestyle="-",
            color="black"
        )
    
        ax.set_title(title, size=15)
        if dependent == "total_volume":
            ax.set_ylim(0,30000)
            ax.set_ylabel("Total volume (mm³)", size=13)
        elif dependent == "mean_b":
            ax.set_ylim(-1,1)
            ax.set_ylabel("Average effect size (β-value)", size=13)
        ax.set_xlabel("Grade", size=13)

    plt.tight_layout()
    plt.savefig(f"{dir}/{MASK_TYPE}_{dependent}.png", dpi=300)
    plt.show()

# %%
