# %% [markdown]
# ## fMRI Group-level RSA analysis (2nd-level)
#
# === STEP 1 ===: Extract the beta maps for each subject
# === STEP 2 ===: Pick ROI, and extract the corresponding voxel, convert it into a vector
# === STEP 3 ===: Compute pearson correlations (r) between two modalities(written and spoken)
# === STEP 4 ===: Fisher transformation
# === STEP 5 ===: Save it as csv file
# === STEP 6 ===: Multiple linear regression
# === STEP 7 ===: Plot significant ROI on surface
# === STEP 8 ===: Plot fisher transformed similarity metrics across grades
# === STEP 9 ===: Plot fisher transformed similarity metrics across accuracy

# %%
# === Packages ===
from   pathlib import Path
from   statsmodels.stats.multitest import multipletests
from   statsmodels.stats.anova import anova_lm
from   scipy.stats import pearsonr
from   itertools import combinations
from   nilearn import plotting, image, datasets, surface
from   matplotlib.colors import ListedColormap, to_rgba
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.formula.api as smf
import ptitprince as pt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors


# %%
# ===  FIXED: Parameters ===
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
DEMO_DIR     = ANALY_DIR / 'demographics'
TEMP_DIR     = ANALY_DIR / 'templates'
MASK_DIR     = TEMP_DIR / 'mask'

# === Parameters ===
MODEL          = 'glm'
SPACE          = 'MNIPediatricAsym_cohort-4_res-2'
CONTRASTS      = [
                'images_psuedo',
                'audios_pseudo', 
                ]
GRADES         = ['1', '2', '4'] # 1, 2, 4 or all
FWHM_SMOOTHING = 9.0 # 6.0, 9.0, 12.0
EXC_SUBJECTS   = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434']
HO_ATLAS_MNI6 = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm') # Harvard-Oxford MNI6Asym
atlas_labels = HO_ATLAS_MNI6.lut 




# %%
# ======================================================
# === STEP 1 ===: Extract the beta maps for each subject
# ======================================================

all_correlations = {}

# Get the list of names from the atlas (skipping 'Background')
roi_names = [label for label in atlas_labels['name'] if label != 'Background']

for MASK in roi_names:
    all_correlations[MASK] = {'left': {}, 'right': {}}

    for GRADE in GRADES:
        subjects      = sorted(DERIV_DIR.glob(f"sub-{GRADE}*"))
        exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
        subjects      = [s for s in subjects if s.name not in exclude]
        subject_names = [p.name.replace('sub-', '') for p in subjects]

        # get the beta paths
        beta_paths = {contrast: {} for contrast in CONTRASTS}
        for contrast in CONTRASTS:
            beta_lists = []
            for sub in subjects:
                path = sub / 'glm' / SPACE / f"FWHM_{int(FWHM_SMOOTHING)}" 
                beta_path = next(path.glob(f"{contrast}_beta.nii.gz"), None)
                beta_lists.append(beta_path)
            beta_paths[contrast] = [str(p) for p in beta_lists]


        # %
        # ===============================================
        # === STEP 2 ===: Pick ROI, and extract the betas
        # ===============================================

        vector = {MASK: {}}
        for hemi in ['right', 'left']:
            vector[MASK][hemi] = {}
            # roi mask for each hemisphere
            roi_path  = MASK_DIR / f"{hemi}_{MASK}_pediatric_MNI.nii.gz"
            roi       = nib.load(roi_path)
            roi_data  = roi.get_fdata().astype(bool)

            for contrast in CONTRASTS:
                vector[MASK][hemi][contrast] = {}

                for sub_name, beta_path in zip(subject_names, beta_paths[contrast]):

                    beta_img  = nib.load(beta_path)
                    beta_data = beta_img.get_fdata()
                            
                    # Apply mask and extract voxels
                    roi_voxels  = beta_data[roi_data]
                    
                    # Convert to 1D vector
                    roi_vector  = roi_voxels.flatten()
                    
                    vector[MASK][hemi][contrast][sub_name] = roi_vector


        # %
        # ====================================================================================
        # === STEP 3 ===: Compute pearson correlations (r) between written and spoken modality
        # ====================================================================================

        for hemi in ['left', 'right']:
            
            z_vals = []
            for sub in subject_names:
                # Stack vectors across contrasts
                contrast_vectors = []

                for contrast in CONTRASTS:
                    vec = vector[MASK][hemi][contrast][sub]
                    contrast_vectors.append(vec)

                contrast_matrix = np.vstack(contrast_vectors)

                # Compute Pearson correlation matrix
                corr_matrix = np.corrcoef(contrast_matrix)

                # Extract the specific correlation of interest (r)
                r_val = corr_matrix[1, 0]
                

                # %
                # =====================================
                # === STEP 4 ===: Fisher transformation
                # =====================================
                z_val = np.arctanh(np.clip(r_val, -0.999, 0.999))
                z_vals.append(z_val)

            # Store the Z-score instead of the r
            all_correlations[MASK][hemi][GRADE] = z_vals

# %
# =====================================
# === STEP 5 ===: save it as csv file
# =====================================

# convert nested directory to pandas dataframe
rows = []

# Iterate through the nested dictionary
for roi, hemis in all_correlations.items():
    for hemi, grades in hemis.items():
        for grade, z_list in grades.items():
            # Iterate through the list of z-scores for this group
            for i, z_val in enumerate(z_list):
                rows.append({
                    'ROI': roi,
                    'Hemisphere': hemi,
                    'Grade': grade,
                    'Subject_Idx': i, # Index of the subject in that grade group
                    'Fisher_Z': float(z_val)
                })

# Create the DataFrame
df_corrs = pd.DataFrame(rows)

# Define the filename
f_name = f'fisher_z_correlations_{CONTRASTS[0]}.csv'
output_filename = OUT_DIR / 'rsa' / f_name
output_filename.parent.mkdir(exist_ok=True, parents=True)

# Save the DataFrame
df_corrs.to_csv(output_filename, index=False)
print(f"Successfully saved results to: {output_filename}")

# Check the first few rows
print(df_corrs.head())



# %%
# ==========================================
# === STEP 6 ===: Multiple linear regression
# ==========================================
# Similarity = β0​ + β1​*(grade) + β2​*(task accuracy) + ε

# Define the filename
f_name = f'fisher_z_correlations_images_words.csv'
condition = f_name.split('_')[-1].replace('.csv', '')
output_filename = OUT_DIR / 'rsa' / f_name
# Read the file
df_results = pd.read_csv(output_filename)

subjects      = sorted(DERIV_DIR.glob("sub-*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]

results = []
for sub in subjects:
    task_path = sub / 'behavior' / 'accuracy_summary.csv'
    task_df  = pd.read_csv(task_path)
    mean_acc = task_df['accuracy_all'].mean()
    # Store results as a dict
    results.append({
        'subject': sub.name,
        'mean_accuracy': mean_acc
    })
# Convert list of dicts to a DataFrame
task_df = pd.DataFrame(results)

# Use all ROIs (not a single TARGET_ROI) for multiple-testing correction
df = df_results.copy()
df["Hemisphere"] = df["Hemisphere"].str.lower()
grade_map = {1:1, 2:2, 4:3}
df["grade_rank"] = df["Grade"].map(grade_map)

rows = []
for roi in df["ROI"].dropna().unique():
    for hemi in ["left", "right"]:
        d = df[(df["ROI"] == roi) & (df["Hemisphere"] == hemi)].copy()
        # d = d.dropna(subset=["Fisher_Z", "grade_rank"])
        # Assuming 'subject' column exists in both df and task_df
        d['mean_acc'] = task_df['mean_accuracy'].values

        m = smf.ols("Fisher_Z ~ grade_rank + mean_acc", data=d).fit()
        # print(f"\n=== {hemi.upper()}{roi} ===")
        # print(m.summary())

        rows.append({
            "ROI": roi,
            "Hemisphere": hemi,
            "n": int(m.nobs),
            "beta_grade": m.params["grade_rank"],
            "p_grade": m.pvalues["grade_rank"],
            "beta_acc": m.params["mean_acc"],
            "p_acc": m.pvalues["mean_acc"],
            "r2": m.rsquared
        })

res = pd.DataFrame(rows)

# FDR (Benjamini-Hochberg) across ALL ROI x hemisphere tests
res_list = []

# Whole-brain FDR: across all ROI x hemisphere tests together
res_corrected = res.copy()


# FDR for grade effect
factors = ['grade', 'acc']
for factor in factors:
    rej_g, q_g, _, _ = multipletests(res_corrected[f"p_{factor}"], alpha=0.05, method="fdr_bh")
    res_corrected[f"q_{factor}_fdr"] = q_g
    res_corrected[f"sig_{factor}_fdr_0.05"] = rej_g

    res_corrected = res_corrected.sort_values([f"q_{factor}_fdr"]).reset_index(drop=True)

    # Filter for only significant results
    significant_results = res_corrected[res_corrected[f"sig_{factor}_fdr_0.05"] == True]

    # Select and print relevant columns
    print(f"=== ROIs showing a significant {factor.upper()} effect (FDR < 0.05) ===")
    if not significant_results.empty:
        print(significant_results[["ROI", "Hemisphere", f"beta_{factor}", f"p_{factor}", f"q_{factor}_fdr"]])
    else:
        print("No ROIs survived FDR correction.")

    # Optional: To see how many survived vs total tested
    print(f"\nSummary: {len(significant_results)} out of {len(res_corrected)} tests were significant.")




    # %
    # ================================================
    # === STEP 7 ===: Plot significant ROIs on surface
    # ================================================ 
    RSA_DIR = FIG_DIR / 'rsa'
    RSA_DIR.mkdir(exist_ok=True, parents=True)

    # Initialize the nested dictionary structure
    sig_rois = {
        "left": {},
        "right": {}
    }
    # Fill the dictionary
    for _, row in significant_results.iterrows():
        hemi = row["Hemisphere"].lower()
        roi_name = row["ROI"]
        fdrP_val = row[f"q_{factor}_fdr"]
        
        # Assign the beta value to the specific ROI under the correct hemisphere
        sig_rois[hemi][roi_name] = fdrP_val

    # MNI atlas
    atlas      = HO_ATLAS_MNI6.maps
    atlas_data = atlas.get_fdata()

    # fsaverage surface
    fsaverage = datasets.fetch_surf_fsaverage()

    hemis = ['left', 'right']
    for hemi in hemis:

        # get the surface for each hemi
        if hemi == 'left':
            mesh       = surface.load_surf_mesh(fsaverage.pial_left)
            white_mesh = surface.load_surf_mesh(fsaverage.white_left)
            sulc       = fsaverage.sulc_left
        elif hemi == 'right':
            mesh       = surface.load_surf_mesh(fsaverage.pial_right)
            white_mesh = surface.load_surf_mesh(fsaverage.white_right)
            sulc       = fsaverage.sulc_right

        rois = sig_rois[hemi]

        # empty surface
        master_beta_surf = np.zeros(len(mesh[0]))

        for roi_name, beta in rois.items():

            # Make sure 'name' matches exactly with the keys in your 'rois' dictionary
            idx_row        = atlas_labels.loc[atlas_labels['name'] == roi_name, 'index'].values
            roi_mask       = np.isin(atlas_data, idx_row)
            roi_atlas_data = np.where(roi_mask, atlas_data, 0)
            # Convert back to a Nifti image for plotting
            roi_img        = nib.Nifti1Image(roi_atlas_data, atlas.affine)

            
            roi_map_data = surface.vol_to_surf(
                roi_img,
                surf_mesh=mesh,
                inner_mesh=white_mesh,
                interpolation='linear',
                n_samples=1   # increase sampling density
            )
            master_beta_surf[roi_map_data > 0] = beta

        master_beta_surf[master_beta_surf == 0] = np.nan   # transparent outside mask

        fig = plt.figure(figsize=(12, 10))
        ax  = fig.add_subplot(111, projection="3d")

        for view in ["lateral", "medial", "ventral"]:

            # Create a colormap
            original_hot = plt.get_cmap('hot_r')
            new_colors   = original_hot(np.linspace(0, 0.7, 256))
            custom_hot   = mcolors.LinearSegmentedColormap.from_list('clipped_hot', new_colors)

            # 3. Use plot_surf_roi for hard edges
            display = plotting.plot_surf_roi(
                surf_mesh=mesh, 
                roi_map=master_beta_surf,  # Values are now 0, 1.9, 2.5, 2.7
                hemi=hemi, 
                view=view,
                bg_map=sulc, 
                cmap=custom_hot, 
                vmin=0,                 # Set vmin to just below your lowest beta
                vmax=0.05,
                alpha=1,
                colorbar=False,
                darkness=None,
                axes=ax
            )
            # 2. Create a new axis at the bottom for the colorbar
            # [left, bottom, width, height] as fractions of the figure
            cbar_ax = fig.add_axes([0.3, 0.15, 0.4, 0.03]) 

            # 3. Add the colorbar manually using a ScalarMappable
            norm = mpl.colors.Normalize(vmin=0, vmax=0.05)
            sm   = mpl.cm.ScalarMappable(cmap=custom_hot, norm=norm)
            
            cbar = fig.colorbar(
                sm, 
                cax=cbar_ax, 
                orientation='horizontal', 
                format='%.2f'
            )

            # 4. Add the label and clean up
            cbar.set_label('p-values (FDR-corrected)', fontsize=14, labelpad=10)
            cbar.ax.tick_params(labelsize=12)

            # save first
            path = RSA_DIR / f"{condition}_{view}_{hemi}_{factor}_surface.png"
            fig.savefig(path, dpi=600, bbox_inches="tight")

            plt.show() 



    if factor == 'grade':
        # %
        # ========================================================================
        # === STEP 8 ===: Plot fisher transformed similarity metrics across grades
        # ========================================================================
        # specify colors 
        grade_colors = {
            1: "darkgoldenrod", 
            2: "darkcyan", 
            4: "firebrick"
        }

        # hemisphere loop
        for hemi in hemis:

            # extract subset data
            rois    = sig_rois[hemi]
            df_hemi = df_results[df_results['Hemisphere'] == hemi].copy()

            # significant ROIs loop
            for roi_name, beta in rois.items():

                # extract subset data
                df = df_hemi[df_hemi['ROI'] == roi_name]

                # plot violin, box, and scatter plot
                fig, ax = plt.subplots(figsize=(6, 4))
                pt.RainCloud(x="Grade", y="Fisher_Z", hue="Grade", data=df,
                            point_size=10,
                            rain_edgecolor='gray',
                            rain_alpha=0.5,
                            width_viol=0.3, 
                            width_box=0.1, 
                            cloud_alpha=0.6,
                            offset=0.15, 
                            palette=grade_colors,
                            pointplot=True,
                            linecolor="dimgrey",
                            ax=ax)
                
                # customize axis 
                ax.set_ylabel("Correlation (z-transformed)", fontsize=12)
                ax.set_xlabel("Grade", fontsize=12)
                y_min, y_max = df['Fisher_Z'].min(), df['Fisher_Z'].max()
                ax.set_ylim([y_min - 0.1, y_max + 0.1])
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
                
                # save the figure
                plt.tight_layout()
                clean_roi = str(roi_name).replace(" ", "_").replace("/", "-")
                path = RSA_DIR / f"{condition}_{hemi}_{clean_roi}_grade.png"
                fig.savefig(path, dpi=600, bbox_inches="tight")

                # show the plot
                plt.show()




    elif factor == 'acc':
        # %
        # ==========================================================================
        # === STEP 9 ===: Plot fisher transformed similarity metrics across accuracy
        # ==========================================================================
        # specify colors 
        grade_colors = {
            1: "darkgoldenrod", 
            2: "darkcyan", 
            4: "firebrick"
        }

        # hemisphere loop
        for hemi in hemis:

            # extract subset data
            rois = sig_rois[hemi]
            df_hemi = df_results[df_results['Hemisphere'] == hemi].copy()

            # significant ROIs loop
            for roi_name, beta in rois.items():

                # extract subset data
                df = df_hemi[df_hemi['ROI'] == roi_name]
                # include the accuracy
                df.loc[:, 'mean_acc'] = task_df['mean_accuracy'].values

                # 1. Initialize the JointGrid
                g = sns.JointGrid(data=df, x="mean_acc", y="Fisher_Z", height=6, ratio=5)

                # 2. Plot the regression line in the center (total group trend)
                sns.regplot(data=df, x="mean_acc", y="Fisher_Z", ax=g.ax_joint, 
                            scatter=False, color="dimgrey", line_kws={"linewidth": 2})

                # 3. Plot the scatter points in the center (colored by Grade)
                sns.scatterplot(data=df, x="mean_acc", y="Fisher_Z", hue="Grade", 
                                palette=grade_colors, ax=g.ax_joint, 
                                s=80, alpha=0.6, edgecolor="black", legend=False)

                # 4. Add the marginal distributions (density plots) on the top and right
                sns.kdeplot(data=df, x="mean_acc", hue="Grade", palette=grade_colors, 
                            ax=g.ax_marg_x, fill=True, legend=False)
                sns.kdeplot(data=df, y="Fisher_Z", hue="Grade", palette=grade_colors, 
                            ax=g.ax_marg_y, fill=True, legend=False)

                # 5. Clean up labels and limits
                g.ax_joint.set_xlabel("Mean accuracy (%)", fontsize=12)
                g.ax_joint.set_ylabel("Correlation (z-transformed)", fontsize=12)
                y_min, y_max = df['Fisher_Z'].min(), df['Fisher_Z'].max()
                g.ax_joint.set_xlim(10, 90)
                g.ax_joint.set_ylim(y_min - 0.1, y_max + 0.1)

                # 6. Save (JointGrid uses g.savefig)
                clean_roi = str(roi_name).replace(" ", "_").replace("/", "-")
                path = RSA_DIR / f"{condition}_{hemi}_{clean_roi}_acc.png"
                g.savefig(path, dpi=600, bbox_inches="tight")        

                # show the plot
                plt.show()



        
# %%
