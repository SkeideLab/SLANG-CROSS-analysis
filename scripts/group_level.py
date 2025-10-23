# %% [markdown]
# ## fMRI Group-level analysis (2nd-level)
#
# Group-level analysis using Nilearn
# with 4 conditions: visual word, visual pseudoword, audio word, audio pseudoword.

# %%
# ===  Load modules ===
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map
import pandas as pd
# %%
# covariates
covars = pd.DataFrame({'age': [25, 30, 28], 
                       'sex': [0, 1, 0], 
                       'accuracy': [0, 1, 0], 
                       'RT': [0, 1, 0], 
                       'd': [],
                       'drift rate':[],})

# subject-level contrast maps
inputs = ['sub-01_contrast.nii.gz', 'sub-02_contrast.nii.gz', 'sub-03_contrast.nii.gz']

# Create the model
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(
    second_level_input=inputs,
    design_matrix=covars
)

# Compute group-level stats
z_map = second_level_model.compute_contrast(output_type='z_score')

# Visualize
plot_stat_map(z_map, title='Group-level activation')