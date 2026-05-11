# SLANG-CROSS
Analysis pipeline for the SLANG-CROSS research project.

---

## Project Description

This repository contains the workflow used in the SLANG-CROSS project, including:

- Behavior data analysis
- Regions of interest (ROIs)
- Subject-level fMRI: general linear model (GLM)
- Group-level fMRI: representational dissimilarity matrix (RDM)
- Group-level fMRI: one-sample t-test and linear regression
- Figure generation

---

## Repository Layout

```text
.
├── figures
│   ├── behavior           # results from behavioral analsyis
│   └── multimodal         # results from multimodal analsyis
│
├── scripts
│   ├── behavior           # scripts for behavioral analsyis
│   ├── gro-level-fMRI     # scripts for group-level analsyis
│   ├── sub-level-fMRI     # scripts for subject-level analsyis
│   ├── helpers.py         # set of functions
│   ├── my_packages.py     # set of libraries
│   └── roi_create.py      # script for ROIs
│
├── .gitignore             # ignore preprocessed fMRI
├── README.md              
├── run_params.json        # setting
└── sub-level-fMRI_run.py  # slurm job for GLM