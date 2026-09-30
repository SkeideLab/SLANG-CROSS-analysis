# %% [markdown]
# ## Group-level Behavioral
#
# **Pipeline Overview**
# 1. === STEP 1 ===: Install packages
# 2. === STEP 2 ===: Set parameters




# %%
# 1. === STEP 1 ===: Install packages
# -----------------------------------------------
# install necessary packages
import sys
from pathlib import Path
import statsmodels.api as sm
# Specify path
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
BIDS_DIR     = Path('/work_beegfs/suknp132/SLANG-CROSS-conversion')
DERIV_DIR    = ANALY_DIR / 'derivatives'
SCRIP_DIR    = ANALY_DIR / 'scripts'
FIG_DIR      = ANALY_DIR / 'figures'
OUT_DIR      = ANALY_DIR / 'outputs'
DEMO_DIR     = ANALY_DIR / 'demographics'
TEMP_DIR     = ANALY_DIR / 'templates'
MASK_DIR     = TEMP_DIR  / 'mask'
# install paclages in a python file
sys.path.append(str(SCRIP_DIR))
import my_packages
from my_packages import *



# %%
# 2. === STEP 2 ===: Prepare datasets
# -----------------------------------------------
EXC_SUBJECTS   = [
                '108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', 
                '201', '205', '206', '208', '220', '225', '226', '227', 
                '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434',                ]

# Subjects
subjects      = sorted(DERIV_DIR.glob(f"sub-*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]
subject_names = [p.name.replace('sub-', '') for p in subjects]


# %%
# 3. === STEP 3 ===: Check demographics
# -----------------------------------------------
demo_path = BIDS_DIR / "participants.tsv" 
demo_df = pd.read_csv(demo_path, sep="\t")
demo_df = demo_df[
    demo_df["participant_id"]
    .str.replace("sub-", "", regex=False)
    .isin(subject_names)
].copy()
demo_df["grade"] = (
    demo_df["participant_id"]
    .str.replace("sub-", "", regex=False)
    .str[0]
    .astype(int)
)

# Mean and SD of age by grade
age_summary = (
    demo_df["age"]
    .agg(["mean", "std"])
)

print("Overall age:")
print("Mean:", age_summary["mean"])
print("SD:", age_summary["std"])

age_summary = (
    demo_df.groupby("grade")["age"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={
        "mean": "age_mean",
        "std": "age_sd"
    })
)

print("Age by grade:")
print(age_summary)

# Sex count by grade
sex_summary = (
    demo_df["sex"]
    .value_counts()
    .sort_index()
)

print("\nOverall sex count:")
print(sex_summary)

sex_summary = (
    demo_df
    .groupby(["grade", "sex"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

print("\nSex count by grade:")
print(sex_summary)

# %%
# 3. === STEP 3 ===: Check fMRI characteristics
# -----------------------------------------------
# mriqc_path
mriqc_path = BIDS_DIR / "derivatives" / "mriqc"

# Store each subject's dataframe
df_list = []

for subject in subjects:

    path = subject / "behavior" / "accuracy_summary.csv"

    if not path.exists():
        print(f"Missing: {path}")
        continue

    df = pd.read_csv(path)

    # Remove rows with invalid accuracy
    df = df[df["accuracy_all"] != -1].copy()

    # Extract grade:
    df["grade"] = (
        df["file_name"]
        .str.extract(r"sub-(\d)")[0]
        .astype(int)
    )

    # Extract participant ID:
    df["subject"] = (
        df["file_name"]
        .str.extract(r"sub-(\d+)")[0]
        .astype(int)
    )

    f_name = df["file_name"].str.replace("_events.tsv", "", regex=False)
    participant_id = subject.name
    fd_path = mriqc_path / participant_id / "ses-01" / participant_id / "ses-01" / "func"
    fd_means = []

    for name in f_name:
        file = fd_path / f"{name}_timeseries.tsv"

        # Read the TSV
        confounds = pd.read_csv(file, sep="\t")

        # Mean framewise displacement
        mean_fd = confounds["framewise_displacement"].mean()

        fd_means.append(mean_fd)

    # Add participant/run-level FD to your dataframe
    df["fd"] = fd_means


    df_list.append(df)


# Concatenate all subjects
df_all = pd.concat(df_list, ignore_index=True)
print(df_all)
print(f"Total rows: {len(df_all)}")



# %%
# 3. === STEP 3 ===: Check task engagement
# -----------------------------------------------
# Mean and SD of responses_all across all subjects
responses_all_summary = (
    df_all["responses_all"]
    .agg(["mean", "std"])
)

print("Overall mean:", responses_all_summary["mean"])
print("Overall SD:", responses_all_summary["std"])

#  Mean and SD of responses_all for each grade
responses_summary = (
    df_all.groupby("grade")["responses_all"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={
        "mean": "responses_mean",
        "std": "responses_sd",
    })
)
print(responses_summary)

# apply linear regression model
X = df_all["grade"]
y = df_all["responses_all"]
# Add intercept
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# %%
# 3. === STEP 3 ===: Check accuracy
# -----------------------------------------------
# Mean and SD of responses_all across all subjects
acc_all_summary = (
    df_all["accuracy_all"]
    .agg(["mean", "std"])
)

print("Overall mean:", acc_all_summary["mean"])
print("Overall SD:", acc_all_summary["std"])

#  Mean and SD of responses_all for each grade
acc_summary = (
    df_all.groupby("grade")["accuracy_all"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={
        "mean": "acc_mean",
        "std": "acc_sd",
    })
)
print(acc_summary)

df_acc_reg = df_all[["grade", "accuracy_all"]].copy()

# Remove NaN and infinite values
df_acc_reg = (
    df_acc_reg
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

# apply linear regression model
X = df_acc_reg["grade"]
y = df_acc_reg["accuracy_all"]
# Add intercept
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())



# %%
# 3. === STEP 3 ===: Check reaction time
# -----------------------------------------------
# Mean and SD of responses_all across all subjects
RT_all_summary = (
    df_all["RT_all"]
    .agg(["mean", "std"])
)

print("Overall mean:", RT_all_summary["mean"])
print("Overall SD:", RT_all_summary["std"])

#  Mean and SD of responses_all for each grade
RT_summary = (
    df_all.groupby("grade")["RT_all"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={
        "mean": "RT_mean",
        "std": "RT_sd",
    })
)
print(RT_summary)

df_RT_reg = df_all[["grade", "RT_all"]].copy()

# Remove NaN and infinite values
df_RT_reg = (
    df_RT_reg
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

# apply linear regression model
X = df_RT_reg["grade"]
y = df_RT_reg["RT_all"]
# Add intercept
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())



# %%
# 3. === STEP 3 ===: Check head motion
# -----------------------------------------------
# Mean and SD of responses_all across all subjects
fd_all_summary = (
    df_all["fd"]
    .agg(["mean", "std"])
)
print("Overall mean:", fd_all_summary["mean"])
print("Overall SD:", fd_all_summary["std"])
# Mean and SD of responses_all for each grade
fd_summary = (
    df_all.groupby("grade")["fd"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={
        "mean": "fd_mean",
        "std": "fd_sd",
    })
)
print(fd_summary)

# apply linear regression model
X = df_all["grade"]
y = df_all["fd"]
# Add intercept
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# %%
# 3. === STEP 3 ===: Check runs available
# -----------------------------------------------
runs_per_subject = (
    df_all
    .groupby(["subject", "grade"])
    .size()
    .reset_index(name="n_runs")
)

print("Number of runs per subject:")
print(runs_per_subject)

runs_by_all = (
    runs_per_subject["n_runs"]
    .agg(["count", "mean", "std", "min", "max"])
    .reset_index()
)

print("\nNumber of runs all grades:")
print(runs_by_all)

runs_by_grade = (
    runs_per_subject
    .groupby("grade")["n_runs"]
    .agg(["count", "mean", "std", "min", "max"])
    .reset_index()
)

print("\nNumber of runs by grade:")
print(runs_by_grade)
X = sm.add_constant(runs_per_subject["grade"])
y = runs_per_subject["n_runs"]

model = sm.OLS(y, X).fit()

print("\nLinear regression: n_runs ~ grade")
print(model.summary())
# %%
