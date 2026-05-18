# %% [markdown]
# ## fMRI task performance: descriptive (2nd-level)
#
# **Pipeline Overview**
# 1. === STEP 1 ===: Install packages
# 2. === STEP 2 ===: Set parameters
# 3. === STEP 3 ===: read behavioral csv file
# 4. === STEP 4 ===: Global Mean and SD
# 5. === STEP 5 ===: Grade-wise Mean and SD

# 6. === STEP 6 ===: Statistics: One-sample t-tetst
# 7. === STEP 7 ===: Statistics: Linear regression



# %%
# 1. === STEP 1 ===: Install packages
# -----------------------------------------------

# install necessary packages
import sys
from pathlib import Path
# Specify path
ANALY_DIR    = Path('/work_beegfs/suknp132/SLANG-CROSS-analysis')
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
# 2. === STEP 2 ===: Set parameters
# -----------------------------------------------
EXC_SUBJECTS   = [
                '108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', 
                '201', '205', '206', '208', '220', '225', '226', '227', 
                '405', '406', '408', '409', '410', '421', '422', '423', '424', '427', '430', '434',
                ]



# %%
# 3. === STEP 3 ===: read behavioral csv file
# -----------------------------------------------

# Subjects
subjects      = sorted(DERIV_DIR.glob(f"sub-*"))
exclude       = [f"sub-{s}" for s in EXC_SUBJECTS]
subjects      = [s for s in subjects if s.name not in exclude]

# read RT
results       = []
for sub in subjects:
    task_path = sub / 'behavior' / 'accuracy_summary.csv'
    task_df   = pd.read_csv(task_path)
    mean_acc  = task_df['accuracy_all'].mean()
    mean_rt   = task_df['RT_all'].mean()
    # Store results as a dict
    results.append({
        'subject': sub.name,
        'grade': sub.name[4],
        'acc': mean_acc,
        'rt': mean_rt,

    })
task_df       = pd.DataFrame(results)



# %%
# 4. === STEP 4 ===: Global Mean and SD
# -----------------------------------------------
metric_list = ['acc', 'rt']

for metric in metric_list:
    mean_val = task_df[metric].mean()
    sd_val   = task_df[metric].std()

    print(f"{metric}: mean = {mean_val:.4f}, SD = {sd_val:.4f}")



# %%
# 5. === STEP 5 ===: Grade-wise Mean and SD

summary = task_df.groupby('grade')[['acc', 'rt']].agg(['mean', 'std'])
print(summary)

# %%


