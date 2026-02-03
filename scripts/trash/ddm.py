# %%
# ===  Load modules ===
from pathlib import Path
from bids import BIDSLayout
import pandas as pd
from scipy.stats import norm
import numpy as np
import pyddm
import zipfile
import re
import shutil
import pyddm.plot
from pyddm import Sample
import matplotlib.pyplot as plt

# ===  Parameters ===
BIDS_DIR     = Path('/ptmp/kazma/SLANG-CROSS-conversion')
FMRIPRE_DIR  = BIDS_DIR / 'derivatives/fmriprep'
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
OUT_DIR      = ANALY_DIR / 'outputs' / 'behavior'
SESSION      = '01'
TASK         = 'language'
CRITERIA     = 'excluded' # all, excluded
MODALITY     = 'visual' # all, audio, visual
EXC_SUBJECTS = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422','423','424', '427', '430', '434']

# make a directory
OUT_DIR      = ANALY_DIR / 'outputs' / 'behavior' / CRITERIA
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
if CRITERIA == 'excluded':
    # === Get the BIDS layout
    layout            = BIDSLayout(BIDS_DIR, derivatives=FMRIPRE_DIR, validate=True, reset_database=True)
    subjects          = layout.get_subjects()  # returns a list like ['01', '02', '03', ...]
    subjects_filtered = [s for s in subjects if s not in EXC_SUBJECTS]

elif CRITERIA == 'all':
    print(f"Using {CRITERIA} subject")
    path = BIDS_DIR / 'sourcedata' / '01'
    # List all log files (you can adjust extensions as needed)
    log_files = [f for f in path.glob("*_log.zip")]

    # Extract subject numbers from each filename
    subjects_filtered = []
    def get_subject_number(fpath):
        match = re.match(r"(\d+)_", fpath.name)
        return int(match.group(1)) if match else float("inf")

    log_files_sorted = sorted(log_files, key=get_subject_number)

    # Print ordered subjects
    for f in log_files_sorted:
        subject_num = get_subject_number(f)
        subjects_filtered.append(subject_num)

# %%
sub_vec = []
v_vec   = []
a_vec   = []
z_vec   = []
t_vec   = []

for n, subject in enumerate(subjects_filtered):

    if CRITERIA == 'excluded':
        # Event files
        events_files = layout.get(subject=subject, session=SESSION, task=TASK,
                                    suffix='events', extension='tsv')
        n_event      = len(events_files)

    elif CRITERIA == 'all':
        # Create temporary folder for this subject
        temp_dir = ANALY_DIR / f"tmp_{subject}"
        temp_dir.mkdir(exist_ok=True)

        path = log_files_sorted[n]

        # Extract contents
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(temp_dir)

        # Now you can access files in temp_path
        events_files = list(temp_dir.rglob('*events.tsv'))
        # sort by the integer after "run-"
        events_files = sorted(events_files, key=lambda p: int(p.stem.split('run-')[1].split('_')[0]))
        n_event      = len(events_files)

    xx=[]
    for i in range(n_event):
        file   = events_files[i]
        if CRITERIA == 'excluded':
            f_name = file.filename
        elif CRITERIA == 'all':
            f_name = file.name
            f_name = re.sub(r"_date-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", f_name)
      
        df_ddm = pd.read_csv(file, sep='\t')

        if   MODALITY=='all':
            responses  = df_ddm['trial_type'].isin(['response_c', 'response_d']).sum()
            total      = df_ddm['trial_type'].isin(['images_words', 'images_pseudo', 'audios_words', 'audios_pseudo']).sum()
        elif MODALITY=='visual':
            df_ddm     = df_ddm[~df_ddm['trial_type'].isin(['feedback_correct', 'feedback_incorrect'])]
            mask_1     = df_ddm['trial_type'].shift(1).str.startswith('images')
            mask_2     = df_ddm['trial_type'].isin(['response_c', 'response_d'])
            masks      = mask_1 & mask_2
            df_ddm_ex  = df_ddm[masks]
            responses  = len(df_ddm_ex)
            total      = df_ddm['trial_type'].isin(['images_words', 'images_pseudo']).sum()
        elif MODALITY=='audio':
            df_ddm     = df_ddm[~df_ddm['trial_type'].isin(['feedback_correct', 'feedback_incorrect'])]
            mask_1     = df_ddm['trial_type'].shift(1).str.startswith('audios')
            mask_2     = df_ddm['trial_type'].isin(['response_c', 'response_d'])
            masks      = mask_1 & mask_2
            df_ddm_ex  = df_ddm[masks]
            responses  = len(df_ddm_ex)
            total      = df_ddm['trial_type'].isin(['audios_words', 'audios_pseudo']).sum()
        if responses > total/2:
            x            = df_ddm[df_ddm['trial_type'].isin(['response_c', 'response_d'])]
            x            = x.drop(columns=['onset', 'duration', 'stim_file'])
            x            = x.dropna(subset=['rt'])
            x['correct'] = x['correct'].replace({True: 1, False: 0})
            x['correct'] = x['correct'].replace({'TRUE': 1, 'FALSE': 0})
            x            = x.reset_index(drop=True)
            xx.append(x)

    if len(xx) > 0:
        xx     = pd.concat(xx, ignore_index=True)
        T_dur  = xx['rt'].max() + 1.0
        sample = Sample.from_pandas_dataframe(xx, rt_column_name="rt", choice_column_name="correct")

        parameters = {"d": (-4,4), 
                    "B": (0.3, 2), 
                    "x0": (-.8, .8),
                    "ndt": (0.1, 0.5)}
        model = pyddm.gddm(drift="d", noise=1.0, bound="B", nondecision="ndt", starting_position="x0",
                                parameters=parameters, T_dur=T_dur)
        model.fit(sample, lossfunction=pyddm.LossBIC, verbose=False)
        # model.show()
        print("Parameters:", model.parameters())
        v = model.parameters()['drift']['drift']
        v = float(v)
        a = model.parameters()['bound']['B']
        a = float(a)
        z = model.parameters()['IC']['x0']
        z = float(z)
        t = model.parameters()['overlay']['nondectime']
        t = float(t)

        sub_vec.append(subject)
        v_vec.append(v)
        a_vec.append(a)
        z_vec.append(z)
        t_vec.append(t)

    else:
        print("\nNo data available")
        print(subject)

    # Delete temp folder after finishing this subject
    # temp_dir is a Path object
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"✅ Deleted temporary folder: {temp_dir}")
    else:
        print(f"ℹ️ Temporary folder does not exist: {temp_dir}")

# Create a DataFrame
df = pd.DataFrame({
    'subject': sub_vec,
    'drift': v_vec,
    'bound': a_vec,
    'start': z_vec,
    'nondecision': t_vec
})

# Output the data ot csv 
OUT_DIR.mkdir(parents=True, exist_ok=True)
f_name = f"{OUT_DIR}/ddm_{MODALITY}.csv"
df.to_csv(f_name, index=False)



# %%
