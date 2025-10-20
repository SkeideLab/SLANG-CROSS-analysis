# %%
# ===  Load modules ===
from pathlib import Path
from bids import BIDSLayout
import pandas as pd
from scipy.stats import norm
import numpy as np
import pyddm
import pyddm.plot
from pyddm import Sample
import matplotlib.pyplot as plt

# ===  Parameters ===
BIDS_DIR     = Path('/ptmp/kazma/SLANG-CROSS-conversion')
FMRIPRE_DIR  = BIDS_DIR / 'derivatives/fmriprep'
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
OUT_DIR      = ANALY_DIR / 'outputs'
SESSION      = '01'
TASK         = 'language'
EXC_SUBJECTS = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422', '424', '427', '430', '434']

# %%
# === Get the BIDS layout
layout            = BIDSLayout(BIDS_DIR, derivatives=FMRIPRE_DIR, validate=True, reset_database=True)
subjects          = layout.get_subjects()  # returns a list like ['01', '02', '03', ...]
subjects_filtered = [s for s in subjects if s not in EXC_SUBJECTS]

# %%
sub_vec = []
v_vec   = []
a_vec   = []
z_vec   = []
t_vec   = []

for subject in subjects_filtered:

    # Event files
    events_files = layout.get(subject=subject, session=SESSION, task=TASK,
                                suffix='events', extension='tsv')
    n_event      = len(events_files)

    xx=[]
    for i in range(n_event):
        file   = events_files[i]
        f_name = file.filename
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


