# %%
# ===  Load modules ===
from pathlib import Path
from bids import BIDSLayout
import pandas as pd

# %%
# ===  Parameters ===
BIDS_DIR     = Path('/ptmp/kazma/SLANG-CROSS-conversion')
FMRIPRE_DIR  = BIDS_DIR / 'derivatives/fmriprep'
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
MODALITY     = 'all' # visual, audio, or all
SESSION      = '01'
TASK         = 'language'
EXC_SUBJECTS = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '206', '208', '220', '227', '405', '406', '408', '410', '421', '422', '424', '427', '430']

# %%
# === Get the BIDS layout
layout            = BIDSLayout(BIDS_DIR, derivatives=FMRIPRE_DIR, validate=True, reset_database=True)
subjects          = layout.get_subjects()  # returns a list like ['01', '02', '03', ...]
subjects_filtered = [s for s in subjects if s not in EXC_SUBJECTS]

# %%
# === read event file for each run ===
for subject in subjects_filtered:

    # Event files
    events_files = layout.get(subject=subject, session=SESSION, task=TASK,
                                suffix='events', extension='tsv')
    n_event      = len(events_files)

    # empty list
    acc_ls_all       = []
    RT_ls_all        = []
    f_name_ls_all    = []

    # empty list for visual modality
    acc_ls_visual    = []
    RT_ls_visual     = []
    f_name_ls_visual = []

   # empty list for audio modality
    acc_ls_audio     = []
    RT_ls_audio      = []
    f_name_ls_audio  = []


    for i in range(n_event):
        file   = events_files[i]
        f_name = file.filename
        df     = pd.read_csv(file, sep='\t')

        # ===  Get the accuracy for all ===
        total  = df['trial_type'].isin(['response_c', 'response_d']).sum()
        corre  = (df['correct'] == True).sum()
        if total > 0:
            acc  = (corre / total) * 100
            acc  = int(acc)
        else:
            acc = 0

        # === Get the RT for all ===
        avg_rt = round(df.loc[df['correct'] == True, 'rt'].mean(), 2)
        
        # store accuracy and RT in a list
        acc_ls_all.append(acc)
        f_name_ls_all.append(f_name)
        RT_ls_all.append(avg_rt)

        # === Get the accuracy for visual ===
        # Filter out unwanted rows
        df_filtered = df[~df['trial_type'].isin(['feedback_correct', 'feedback_incorrect'])]

        # Create a boolean mask for response_c or response_d
        mask_response = df_filtered['trial_type'].isin(['response_c', 'response_d'])

        # Create a boolean mask for previous trial starting with 'images'
        mask_prev_images = df_filtered['trial_type'].shift(1).str.startswith('images')

        # Combine masks: only keep rows where both conditions are True
        mask = mask_response & mask_prev_images
        df_filtfilt = df_filtered[mask]
        total  = df_filtfilt['trial_type'].isin(['response_c', 'response_d']).sum()
        corre  = (df_filtfilt['correct'] == True).sum()
        if total > 0:
            acc  = (corre / total) * 100
            acc  = int(acc)
        else:
            acc = 0

        # === Get the RT for visual ===
        avg_rt = round(df_filtfilt.loc[df_filtfilt['correct'] == True, 'rt'].mean(), 2)
        
        # store accuracy and RT in a list
        acc_ls_visual.append(acc)
        RT_ls_visual.append(avg_rt)

        # === Get the accuracy for visual ===
        # Create a boolean mask for previous trial starting with 'images'
        mask_prev_images = df_filtered['trial_type'].shift(1).str.startswith('audios')

        # Combine masks: only keep rows where both conditions are True
        mask = mask_response & mask_prev_images
        df_filtfilt = df_filtered[mask]
        total  = df_filtfilt['trial_type'].isin(['response_c', 'response_d']).sum()
        corre  = (df_filtfilt['correct'] == True).sum()
        if total > 0:
            acc  = (corre / total) * 100
            acc  = int(acc)
        else:
            acc = 0

        # === Get the RT for visual ===
        avg_rt = round(df_filtfilt.loc[df_filtfilt['correct'] == True, 'rt'].mean(), 2)
        
        # store accuracy and RT in a list
        acc_ls_audio.append(acc)
        RT_ls_audio.append(avg_rt)


    # === output as csv file ===
    df_acc = pd.DataFrame({
        'file_name': f_name_ls_all,
        'accuracy_all': acc_ls_all,
        'RT_all': RT_ls_all,
        'accuracy_visual': acc_ls_visual,
        'RT_visual': RT_ls_visual,
        'accuracy_audio': acc_ls_audio,
        'RT_audio': RT_ls_audio
    })

    save_path = DERIV_DIR / f'sub-{subject}/behavior/accuracy_summary.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save to CSV
    df_acc.to_csv(save_path, index=False)
    print(f"Saved the csv to {save_path}")


# %%
