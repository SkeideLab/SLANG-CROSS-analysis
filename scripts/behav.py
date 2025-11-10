# %%
# ===  Load modules ===
from pathlib import Path
from bids import BIDSLayout
import pandas as pd
from scipy.stats import norm
import numpy as np
import zipfile
import re
import tempfile
import shutil

# %%
# ===  Parameters ===
BIDS_DIR     = Path('/ptmp/kazma/SLANG-CROSS-conversion')
FMRIPRE_DIR  = BIDS_DIR / 'derivatives/fmriprep'
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
DERIV_DIR    = ANALY_DIR / 'derivatives'
SESSION      = '01'
TASK         = 'language'
CRITERIA     = 'all' # all, excluded
EXC_SUBJECTS = ['108', '111', '113', '116', '118', '120', '121', '122', '124', '125', '126', '128', '201', '205', '206', '208', '220', '225', '226', '227', '405', '406', '408', '409', '410', '421', '422', '424', '427', '430', '434']

# %%
if CRITERIA == 'excluded':
    # === Get the BIDS layout
    layout            = BIDSLayout(BIDS_DIR, derivatives=FMRIPRE_DIR, validate=True, reset_database=True)
    subjects          = layout.get_subjects()  # returns a list like ['01', '02', '03', ...]
    print(f"Using {CRITERIA} subject")
    subjects_filtered = [s for s in subjects if s not in EXC_SUBJECTS]

elif CRITERIA == 'all':
    print(f"Using {CRITERIA} subject")
    path = BIDS_DIR / 'sourcedata/01'
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
# Accuracy and RT
# === read event file for each run ===
for n, subject in enumerate(subjects_filtered):

    # Event files
    if CRITERIA == 'excluded':
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


    # ALL: empty list
    acc_ls_all       = []
    RT_ls_all        = []
    f_name_ls_all    = []
    dprime_ls_all    = []
    c_ls_all         = []
    beta_ls_all      = []

    # VISUAL: empty list
    acc_ls_visual    = []
    RT_ls_visual     = []
    f_name_ls_visual = []
    dprime_ls_visual = []
    c_ls_visual      = []
    beta_ls_visual   = []

   # AUDIO: empty list
    acc_ls_audio     = []
    RT_ls_audio      = []
    f_name_ls_audio  = []
    dprime_ls_audio  = []
    c_ls_audio       = []
    beta_ls_audio    = []


    for i in range(n_event):
        file = events_files[i]
        if CRITERIA == 'excluded':
            f_name = file.filename
        elif CRITERIA == 'all':
            f_name = file.name
            f_name = re.sub(r"_date-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", f_name)
        df = pd.read_csv(file, sep='\t')

        # ===  ALL: Accuracy ===
        responses  = df['trial_type'].isin(['response_c', 'response_d']).sum()
        # here counts the number of tasks as total
        total  = df['trial_type'].isin(['images_words', 'images_pseudo', 'audios_words', 'audios_pseudo']).sum()
        corre  = (df['correct'] == True).sum()
        if responses > total/2:
            acc  = (corre / total) * 100
            acc  = int(acc)

            # === ALL: SDT ===
            df_filtered = df[~df['trial_type'].isin(['feedback_correct', 'feedback_incorrect'])]

            # Create a boolean mask for previous trial starting with 'images'
            mask_prev_words       = df_filtered['trial_type'].shift(1).str.endswith('words')
            mask_prev_pseudowords = df_filtered['trial_type'].shift(1).str.endswith('pseudo')

            # Create a boolean mask for response_c or response_d
            mask_correct   = df_filtered['correct']==True
            mask_incorrect = df_filtered['correct']==False

            # create a mask for each 
            mask_hit = mask_correct & mask_prev_words
            mask_miss = mask_incorrect & mask_prev_words
            mask_false_alarm = mask_incorrect & mask_prev_pseudowords
            mask_correct_rejection = mask_correct & mask_prev_pseudowords

            # value for each parameter
            hit               = mask_hit.sum()
            miss              = mask_miss.sum()
            false_alarm       = mask_false_alarm.sum()
            correct_rejection = mask_correct_rejection.sum()

            # Adjusted hit and false alarm rates
            hit_rate = (hit + 0.5) / (hit + miss + 1)
            fa_rate  = (false_alarm + 0.5) / (false_alarm + correct_rejection + 1)

            # Compute z-scores
            z_hit = norm.ppf(hit_rate)
            z_fa = norm.ppf(fa_rate)

            # Compute SDT metrics
            d_prime = z_hit - z_fa
            c = -0.5 * (z_hit + z_fa)
            beta = np.exp((z_fa**2 - z_hit**2) / 2)

        else:
            acc     = -1
            d_prime = -1
            c       = -1
            beta    = -1

        # === ALL: RT ===
        avg_rt = round(df.loc[df['correct'] == True, 'rt'].mean(), 2)
        
        # === ALL: Accuracy, RT, SDT ===
        acc_ls_all.append(acc)
        f_name_ls_all.append(f_name)
        RT_ls_all.append(avg_rt)
        dprime_ls_all.append(d_prime)
        c_ls_all.append(c)
        beta_ls_all.append(beta)



        # === VISUAL: Accuracy ===
        # Filter out unwanted rows
        df_filtered = df[~df['trial_type'].isin(['feedback_correct', 'feedback_incorrect'])]

        # Create a boolean mask for response_c or response_d
        mask_response = df_filtered['trial_type'].isin(['response_c', 'response_d'])

        # Create a boolean mask for previous trial starting with 'images'
        mask_prev_images = df_filtered['trial_type'].shift(1).str.startswith('images')

        # Combine masks: only keep rows where both conditions are True
        mask = mask_response & mask_prev_images
        df_filtfilt = df_filtered[mask]
        responses  = df_filtfilt['trial_type'].isin(['response_c', 'response_d']).sum()
        total  = df['trial_type'].isin(['images_words', 'images_pseudo']).sum()
        corre  = (df_filtfilt['correct'] == True).sum()
        if responses > total/2:
            acc  = (corre / total) * 100
            acc  = int(acc)

            # === VISUAL: SDT ===
            # Create a boolean mask for previous trial starting with 'images'
            mask_prev_words       = df_filtered['trial_type'].shift(1).str.endswith('images_words')
            mask_prev_pseudowords = df_filtered['trial_type'].shift(1).str.endswith('images_pseudo')

            # Create a boolean mask for response_c or response_d
            mask_correct   = df_filtfilt['correct']==True
            mask_incorrect = df_filtfilt['correct']==False

            # create a mask for each 
            mask_hit = mask_correct & mask_prev_words
            mask_miss = mask_incorrect & mask_prev_words
            mask_false_alarm = mask_incorrect & mask_prev_pseudowords
            mask_correct_rejection = mask_correct & mask_prev_pseudowords

            # value for each parameter
            hit               = mask_hit.sum()
            miss              = mask_miss.sum()
            false_alarm       = mask_false_alarm.sum()
            correct_rejection = mask_correct_rejection.sum()

            # Adjusted hit and false alarm rates
            hit_rate = (hit + 0.5) / (hit + miss + 1)
            fa_rate  = (false_alarm + 0.5) / (false_alarm + correct_rejection + 1)

            # Compute z-scores
            z_hit = norm.ppf(hit_rate)
            z_fa = norm.ppf(fa_rate)

            # Compute SDT metrics
            d_prime = z_hit - z_fa
            c = -0.5 * (z_hit + z_fa)
            beta = np.exp((z_fa**2 - z_hit**2) / 2)
        else:
            acc     = -1
            d_prime = -1
            c       = -1
            beta    = -1

        # === VISUAL: RT ===
        avg_rt = round(df_filtfilt.loc[df_filtfilt['correct'] == True, 'rt'].mean(), 2)
        
        # === VISUAL: Accuracy, RT, SDT ===
        acc_ls_visual.append(acc)
        RT_ls_visual.append(avg_rt)
        dprime_ls_visual.append(d_prime)
        c_ls_visual.append(c)
        beta_ls_visual.append(beta)



        # === AUDIO: Accuracy ===
        # Create a boolean mask for previous trial starting with 'audios'
        mask_prev_images = df_filtered['trial_type'].shift(1).str.startswith('audios')

        # Combine masks: only keep rows where both conditions are True
        mask        = mask_response & mask_prev_images
        df_filtfilt = df_filtered[mask]
        responses   = df_filtfilt['trial_type'].isin(['response_c', 'response_d']).sum()
        total       = df['trial_type'].isin(['audios_words', 'audios_pseudo']).sum()
        corre       = (df_filtfilt['correct'] == True).sum()

        if responses > total/2:
            acc  = (corre / total) * 100
            acc  = int(acc)

            # === AUDIO: SDT ===

            # Create a boolean mask for previous trial starting with 'images'
            mask_prev_words       = df_filtered['trial_type'].shift(1).str.endswith('audios_words')
            mask_prev_pseudowords = df_filtered['trial_type'].shift(1).str.endswith('audios_pseudo')

            # Create a boolean mask for response_c or response_d
            mask_correct   = df_filtered['correct']==True
            mask_incorrect = df_filtered['correct']==False

            # create a mask for each 
            mask_hit               = mask_correct & mask_prev_words
            mask_miss              = mask_incorrect & mask_prev_words
            mask_false_alarm       = mask_incorrect & mask_prev_pseudowords
            mask_correct_rejection = mask_correct & mask_prev_pseudowords

            # value for each parameter
            hit               = mask_hit.sum()
            miss              = mask_miss.sum()
            false_alarm       = mask_false_alarm.sum()
            correct_rejection = mask_correct_rejection.sum()

            # Adjusted hit and false alarm rates
            hit_rate = (hit + 0.5) / (hit + miss + 1)
            fa_rate  = (false_alarm + 0.5) / (false_alarm + correct_rejection + 1)

            # Compute z-scores
            z_hit   = norm.ppf(hit_rate)
            z_fa    = norm.ppf(fa_rate)

            # Compute SDT metrics
            d_prime = z_hit - z_fa
            c       = -0.5 * (z_hit + z_fa)
            beta    = np.exp((z_fa**2 - z_hit**2) / 2)

        else:
            acc     = -1
            d_prime = -1
            c       = -1
            beta    = -1

        # === AUDIO: RT ===
        avg_rt = round(df_filtfilt.loc[df_filtfilt['correct'] == True, 'rt'].mean(), 2)
        
        # === AUDIO: Accuracy, RT, SDT ===
        acc_ls_audio.append(acc)
        RT_ls_audio.append(avg_rt)
        dprime_ls_audio.append(d_prime)
        c_ls_audio.append(c)
        beta_ls_audio.append(beta)

    # === output as csv file ===
    df_acc = pd.DataFrame({
        'file_name':       f_name_ls_all,
        'accuracy_all':    acc_ls_all,
        'RT_all':          RT_ls_all,
        'dprime_all':      dprime_ls_all,
        'c_all':           c_ls_all,
        'beta_all':        beta_ls_all,
        'accuracy_visual': acc_ls_visual,
        'RT_visual':       RT_ls_visual,
        'dprime_visual':   dprime_ls_visual,
        'c_visual':        c_ls_visual,
        'beta_visual':     beta_ls_visual,
        'accuracy_audio':  acc_ls_audio,
        'RT_audio':        RT_ls_audio,
        'dprime_audio':    dprime_ls_audio,
        'c_audio':         c_ls_audio,
        'beta_audio':      beta_ls_audio
    })

    save_path = DERIV_DIR / f'sub-{subject}/behavior/{CRITERIA}/accuracy_summary.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save to CSV
    df_acc.to_csv(save_path, index=False)
    print(f"Saved the csv to {save_path}")

    # Delete temp folder after finishing this subject
    # temp_dir is a Path object
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"✅ Deleted temporary folder: {temp_dir}")
    else:
        print(f"ℹ️ Temporary folder does not exist: {temp_dir}")
# %%
