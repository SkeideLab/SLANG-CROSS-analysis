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

SESSION      = '01'
TASK         = 'language'
EXC_SUBJECTS = ['108', '113', '116', '122', '125', '126', '206', '220', '227', '405', '410', '421','424', '427', '430']

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
    acc_ls       = []
    f_name_ls    = []

    for i in range(n_event):
        file   = events_files[i]
        f_name = file.filename
        df     = pd.read_csv(file, sep='\t')

        # ===  Get the accuracy ===
        total  = df['trial_type'].isin(['images_words', 'images_pseudo', 'audios_words', 'audios_pseudo']).sum()
        corre  = (df['correct'] == True).sum()
        acc    = (corre / total) * 100
        acc    = int(acc)

        # store accuracy in a list
        acc_ls.append(acc)
        f_name_ls.append(f_name)

    # === output as csv file ===
    df_acc = pd.DataFrame({
        'file_name': f_name_ls,
        'accuracy': acc_ls
    })

    save_path = DERIV_DIR / f'sub-{subject}/behavior/accuracy_summary.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save to CSV
    df_acc.to_csv(save_path, index=False)
    print(f"Saved the csv to {save_path}")


# %%
