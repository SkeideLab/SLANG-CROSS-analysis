# %% Load modules
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import zipfile
import pydicom
from pathlib import Path
from collections import defaultdict

# %% Parameters
ANALY_DIR    = Path('/ptmp/kazma/SLANG-CROSS-analysis')
BIDS_DIR     = Path('/ptmp/kazma/SLANG-CROSS-conversion')
SOURCE_DIR   = BIDS_DIR / 'sourcedata' / '01'

# %% Example Figure 1, 2, 3
groups    = ['1', '2', '4']  # x-axis
visual_w  = [300, 350, 320]  # y-values in mm
spoken_w  = [500, 630, 1050]

# Plot lines
plt.plot(groups, visual_w, marker='o', label='Visual word')
plt.plot(groups, spoken_w, marker='s', label='Auditory word')

# Add labels and title
plt.xlabel('Grades')
plt.ylabel(r'Total volume (mm$^3$)')
plt.title('Left hemisphere (MTG)')
plt.ylim(0,2000)
plt.legend()
plt.grid(False)
plt.savefig('left.png', dpi=300, bbox_inches='tight')  # PNG format, high resolution
plt.show()

# %% Example Figure 4
# %% Example Figure 1, 2, 3
groups    = ['1', '2', '4']  # x-axis
corr      = [0.2, 0.25, 0.3] # y-values in r

# Plot lines
plt.plot(groups, corr, marker='o', color='purple')

# Add labels and title
plt.xlabel('Grades')
plt.ylabel('Correlation (r)')
plt.title('Right hemisphere (pSTG)')
plt.ylim(0,1)
plt.legend()
plt.grid(False)
plt.savefig('right_corr.png', dpi=300, bbox_inches='tight')  # PNG format, high resolution
plt.show()

# %% Remove a run from DICOM dataset
subject  = '205'
zip_path = list(SOURCE_DIR.glob(f"{subject}_*_mri.zip"))
zip      = zip_path[0]
temp_dir = ANALY_DIR / f"tmp_{subject}"
temp_dir.mkdir(exist_ok=True)
# Extract contents
with zipfile.ZipFile(zip, 'r') as zf:
    zf.extractall(temp_dir)

# check the number of scans in a run
# %%

dicom_keys = {
    "205": '1.2.840.113619.6.475.216053225509435518884072285396318249265',
    "423": '1.2.840.113619.6.475.224499192250843020009328387779392813090'
}

dicom_dir = ANALY_DIR / 'tmp_205' / 'New' / '1.2.840.113619.6.475.216053225509435518884072285396318249265'

series_info = {}
for dcm_path in dicom_dir.glob("*.dcm"):
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        uid = ds.SeriesInstanceUID
        desc = getattr(ds, "SeriesDescription", "").lower()
        protocol = getattr(ds, "ProtocolName", "").lower()

        if uid not in series_info:
            series_info[uid] = {
                "count": 0,
                "SeriesDescription": desc,
                "ProtocolName": protocol,
            }
        series_info[uid]["count"] += 1
        
    except Exception as e:
        print(f"Error reading {dcm_path.name}: {e}")



# %%
# Classify each series
for uid, info in series_info.items():
    desc = info["SeriesDescription"]
    protocol = info["ProtocolName"]

    if any(x in desc or x in protocol for x in ["bold", "fmri", "task", "rest"]):
        scan_type = "Functional (fMRI)"
    elif any(x in desc or x in protocol for x in ["t1", "mprage", "anat"]):
        scan_type = "Anatomical (T1w)"
    elif any(x in desc or x in protocol for x in ["fmap", "gre", "field map", "epse"]):
        scan_type = "Fieldmap (fmap)"
    elif any(x in desc or x in protocol for x in ["localizer"]):
        scan_type = "Localizer"
    else:
        scan_type = "Unknown"

    print(f"{uid}: {info['count']} files → {scan_type}")

# %%

target_uid = "1.2.840.113619.2.475.14196467.2133404.30205.1708829155.558"
new_description = "3D Ax T1 MPRAGE"
scanning_sequence = ['RM', 'IR']

for dcm_path in dicom_dir.glob("*.dcm"):
    try:
        ds = pydicom.dcmread(dcm_path)
        if ds.SeriesInstanceUID == target_uid:

            # ds.SeriesDescription = new_description
            # ds.save_as(dcm_path)
            print(f"Updated {dcm_path.name} -> SeriesDescription={new_description}")
    except Exception as e:
        print(f"Error reading {dcm_path.name}: {e}")
# %%
path = dicom_dir / '1.2.840.113619.2.475.14196467.2133404.24723.1708829200.917.dcm'
# Read the DICOM file
ds = pydicom.dcmread(path)

# Print all available metadata
print(ds)
# %%
