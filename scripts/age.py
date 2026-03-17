# %%
import zipfile
import pydicom
import io
from pathlib import Path

# Define the directory path
base_path = Path('/work_beegfs/suknp132/SLANG-CROSS-conversion/sourcedata/01/')

# Use .glob() to find all files ending in _mri.zip
zip_files = list(base_path.glob('*_mri.zip'))
# Sort by extracting the prefix (e.g., '420') and converting to int
sorted_zip_files = sorted(zip_files, key=lambda p: int(p.name.split('_')[0]))
print(f"Found {len(zip_files)} MRI zip files. Starting extraction...\n")

for zip_path in sorted_zip_files:
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Filter for DICOM files
            dicom_files = [f for f in z.namelist() if f.lower().endswith('.dcm') or 'IM' in f]
            
            if not dicom_files:
                print(f"Skipping {zip_path.name}: No DICOMs found.")
                continue
                
            # Read the first DICOM file header only (for speed)
            first_file = dicom_files[0]
            with z.open(first_file) as f:
                ds = pydicom.dcmread(io.BytesIO(f.read()), stop_before_pixels=True)

                # Extract specific fields using .get() to avoid errors if a field is missing
                name = ds.get("PatientName", "N/A")
                sex  = ds.get("PatientSex", "N/A")
                age  = ds.get("PatientAge", "N/A")
                
                print(f"--- Info for: {zip_path.name} ---")
                print(f"  Name: {name}")
                print(f"  Sex:  {sex}")
                print(f"  Age:  {age}")
                print("-" * 40)

    except Exception as e:
        print(f"Error processing {zip_path.name}: {e}")
# %%
