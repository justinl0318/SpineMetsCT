#!/usr/bin/env python3
import os
import glob
import numpy as np
from tqdm import tqdm

# Path to validation data
val_dir = "processed_data/SpineMetsCT_npz/val"

# Get list of validation NPZ files
val_files = glob.glob(os.path.join(val_dir, "*.npz"))
print(f"Found {len(val_files)} validation files")

# Check the first few NPZ files to determine their structure
for i, npz_path in enumerate(val_files[:5]):
    print(f"\nNPZ file: {os.path.basename(npz_path)}")
    data = np.load(npz_path)
    print(f"Available keys: {list(data.keys())}")
    
    # Print shape information for each key
    for key in data.keys():
        print(f"  - {key}: shape {data[key].shape}, dtype {data[key].dtype}")
    
    if i >= 4:  # Check 5 files max
        break