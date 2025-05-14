#!/usr/bin/env python
# Script to analyze the Spine-Mets-CT-SEG dataset structure and find metastatic segmentations

import os
import glob
import pandas as pd
import numpy as np
import pydicom
import re
from pprint import pprint

def read_clinical_excel(file_path):
    """Read the clinical Excel file and extract relevant information"""
    print(f"Reading clinical data from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File {file_path} not found!")
        return None
    
    # Read the Excel file
    try:
        # Try reading with different engines in case one fails
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except:
            df = pd.read_excel(file_path, engine='xlrd')
            
        print(f"Successfully loaded Excel file with {len(df)} rows")
        
        # Print the columns to understand the structure
        print("\nColumns in the Excel file:")
        for col in df.columns:
            print(f"- {col}")
        
        # Print a sample row
        print("\nSample data (first row):")
        for col in df.columns:
            print(f"{col}: {df[col].iloc[0]}")
            
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def find_all_segment_types(base_dir):
    """Scan all segmentation directories to find all segment types"""
    print(f"\nScanning for all segment types in: {base_dir}")
    
    # Get all patient directories
    patient_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) 
                   if os.path.isdir(d) and not d.endswith("LICENSE")]
    
    segment_types = {}
    
    # Process a limited number of patients to avoid excessive runtime
    for patient_dir in patient_dirs[:10]:  # Limit to 10 patients for speed
        patient_id = os.path.basename(patient_dir)
        print(f"\nProcessing patient: {patient_id}")
        
        # Find study directories
        study_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]
        
        for study_dir in study_dirs:
            study_name = os.path.basename(study_dir)
            print(f"  Study: {study_name}")
            
            # Find all segmentation directories
            seg_dirs = glob.glob(os.path.join(study_dir, "*Segmentation*"))
            
            for seg_dir in seg_dirs:
                seg_name = os.path.basename(seg_dir)
                print(f"    Analyzing segmentation: {seg_name}")
                
                # Read a sample DICOM file to check segment information
                dicom_files = glob.glob(os.path.join(seg_dir, "*.dcm"))
                if not dicom_files:
                    print(f"    No DICOM files found in {seg_name}")
                    continue
                
                try:
                    # Read first DICOM file
                    dcm = pydicom.dcmread(dicom_files[0])
                    
                    # Check for segment sequence
                    if hasattr(dcm, 'SegmentSequence'):
                        print(f"    Found segment sequence with {len(dcm.SegmentSequence)} segments")
                        
                        for i, segment in enumerate(dcm.SegmentSequence):
                            label = getattr(segment, 'SegmentLabel', f"Unknown-{i}")
                            print(f"      Segment {i}: {label}")
                            
                            # Count segment types
                            if label not in segment_types:
                                segment_types[label] = 1
                            else:
                                segment_types[label] += 1
                    else:
                        print(f"    No segment sequence found in {os.path.basename(dicom_files[0])}")
                        
                        # Try to check raw pixel values
                        pixel_array = dcm.pixel_array
                        unique_vals = np.unique(pixel_array)
                        print(f"    Unique pixel values: {unique_vals}")
                        
                        # Add to our counts with directory name as key
                        key = f"Unknown (from {seg_name})"
                        if key not in segment_types:
                            segment_types[key] = 1
                        else:
                            segment_types[key] += 1
                            
                except Exception as e:
                    print(f"    Error analyzing {os.path.basename(dicom_files[0])}: {e}")
    
    print("\nFound segment types:")
    for segment_type, count in sorted(segment_types.items(), key=lambda x: -x[1]):
        print(f"- {segment_type}: {count} occurrences")
    
    return segment_types

def find_lesion_segmentations(base_dir):
    """Try to find directories that might contain lesion/metastasis segmentations"""
    print(f"\nLooking for possible lesion segmentations in: {base_dir}")
    
    # Keywords that might indicate lesion/metastasis segmentations
    lesion_keywords = ['lesion', 'tumor', 'mets', 'metastasis', 'metastases', 'met', 'cancer']
    
    # Get all patient directories
    all_dirs = glob.glob(os.path.join(base_dir, "**"), recursive=True)
    
    # Filter directories
    potential_lesion_dirs = []
    for dir_path in all_dirs:
        if not os.path.isdir(dir_path):
            continue
            
        dir_name = os.path.basename(dir_path).lower()
        
        # Check if directory name contains any of our keywords
        if any(keyword in dir_name for keyword in lesion_keywords):
            potential_lesion_dirs.append(dir_path)
    
    print(f"\nFound {len(potential_lesion_dirs)} potential lesion segmentation directories:")
    for dir_path in potential_lesion_dirs:
        print(f"- {dir_path}")
        
    return potential_lesion_dirs

def check_npz_file_contents():
    """Check the contents of existing NPZ files to understand what was preprocessed"""
    print("\nChecking existing NPZ files in processed_data/")
    
    npz_files = glob.glob(os.path.join("processed_data", "**/*.npz"), recursive=True)
    
    if not npz_files:
        print("No NPZ files found!")
        return
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # Analyze a sample of NPZ files
    for npz_file in npz_files[:5]:  # Analyze first 5 files
        print(f"\nAnalyzing: {npz_file}")
        try:
            data = np.load(npz_file)
            print(f"Keys: {list(data.keys())}")
            
            # Check what keys exist and their shapes
            for key in data:
                print(f"- {key}: shape={data[key].shape}, dtype={data[key].dtype}")
                # If binary mask, calculate the percentage of positive pixels
                if key in ['gts', 'mask', 'masks'] and data[key].dtype == np.uint8:
                    positive_ratio = np.sum(data[key] > 0) / np.prod(data[key].shape)
                    print(f"  Positive pixel ratio: {positive_ratio:.6f} ({positive_ratio*100:.4f}%)")
                    
                    # If there are very few positive pixels, this is likely not a vertebra segmentation
                    if positive_ratio < 0.01:  # Less than 1%
                        print("  This mask has very sparse positive pixels, could be lesion segmentation")
                        
                        # Check if the mask has multiple values or just binary
                        unique_vals = np.unique(data[key])
                        print(f"  Unique values in mask: {unique_vals}")
                        
        except Exception as e:
            print(f"Error reading {npz_file}: {e}")

def extract_metadata_from_filename(filename):
    """Try to extract patient, study, image, and segmentation info from NPZ filenames"""
    # Expected format from the preprocessing code:
    # f"{patient_id}_{study_name}_{img_name}_{seg_name}_{i}.npz"
    
    basename = os.path.basename(filename).replace('.npz', '')
    
    # Split but limit the splits to preserve the structure
    parts = basename.rsplit('_', 1)  # Split off the slice index
    if len(parts) != 2:
        return None
    
    try:
        slice_idx = int(parts[1])
        
        # Further split the remaining part
        remaining = parts[0].split('_', 3)  # Try to get patient, study, image, segmentation
        if len(remaining) < 4:
            return None
            
        patient_id = remaining[0]
        study_name = remaining[1]
        img_name = remaining[2]
        seg_name = remaining[3]
        
        return {
            'patient_id': patient_id,
            'study_name': study_name,
            'image_name': img_name,
            'segmentation_name': seg_name,
            'slice_idx': slice_idx
        }
    except:
        return None

def main():
    # Read the clinical data
    clinical_data = read_clinical_excel("Spine-Mets-CT-SEG_Clinical.xlsx")
    
    # Find all segment types in the dataset
    base_dir = "SpineMetsCT_Data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG"
    segment_types = find_all_segment_types(base_dir)
    
    # Look for potential lesion segmentations
    lesion_dirs = find_lesion_segmentations(base_dir)
    
    # Check NPZ file contents
    check_npz_file_contents()
    
    # Analyze file naming pattern of existing NPZ files
    print("\nAnalyzing NPZ filenames to extract segmentation information:")
    npz_files = glob.glob(os.path.join("processed_data", "**/*.npz"), recursive=True)
    
    segmentation_names = {}
    for npz_file in npz_files[:20]:  # Analyze a subset
        metadata = extract_metadata_from_filename(npz_file)
        if metadata:
            seg_name = metadata['segmentation_name']
            if seg_name not in segmentation_names:
                segmentation_names[seg_name] = 1
            else:
                segmentation_names[seg_name] += 1
    
    print("\nSegmentation names found in NPZ files:")
    for name, count in sorted(segmentation_names.items(), key=lambda x: -x[1]):
        print(f"- {name}: {count} occurrences")

if __name__ == "__main__":
    main()
