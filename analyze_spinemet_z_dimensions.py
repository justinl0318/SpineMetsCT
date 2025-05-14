#!/usr/bin/env python
# Script to analyze the structure of the SpineMetsCT dataset and investigate Z-dimension discrepancies

import os
import glob
import numpy as np
import pydicom
import pandas as pd
from collections import defaultdict

def analyze_dataset_structure(base_dir):
    """Analyze the directory structure of the SpineMetsCT dataset"""
    print(f"Analyzing dataset structure in: {base_dir}")
    
    # Get all patient directories
    patient_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) 
                   if os.path.isdir(d) and not d.endswith("LICENSE")]
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Collect structure statistics
    structure_data = []
    
    for patient_dir in patient_dirs[:5]:  # Limit to 5 patients for speed
        patient_id = os.path.basename(patient_dir)
        print(f"\nProcessing patient: {patient_id}")
        
        # Find study directories
        study_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]
        print(f"  Found {len(study_dirs)} study directories")
        
        for study_dir in study_dirs:
            study_name = os.path.basename(study_dir)
            print(f"  Study: {study_name}")
            
            # Find image directories (exclude segmentation directories)
            img_dirs = [d for d in glob.glob(os.path.join(study_dir, "*")) 
                       if os.path.isdir(d) and "Segmentation" not in d]
            
            # Find segmentation directories
            seg_dirs = glob.glob(os.path.join(study_dir, "*Segmentation*"))
            
            print(f"    Found {len(img_dirs)} image directories and {len(seg_dirs)} segmentation directories")
            
            # Analyze each image directory
            for img_dir in img_dirs:
                img_name = os.path.basename(img_dir)
                print(f"    Image: {img_name}")
                
                # Count DICOM files
                dicom_files = glob.glob(os.path.join(img_dir, "*.dcm"))
                img_slice_count = len(dicom_files)
                print(f"      Found {img_slice_count} image slices")
                
                # Get one slice to determine image dimensions
                img_dimensions = (0, 0)
                if dicom_files:
                    try:
                        dcm = pydicom.dcmread(dicom_files[0])
                        img_dimensions = dcm.pixel_array.shape
                        print(f"      Image dimensions: {img_dimensions}")
                    except Exception as e:
                        print(f"      Error reading image DICOM: {e}")
                
                # Analyze each segmentation directory
                for seg_dir in seg_dirs:
                    seg_name = os.path.basename(seg_dir)
                    print(f"      Segmentation: {seg_name}")
                    
                    # Count segmentation DICOM files
                    seg_files = glob.glob(os.path.join(seg_dir, "*.dcm"))
                    seg_slice_count = len(seg_files)
                    print(f"        Found {seg_slice_count} segmentation slices")
                    
                    # Check for multi-frame segmentations
                    multi_frame = False
                    total_frames = 0
                    seg_dimensions = (0, 0)
                    
                    if seg_files:
                        try:
                            seg_dcm = pydicom.dcmread(seg_files[0])
                            
                            # Check if this is a multi-frame DICOM
                            if hasattr(seg_dcm, 'NumberOfFrames'):
                                multi_frame = True
                                total_frames = int(seg_dcm.NumberOfFrames)
                                print(f"        This is a multi-frame segmentation with {total_frames} frames")
                            
                            # Get segmentation dimensions
                            pixel_array = seg_dcm.pixel_array
                            seg_dimensions = pixel_array.shape if not multi_frame else pixel_array.shape[1:]
                            print(f"        Segmentation dimensions: {seg_dimensions}")
                            
                            # Check if we have segment sequence information
                            if hasattr(seg_dcm, 'SegmentSequence'):
                                print(f"        Found segment sequence with {len(seg_dcm.SegmentSequence)} segments")
                                
                        except Exception as e:
                            print(f"        Error reading segmentation DICOM: {e}")
                    
                    # Record the data
                    structure_data.append({
                        'patient_id': patient_id,
                        'study_name': study_name,
                        'image_name': img_name,
                        'segmentation_name': seg_name,
                        'image_slice_count': img_slice_count,
                        'segmentation_slice_count': seg_slice_count,
                        'image_dimensions': img_dimensions,
                        'segmentation_dimensions': seg_dimensions,
                        'is_multi_frame': multi_frame,
                        'multi_frame_count': total_frames if multi_frame else 0
                    })
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(structure_data)
    
    # Analyze Z-dimension discrepancies
    print("\nAnalyzing Z-dimension discrepancies:")
    df['z_dimension_match'] = df['image_slice_count'] == df['segmentation_slice_count']
    
    matched = df['z_dimension_match'].sum()
    total = len(df)
    print(f"Z-dimensions match in {matched} out of {total} cases ({matched/total*100:.2f}%)")
    
    # Check if multi-frame is the cause
    if 'is_multi_frame' in df.columns:
        multi_frame_cases = df[df['is_multi_frame']].shape[0]
        print(f"Multi-frame segmentations found in {multi_frame_cases} cases")
        
        # Check if the total frame count matches the image slice count
        df['frame_count_matches'] = df.apply(
            lambda row: row['multi_frame_count'] == row['image_slice_count'] if row['is_multi_frame'] else False, 
            axis=1
        )
        
        frame_matches = df['frame_count_matches'].sum()
        print(f"Multi-frame count matches image slice count in {frame_matches} out of {multi_frame_cases} multi-frame cases")
    
    return df

def analyze_npz_files(npz_dir):
    """Analyze the NPZ files generated by the preprocessing script"""
    print(f"\nAnalyzing NPZ files in: {npz_dir}")
    
    npz_files = glob.glob(os.path.join(npz_dir, "**/*.npz"), recursive=True)
    print(f"Found {len(npz_files)} NPZ files")
    
    if not npz_files:
        return None
    
    npz_data = []
    
    # Analyze a sample of NPZ files
    sample_size = min(20, len(npz_files))
    for npz_file in npz_files[:sample_size]:
        try:
            data = np.load(npz_file)
            
            # Extract information
            filename = os.path.basename(npz_file)
            
            # Check shapes of data
            imgs_shape = data['imgs'].shape if 'imgs' in data else None
            gts_shape = data['gts'].shape if 'gts' in data else None
            
            # Check for resampling clues in filename
            sample_info = extract_sample_info(filename)
            
            npz_data.append({
                'filename': filename,
                'imgs_shape': imgs_shape,
                'gts_shape': gts_shape,
                'sample_info': sample_info
            })
            
        except Exception as e:
            print(f"Error analyzing {npz_file}: {e}")
    
    # Print summary of NPZ data
    print("\nNPZ File Analysis Summary:")
    for item in npz_data:
        print(f"File: {item['filename']}")
        print(f"  Images shape: {item['imgs_shape']}")
        print(f"  Ground truth shape: {item['gts_shape']}")
        if item['sample_info']:
            print(f"  Sample info: {item['sample_info']}")
        print()
    
    return npz_data

def extract_sample_info(filename):
    """Extract sample information from NPZ filename"""
    # Expected format: {patient_id}_{study_name}_{img_name}_{seg_name}_{i}.npz
    try:
        # Remove extension
        basename = os.path.basename(filename).replace('.npz', '')
        
        # Split off slice index
        parts = basename.rsplit('_', 1)
        if len(parts) != 2:
            return None
        
        slice_idx = int(parts[1])
        
        # Further split remaining parts
        name_parts = parts[0].split('_', 3)
        if len(name_parts) < 4:
            return None
            
        return {
            'patient_id': name_parts[0],
            'study_name': name_parts[1],
            'image_name': name_parts[2],
            'segmentation_name': name_parts[3],
            'slice_idx': slice_idx
        }
    except:
        return None

def analyze_resampling_impact():
    """Analyze the impact of resampling masks in the preprocessing script"""
    print("\nAnalyzing the impact of z-dimension resampling on segmentation masks:")
    
    # This analysis would ideally involve testing how the resampling affects mask quality
    # We'll describe the approach used in the preprocessing code
    
    print("The preprocessing script uses the following approach when z-dimensions don't match:")
    print("1. When segmentation has different number of slices than the image volume:")
    print("   - Create an empty mask array with the same dimensions as the image volume")
    print("   - Calculate a scaling factor: mask_slices / volume_slices")
    print("   - For each volume slice, find the corresponding mask slice using the scaling factor")
    print("   - Use nearest neighbor interpolation to resize in x-y plane if needed")
    
    print("\nPotential issues with this approach:")
    print("1. Information loss when downsampling (if mask has more slices than volume)")
    print("2. Interpolation artifacts when upsampling (if mask has fewer slices than volume)")
    print("3. Slice alignment issues if the first/last slices don't correspond correctly")
    print("4. Loss of thin or small structures that might span fewer slices in the original segmentation")
    
    print("\nRecommendations:")
    print("1. Check the DICOM metadata to ensure proper slice alignment")
    print("2. Consider using more sophisticated interpolation for z-dimension resampling")
    print("3. Validate resampled masks against original segmentations for accuracy")

def main():
    base_dir = "SpineMetsCT_Data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG"
    npz_dir = "processed_data/SpineMetsCT_npz"
    
    # Analyze dataset structure
    structure_data = analyze_dataset_structure(base_dir)
    
    # Analyze NPZ files
    npz_data = analyze_npz_files(npz_dir)
    
    # Analyze resampling impact
    analyze_resampling_impact()
    
    print("\nSummary:")
    print("1. The SpineMetsCT dataset has z-dimension mismatches between image volumes and segmentation masks")
    print("2. The preprocessing script handles this by resampling masks to match volume dimensions")
    print("3. This resampling may affect the quality and accuracy of the segmentation masks")
    print("4. Some segmentations are stored as multi-frame DICOM files, which need special handling")
    
    print("\nRecommendations for improving the preprocessing:")
    print("1. Check and use spatial metadata from DICOM files for proper alignment")
    print("2. Consider more advanced interpolation methods for z-axis resampling")
    print("3. Add validation step to compare original vs. resampled masks")
    print("4. For multi-frame segmentations, ensure all frames are properly extracted")

if __name__ == "__main__":
    main()
