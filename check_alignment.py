#!/usr/bin/env python
# Script to check alignment between CT volumes and segmentation masks

import os
import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import cv2

def find_dicom_pair(base_dir, patient_id=None):
    """Find a pair of image and segmentation directories for analysis"""
    # Print diagnostic info
    print(f"Base directory: {base_dir}")
    print(f"Looking for patient ID: {patient_id if patient_id else 'any'}")
    
    # List all items in base_dir
    if os.path.exists(base_dir):
        all_items = os.listdir(base_dir)
        dirs = [d for d in all_items if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Found {len(dirs)} directories in base_dir: {dirs[:5]}{'...' if len(dirs) > 5 else ''}")
    else:
        print(f"Base directory {base_dir} does not exist!")
        return None, None
    
    if patient_id:
        # Look for specific patient
        patient_dir = os.path.join(base_dir, patient_id)
        if not os.path.exists(patient_dir):
            print(f"Patient directory {patient_id} not found")
            return None, None
    else:
        # Get all patient directories
        patient_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) 
                       if os.path.isdir(d) and not d.endswith("LICENSE")]
        if not patient_dirs:
            print("No patient directories found")
            return None, None
        patient_dir = patient_dirs[0]  # Use the first patient directory
        print(f"Selected patient directory: {patient_dir}")
        
    # Find study directories
    study_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]
    if not study_dirs:
        print(f"No study directories found in {patient_dir}")
        return None, None
    
    print(f"Found {len(study_dirs)} study directories: {[os.path.basename(d) for d in study_dirs[:3]]}{'...' if len(study_dirs) > 3 else ''}")
    
    # Find image and segmentation directories
    for study_dir in study_dirs:
        seg_dirs = glob.glob(os.path.join(study_dir, "*Segmentation*"))
        img_dirs = [d for d in glob.glob(os.path.join(study_dir, "*")) 
                    if os.path.isdir(d) and "Segmentation" not in d]
        
        if seg_dirs and img_dirs:
            print(f"Found image dir: {os.path.basename(img_dirs[0])}")
            print(f"Found segmentation dir: {os.path.basename(seg_dirs[0])}")
            return img_dirs[0], seg_dirs[0]
    
    print(f"No matching image and segmentation directories found")
    return None, None

def read_dicom_info(directory, max_files=5):
    """Read a sample of DICOM files and extract orientation information"""
    dicom_files = sorted(glob.glob(os.path.join(directory, "*.dcm")))
    if not dicom_files:
        return []
    
    # Take a sample of files
    sample_files = dicom_files[:min(max_files, len(dicom_files))]
    
    info_list = []
    for file in sample_files:
        try:
            dcm = pydicom.dcmread(file)
            info = {
                'filename': os.path.basename(file),
                'position': dcm.ImagePositionPatient if hasattr(dcm, 'ImagePositionPatient') else None,
                'orientation': dcm.ImageOrientationPatient if hasattr(dcm, 'ImageOrientationPatient') else None,
                'dimensions': (dcm.Rows, dcm.Columns) if hasattr(dcm, 'Rows') and hasattr(dcm, 'Columns') else None,
                'spacing': dcm.PixelSpacing if hasattr(dcm, 'PixelSpacing') else None,
                'thickness': dcm.SliceThickness if hasattr(dcm, 'SliceThickness') else None
            }
            info_list.append(info)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    return info_list

def read_dicom_series(directory, return_info=False):
    """Read all DICOM files in the directory and return as a 3D volume"""
    dicom_files = sorted(glob.glob(os.path.join(directory, "*.dcm")))
    if not dicom_files:
        return None if not return_info else (None, None)

    # Read all slices
    slices = []
    for file in dicom_files:
        try:
            dcm = pydicom.dcmread(file)
            slices.append(dcm)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not slices:
        return None if not return_info else (None, None)
    
    # Sort slices by slice location
    if hasattr(slices[0], 'ImagePositionPatient'):
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else 0)
    
    # Extract orientation info for the first and last slice
    slice_info = []
    if slices:
        for idx in [0, -1]:
            dcm = slices[idx]
            info = {
                'position': dcm.ImagePositionPatient if hasattr(dcm, 'ImagePositionPatient') else None,
                'orientation': dcm.ImageOrientationPatient if hasattr(dcm, 'ImageOrientationPatient') else None,
                'dimensions': (dcm.Rows, dcm.Columns) if hasattr(dcm, 'Rows') and hasattr(dcm, 'Columns') else None
            }
            slice_info.append(info)
    
    # Extract pixel arrays from each slice
    vol_data = []
    for dcm in slices:
        try:
            # Convert to Hounsfield Units if modality is CT
            if hasattr(dcm, 'Modality') and dcm.Modality == 'CT':
                pixel_array = dcm.pixel_array.astype(np.int16)
                # Convert to Hounsfield Units (HU)
                intercept = dcm.RescaleIntercept if hasattr(dcm, 'RescaleIntercept') else 0
                slope = dcm.RescaleSlope if hasattr(dcm, 'RescaleSlope') else 1
                pixel_array = pixel_array * slope + intercept
            else:
                # For other modalities, use pixel array directly
                pixel_array = dcm.pixel_array
            
            vol_data.append(pixel_array)
        except Exception as e:
            print(f"Error processing slice: {e}")
    
    if not vol_data:
        return None if not return_info else (None, None)
    
    # Stack slices to form a 3D volume
    volume = np.stack(vol_data)
    
    if return_info:
        return volume, slice_info
    return volume

def read_segmentation_series(directory, return_info=False):
    """Read all segmentation DICOM files and return as a 3D binary mask"""
    dicom_files = sorted(glob.glob(os.path.join(directory, "*.dcm")))
    if not dicom_files:
        return None if not return_info else (None, None)
    
    # Read all slices
    slices = []
    for file in dicom_files:
        try:
            dcm = pydicom.dcmread(file)
            slices.append(dcm)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not slices:
        return None if not return_info else (None, None)
    
    # Sort slices by slice location if available
    if hasattr(slices[0], 'ImagePositionPatient'):
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else 0)
    
    # Extract orientation info for the first and last slice
    slice_info = []
    if slices:
        for idx in [0, -1]:
            dcm = slices[idx]
            info = {
                'position': dcm.ImagePositionPatient if hasattr(dcm, 'ImagePositionPatient') else None,
                'orientation': dcm.ImageOrientationPatient if hasattr(dcm, 'ImageOrientationPatient') else None,
                'dimensions': (dcm.Rows, dcm.Columns) if hasattr(dcm, 'Rows') and hasattr(dcm, 'Columns') else None
            }
            slice_info.append(info)
    
    # Extract pixel arrays from each slice
    mask_data = []
    for dcm in slices:
        try:
            pixel_array = dcm.pixel_array
            # Check for multi-frame
            if len(pixel_array.shape) > 2:
                print(f"Multi-dimensional mask found with shape {pixel_array.shape}")
                # Take all frames
                for i in range(pixel_array.shape[0]):
                    mask = pixel_array[i].astype(bool).astype(np.uint8)
                    mask_data.append(mask)
            else:
                mask = pixel_array.astype(bool).astype(np.uint8)
                mask_data.append(mask)
        except Exception as e:
            print(f"Error processing slice: {e}")
    
    if not mask_data:
        return None if not return_info else (None, None)
    
    # Stack slices to form a 3D volume
    mask_volume = np.stack(mask_data)
    
    if return_info:
        return mask_volume, slice_info
    return mask_volume

def try_different_alignments(volume, masks, output_prefix="alignment_test"):
    """Try different alignments and visualize the results"""
    if volume is None or masks is None:
        print("No volume or mask data to visualize")
        return
    
    # Find slices with mask content
    mask_sums = [np.sum(masks[i]) for i in range(masks.shape[0])]
    if sum(mask_sums) == 0:
        print("No mask content found in any slice")
        return
    
    # Find slices with significant mask content
    mask_indices = np.argsort(mask_sums)[-3:]  # Get indices of top 3 slices with most mask content
    
    # Ensure we don't exceed the volume dimensions
    vol_slices = volume.shape[0]
    mask_slices = masks.shape[0]
    
    # Calculate potential scaling factor between masks and volume
    scale_z = mask_slices / vol_slices
    
    # For each mask slice with content, try different alignments
    for mask_idx in mask_indices:
        # Skip if mask index is out of bounds
        if mask_idx >= mask_slices:
            continue
            
        # Map mask index to volume index
        vol_idx = min(int(mask_idx / scale_z), vol_slices - 1)
        
        # Get the mask slice
        mask_slice = masks[mask_idx]
        
        # Create variations:
        # 1. Original mask
        mask_original = mask_slice
        
        # 2. Vertical flip
        mask_vertical_flip = np.flipud(mask_original)
        
        # 3. Horizontal flip
        mask_horizontal_flip = np.fliplr(mask_original)
        
        # 4. Both flips
        mask_both_flips = np.fliplr(np.flipud(mask_original))
        
        # Try multiple CT slices around the estimated corresponding slice
        offsets = [-5, -2, 0, 2, 5]
        
        for offset in offsets:
            test_vol_idx = vol_idx + offset
            if test_vol_idx < 0 or test_vol_idx >= vol_slices:
                continue
                
            # Get the volume slice
            vol_slice = volume[test_vol_idx]
            
            # Normalize for visualization
            vol_display = vol_slice.copy()
            if vol_display.max() > vol_display.min():
                vol_display = (vol_display - vol_display.min()) / (vol_display.max() - vol_display.min()) * 255
            vol_display = vol_display.astype(np.uint8)
            
            # Create a figure to show the different alignments
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Display settings
            vmin = np.percentile(vol_slice, 1)
            vmax = np.percentile(vol_slice, 99)
            
            # Original mask
            axes[0, 0].imshow(vol_slice, cmap='gray', vmin=vmin, vmax=vmax)
            mask_rgb = np.zeros((*mask_original.shape, 4))
            mask_rgb[mask_original > 0] = [1, 0, 0, 0.5]  # Red with alpha=0.5
            axes[0, 0].imshow(mask_rgb)
            axes[0, 0].set_title(f"Original Mask\nMask idx: {mask_idx}, Vol idx: {test_vol_idx}")
            axes[0, 0].axis('off')
            
            # Vertical flip
            axes[0, 1].imshow(vol_slice, cmap='gray', vmin=vmin, vmax=vmax)
            mask_rgb = np.zeros((*mask_vertical_flip.shape, 4))
            mask_rgb[mask_vertical_flip > 0] = [0, 1, 0, 0.5]  # Green with alpha=0.5
            axes[0, 1].imshow(mask_rgb)
            axes[0, 1].set_title(f"Vertical Flip\nMask idx: {mask_idx}, Vol idx: {test_vol_idx}")
            axes[0, 1].axis('off')
            
            # Horizontal flip
            axes[1, 0].imshow(vol_slice, cmap='gray', vmin=vmin, vmax=vmax)
            mask_rgb = np.zeros((*mask_horizontal_flip.shape, 4))
            mask_rgb[mask_horizontal_flip > 0] = [0, 0, 1, 0.5]  # Blue with alpha=0.5
            axes[1, 0].imshow(mask_rgb)
            axes[1, 0].set_title(f"Horizontal Flip\nMask idx: {mask_idx}, Vol idx: {test_vol_idx}")
            axes[1, 0].axis('off')
            
            # Both flips
            axes[1, 1].imshow(vol_slice, cmap='gray', vmin=vmin, vmax=vmax)
            mask_rgb = np.zeros((*mask_both_flips.shape, 4))
            mask_rgb[mask_both_flips > 0] = [1, 1, 0, 0.5]  # Yellow with alpha=0.5
            axes[1, 1].imshow(mask_rgb)
            axes[1, 1].set_title(f"Both Flips\nMask idx: {mask_idx}, Vol idx: {test_vol_idx}")
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            output_file = f"{output_prefix}_mask{mask_idx}_vol{test_vol_idx}.png"
            plt.savefig(output_file)
            print(f"Saved visualization to {output_file}")
            plt.close(fig)

def analyze_segmentation_values(img_dir, seg_dir):
    """Analyze the segmentation values to detect potential issues"""
    print("\nAnalyzing segmentation values:")
    
    # Read a sample of segmentation files
    dicom_files = sorted(glob.glob(os.path.join(seg_dir, "*.dcm")))
    if not dicom_files:
        print("No segmentation files found")
        return
    
    # Take a sample of files
    sample_files = dicom_files[:min(5, len(dicom_files))]
    
    unique_values = set()
    for file in sample_files:
        try:
            dcm = pydicom.dcmread(file)
            pixel_array = dcm.pixel_array
            
            # Check if multi-frame
            if len(pixel_array.shape) > 2:
                for i in range(pixel_array.shape[0]):
                    unique_values.update(np.unique(pixel_array[i]))
            else:
                unique_values.update(np.unique(pixel_array))
                
            # Check if there's any metadata about the segment meanings
            if hasattr(dcm, 'SegmentSequence'):
                print("\nSegment information found:")
                for i, segment in enumerate(dcm.SegmentSequence):
                    print(f"Segment {i}:")
                    if hasattr(segment, 'SegmentLabel'):
                        print(f"  Label: {segment.SegmentLabel}")
                    if hasattr(segment, 'SegmentDescription'):
                        print(f"  Description: {segment.SegmentDescription}")
                    if hasattr(segment, 'SegmentAlgorithmType'):
                        print(f"  Algorithm: {segment.SegmentAlgorithmType}")
        
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    print(f"\nUnique pixel values in segmentation: {sorted(unique_values)}")
    print(f"This suggests the segmentation {'is binary' if len(unique_values) <= 2 else 'has multiple labels'}")

def main():
    parser = argparse.ArgumentParser(description='Check alignment between CT volumes and segmentation masks')
    parser.add_argument('--input_dir', type=str, default='SpineMetsCT_Data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG',
                        help='Input directory containing the SpineMetsCT dataset')
    parser.add_argument('--patient_id', type=str, default=None,
                        help='Specific patient ID to process (optional)')
    parser.add_argument('--output_prefix', type=str, default='alignment_test',
                        help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Find a pair of image and segmentation directories
    img_dir, seg_dir = find_dicom_pair(args.input_dir, args.patient_id)
    if not img_dir or not seg_dir:
        print("Failed to find suitable image and segmentation directories")
        return
    
    # Compare basic DICOM metadata
    print("\nComparing DICOM metadata:")
    img_info = read_dicom_info(img_dir)
    seg_info = read_dicom_info(seg_dir)
    
    if img_info and seg_info:
        print("\nImage DICOM info:")
        for i, info in enumerate(img_info):
            print(f"File {i+1}: {info['filename']}")
            print(f"  Position: {info['position']}")
            print(f"  Orientation: {info['orientation']}")
            print(f"  Dimensions: {info['dimensions']}")
            print(f"  Spacing: {info['spacing']}")
            print(f"  Thickness: {info['thickness']}")
        
        print("\nSegmentation DICOM info:")
        for i, info in enumerate(seg_info):
            print(f"File {i+1}: {info['filename']}")
            print(f"  Position: {info['position']}")
            print(f"  Orientation: {info['orientation']}")
            print(f"  Dimensions: {info['dimensions']}")
    
    # Read the image and segmentation volumes
    print("\nReading volumes:")
    volume, img_info = read_dicom_series(img_dir, return_info=True)
    masks, seg_info = read_segmentation_series(seg_dir, return_info=True)
    
    if volume is not None and masks is not None:
        print(f"Image volume shape: {volume.shape}")
        print(f"Mask volume shape: {masks.shape}")
        
        # Analyze the first and last slice info
        if img_info and seg_info:
            print("\nImage first slice position:", img_info[0]['position'])
            print("Image last slice position:", img_info[1]['position'])
            print("Seg first slice position:", seg_info[0]['position'])
            print("Seg last slice position:", seg_info[1]['position'])
        
        # Check if shapes match
        if volume.shape != masks.shape:
            print("Warning: Volume and mask shapes don't match!")
            
            # Calculate rough mapping between the volumes based on positions
            if img_info and seg_info and img_info[0]['position'] and seg_info[0]['position']:
                img_z_start = img_info[0]['position'][2]
                img_z_end = img_info[1]['position'][2]
                seg_z_start = seg_info[0]['position'][2]
                seg_z_end = seg_info[1]['position'][2]
                
                img_z_range = abs(img_z_end - img_z_start)
                seg_z_range = abs(seg_z_end - seg_z_start)
                
                print(f"Image Z range: {img_z_range:.2f} mm over {volume.shape[0]} slices")
                print(f"Segmentation Z range: {seg_z_range:.2f} mm over {masks.shape[0]} slices")
                
                # Check for overlap
                z_min = max(min(img_z_start, img_z_end), min(seg_z_start, seg_z_end))
                z_max = min(max(img_z_start, img_z_end), max(seg_z_start, seg_z_end))
                if z_min <= z_max:
                    overlap = abs(z_max - z_min)
                    print(f"Z-axis overlap: {overlap:.2f} mm ({overlap/img_z_range*100:.1f}% of image range)")
                else:
                    print("Warning: No Z-axis overlap detected!")
        
        # Try different alignments
        try_different_alignments(volume, masks, args.output_prefix)
        
        # Analyze the segmentation values
        analyze_segmentation_values(img_dir, seg_dir)
    else:
        if volume is None:
            print("Failed to read image volume")
        if masks is None:
            print("Failed to read segmentation volume")

if __name__ == "__main__":
    main()
