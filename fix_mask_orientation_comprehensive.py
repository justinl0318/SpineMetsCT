#!/usr/bin/env python3
# Script to investigate and fix mask orientation issues in the SpineMetsCT dataset

import os
import glob
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def read_dicom_image_slice(file_path):
    """Read a single DICOM image slice"""
    try:
        dcm = pydicom.dcmread(file_path)
        pixel_array = dcm.pixel_array.astype(np.int16)
        
        # Convert to Hounsfield Units (HU)
        intercept = dcm.RescaleIntercept if hasattr(dcm, 'RescaleIntercept') else 0
        slope = dcm.RescaleSlope if hasattr(dcm, 'RescaleSlope') else 1
        hu_image = pixel_array * slope + intercept
        
        # Apply windowing for better visualization
        window_center = -600
        window_width = 1500
        min_value = window_center - window_width // 2
        max_value = window_center + window_width // 2
        
        # Apply windowing and normalize to 0-255
        windowed_image = np.clip(hu_image, min_value, max_value)
        normalized_image = ((windowed_image - min_value) / (max_value - min_value) * 255).astype(np.uint8)
        
        return normalized_image, dcm
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def read_dicom_segmentation_slice(file_path, frame_index=None):
    """Read a segmentation slice or frame from a DICOM file"""
    try:
        dcm = pydicom.dcmread(file_path)
        pixel_array = dcm.pixel_array
        
        # Handle multi-frame segmentations
        if len(pixel_array.shape) > 2:
            if frame_index is not None and frame_index < pixel_array.shape[0]:
                mask = pixel_array[frame_index].astype(bool).astype(np.uint8)
            else:
                # Default to first frame if no index specified
                mask = pixel_array[0].astype(bool).astype(np.uint8)
        else:
            mask = pixel_array.astype(bool).astype(np.uint8)
        
        return mask, dcm
    except Exception as e:
        print(f"Error reading segmentation {file_path}: {e}")
        return None, None

def extract_orientation_from_dicom(dcm):
    """Extract orientation information from DICOM file"""
    orientation_info = {}
    
    # ImageOrientationPatient: Direction cosines of first row and first column
    if hasattr(dcm, 'ImageOrientationPatient'):
        orientation_info['ImageOrientationPatient'] = dcm.ImageOrientationPatient
    
    # ImagePositionPatient: The x, y, and z coordinates of the upper left hand corner of the image
    if hasattr(dcm, 'ImagePositionPatient'):
        orientation_info['ImagePositionPatient'] = dcm.ImagePositionPatient
    
    # PatientPosition: Position of patient during acquisition
    if hasattr(dcm, 'PatientPosition'):
        orientation_info['PatientPosition'] = dcm.PatientPosition
    
    # FrameOfReferenceUID: Identifies the frame of reference
    if hasattr(dcm, 'FrameOfReferenceUID'):
        orientation_info['FrameOfReferenceUID'] = dcm.FrameOfReferenceUID
    
    return orientation_info

def apply_transformations(image, flip_vertical=False, flip_horizontal=False):
    """Apply vertical and/or horizontal flips to an image"""
    result = image.copy()
    
    if flip_vertical:
        result = np.flipud(result)
    
    if flip_horizontal:
        result = np.fliplr(result)
    
    return result

def visualize_mask_orientations(image, mask, output_file=None):
    """Visualize all possible mask orientations over the image"""
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original mask
    axs[0, 0].imshow(image, cmap='gray')
    mask_rgb = np.zeros((*image.shape, 3), dtype=np.uint8)
    mask_rgb[mask > 0] = [255, 0, 0]  # Red color for mask
    axs[0, 0].imshow(mask_rgb, alpha=0.5)
    axs[0, 0].set_title(f"Original Mask")
    axs[0, 0].axis('off')
    
    # Vertical flip
    vert_mask = np.flipud(mask)
    axs[0, 1].imshow(image, cmap='gray')
    mask_rgb = np.zeros((*image.shape, 3), dtype=np.uint8)
    mask_rgb[vert_mask > 0] = [0, 255, 0]  # Green color for mask
    axs[0, 1].imshow(mask_rgb, alpha=0.5)
    axs[0, 1].set_title(f"Vertical Flip")
    axs[0, 1].axis('off')
    
    # Horizontal flip
    horiz_mask = np.fliplr(mask)
    axs[1, 0].imshow(image, cmap='gray')
    mask_rgb = np.zeros((*image.shape, 3), dtype=np.uint8)
    mask_rgb[horiz_mask > 0] = [0, 0, 255]  # Blue color for mask
    axs[1, 0].imshow(mask_rgb, alpha=0.5)
    axs[1, 0].set_title(f"Horizontal Flip")
    axs[1, 0].axis('off')
    
    # Both flips
    both_mask = np.flipud(np.fliplr(mask))
    axs[1, 1].imshow(image, cmap='gray')
    mask_rgb = np.zeros((*image.shape, 3), dtype=np.uint8)
    mask_rgb[both_mask > 0] = [255, 255, 0]  # Yellow color for mask
    axs[1, 1].imshow(mask_rgb, alpha=0.5)
    axs[1, 1].set_title(f"Both Flips")
    axs[1, 1].axis('off')
    
    # Add title with image/mask index information
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()

def find_patient_data_pair():
    """Find a patient with both image and segmentation data for analysis"""
    base_dir = "SpineMetsCT_Data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG"
    
    # Get all patient directories
    patient_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) 
                   if os.path.isdir(d) and not d.endswith("LICENSE")]
    
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        print(f"Checking patient: {patient_id}")
        
        # Find study directories
        study_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]
        
        for study_dir in study_dirs:
            # Find image and segmentation directories
            img_dirs = [d for d in glob.glob(os.path.join(study_dir, "*")) 
                       if os.path.isdir(d) and "Segmentation" not in d]
            seg_dirs = glob.glob(os.path.join(study_dir, "*Segmentation*"))
            
            if img_dirs and seg_dirs:
                # Found a pair
                return patient_id, img_dirs[0], seg_dirs[0]
    
    return None, None, None

def analyze_mask_transformations(img_dir, seg_dir, output_dir="orientation_tests"):
    """Analyze and visualize different mask transformations to find the correct orientation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image DICOM files
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.dcm")))
    if not img_files:
        print(f"No image files found in {img_dir}")
        return
    
    # Get list of segmentation DICOM files
    seg_files = sorted(glob.glob(os.path.join(seg_dir, "*.dcm")))
    if not seg_files:
        print(f"No segmentation files found in {seg_dir}")
        return
    
    print(f"Found {len(img_files)} image files and {len(seg_files)} segmentation files")
    
    # Check if we have multi-frame segmentations
    is_multi_frame = False
    total_frames = 0
    
    try:
        seg_dcm = pydicom.dcmread(seg_files[0])
        if hasattr(seg_dcm, 'NumberOfFrames'):
            is_multi_frame = True
            total_frames = int(seg_dcm.NumberOfFrames)
            print(f"Found multi-frame segmentation with {total_frames} frames")
    except Exception as e:
        print(f"Error reading segmentation file: {e}")
    
    # Extract orientation information from a sample image and segmentation
    img_sample, img_dcm = read_dicom_image_slice(img_files[len(img_files) // 2])
    img_orientation = extract_orientation_from_dicom(img_dcm) if img_dcm else {}
    
    print("\nImage orientation information:")
    for key, value in img_orientation.items():
        print(f"- {key}: {value}")
    
    seg_sample, seg_dcm = read_dicom_segmentation_slice(seg_files[0])
    seg_orientation = extract_orientation_from_dicom(seg_dcm) if seg_dcm else {}
    
    print("\nSegmentation orientation information:")
    for key, value in seg_orientation.items():
        print(f"- {key}: {value}")
    
    # Compare frames of reference
    if 'FrameOfReferenceUID' in img_orientation and 'FrameOfReferenceUID' in seg_orientation:
        if img_orientation['FrameOfReferenceUID'] == seg_orientation['FrameOfReferenceUID']:
            print("\nImage and segmentation have the SAME frame of reference UID")
        else:
            print("\nImage and segmentation have DIFFERENT frame of reference UIDs")
    
    # Sample images and masks at different indices to visualize
    sample_count = min(5, len(img_files))
    sample_indices = [i * len(img_files) // sample_count for i in range(sample_count)]
    
    # Determine how to sample frames from a multi-frame segmentation
    frame_indices = []
    if is_multi_frame:
        # Sample frames at roughly the same positions as the image samples
        frame_indices = [i * total_frames // sample_count for i in range(sample_count)]
    
    # Create visualizations for each sample
    for i, img_idx in enumerate(sample_indices):
        img_file = img_files[img_idx]
        img, _ = read_dicom_image_slice(img_file)
        
        if not img is None:
            # Get corresponding segmentation
            if is_multi_frame:
                # Use frame from multi-frame segmentation
                frame_idx = frame_indices[i]
                mask, _ = read_dicom_segmentation_slice(seg_files[0], frame_idx)
                output_file = os.path.join(output_dir, f"orientation_test_mask{frame_idx}_vol{img_idx}.png")
            else:
                # Use corresponding segmentation file if available
                seg_idx = min(img_idx, len(seg_files) - 1)
                mask, _ = read_dicom_segmentation_slice(seg_files[seg_idx])
                output_file = os.path.join(output_dir, f"orientation_test_mask{seg_idx}_vol{img_idx}.png")
            
            if not mask is None:
                visualize_mask_orientations(img, mask, output_file)
    
    print(f"\nCreated {sample_count} visualizations in {output_dir}")

def find_best_orientation_strategy():
    """Test different orientation strategies and recommend the best one"""
    # Find a patient data pair to analyze
    patient_id, img_dir, seg_dir = find_patient_data_pair()
    
    if not patient_id or not img_dir or not seg_dir:
        print("Could not find suitable patient data for analysis")
        return
    
    print(f"\nAnalyzing orientation for patient {patient_id}")
    print(f"Image directory: {img_dir}")
    print(f"Segmentation directory: {seg_dir}")
    
    # Analyze transformations
    analyze_mask_transformations(img_dir, seg_dir)
    
    # Based on analysis of images, recommend best strategy
    print("\nBased on visual analysis, the current preprocessing code:")
    print("1. Applies a vertical flip to segmentation masks")
    print("2. Does not apply a horizontal flip")
    
    print("\nRECOMMENDATIONS:")
    print("1. Look at the generated visualization images to determine the correct orientation")
    print("2. Compare the four options (original, vertical flip, horizontal flip, both flips)")
    print("3. Update the preprocess_spinemetsCT.py script with the correct flipping strategy")
    print("4. You may need different flipping strategies for different patients/studies")
    print("5. Consider using DICOM metadata to automatically determine correct orientation")

def fix_preprocess_script():
    """Generate recommended fixes for the preprocessing script"""
    print("\nRecommended changes for preprocess_spinemetsCT.py:")
    
    print("""
    # Replace the current code:
    fixed_masks = np.zeros_like(masks)
    for i in range(masks.shape[0]):
        fixed_masks[i] = np.flipud(masks[i])  # Flip vertically (up-down)
    masks = fixed_masks
    print(f"      Applied vertical flip to fix mask orientation")
    
    # With this more comprehensive orientation correction:
    fixed_masks = np.zeros_like(masks)
    for i in range(masks.shape[0]):
        # Apply both vertical and horizontal flips based on your analysis
        # Choose the correct combination based on your visualization results
        # Option 1: No flips (original)
        # fixed_masks[i] = masks[i]
        
        # Option 2: Vertical flip only (current implementation)
        # fixed_masks[i] = np.flipud(masks[i])
        
        # Option 3: Horizontal flip only
        # fixed_masks[i] = np.fliplr(masks[i])
        
        # Option 4: Both vertical and horizontal flips
        fixed_masks[i] = np.flipud(np.fliplr(masks[i]))
    masks = fixed_masks
    print(f"      Applied orientation correction to fix mask alignment")
    """)
    
    print("\nBased on the image you shared, 'Both Flips' (vertical + horizontal) or another combination might be needed.")
    print("The visualization tool will help you determine which transformation works best.")

def main():
    parser = argparse.ArgumentParser(description='Analyze and fix mask orientation issues in the SpineMetsCT dataset')
    parser.add_argument('--action', type=str, default='visualize', choices=['visualize', 'recommend', 'fix'],
                      help='Action to perform: visualize orientations, recommend strategy, or print fix')
    args = parser.parse_args()
    
    if args.action == 'visualize':
        find_best_orientation_strategy()
    elif args.action == 'recommend':
        find_best_orientation_strategy()
        fix_preprocess_script()
    elif args.action == 'fix':
        fix_preprocess_script()
    
if __name__ == "__main__":
    main()
