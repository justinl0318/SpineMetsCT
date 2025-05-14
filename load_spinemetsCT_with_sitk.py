#!/usr/bin/env python3
# Script to load and process SpineMetsCT data using SimpleITK

import os
import glob
import numpy as np
import SimpleITK as sitk
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

def load_ct_volume(ct_dir):
    """Load a CT volume from DICOM directory using SimpleITK"""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ct_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image  # SimpleITK Image

def load_and_resample_seg(seg_path, reference_img):
    """Load and resample segmentation DICOM to match reference CT image"""
    # Read segmentation DICOM-SEG
    seg_img = sitk.ReadImage(seg_path)

    # Resample segmentation to CT space
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_img)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetTransform(sitk.Transform())
    resampled_seg = resample.Execute(seg_img)

    # Convert to numpy
    seg_np = sitk.GetArrayFromImage(resampled_seg)  # (Z, H, W)
    seg_np = (seg_np > 0).astype(np.uint8)  # Binarize
    return seg_np

def normalize_ct_volume(ct_img, window_center=-600, window_width=1500):
    """Apply windowing to CT volume and normalize to 0-255"""
    # Convert to numpy array
    ct_np = sitk.GetArrayFromImage(ct_img)
    
    # Apply windowing
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    ct_np = np.clip(ct_np, min_value, max_value)
    
    # Normalize to 0-255
    ct_np = ((ct_np - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    
    return ct_np

def save_as_npz(patient_id, study_name, img_name, seg_name, ct_np, seg_np, output_dir, slice_range=8):
    """Save CT and segmentation volumes as NPZ files with sliding window"""
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples with sliding window
    samples_saved = 0
    
    # Skip if shapes don't match
    if ct_np.shape != seg_np.shape:
        print(f"Shape mismatch: CT {ct_np.shape}, segmentation {seg_np.shape}")
        return samples_saved
    
    # Create sliding windows with 50% overlap
    for i in range(0, len(ct_np) - slice_range + 1, slice_range // 2):
        # Get a range of slices
        ct_slice = ct_np[i:i+slice_range]
        seg_slice = seg_np[i:i+slice_range]
        
        # Skip if no segmentation in this slice range
        if np.sum(seg_slice) == 0:
            continue
        
        # Create a unique ID for this sample
        sample_id = f"{patient_id}_{study_name}_{img_name}_{seg_name}_{i}"
        output_path = os.path.join(output_dir, f"{sample_id}.npz")
        
        # Save as NPZ file
        try:
            np.savez_compressed(
                output_path,
                imgs=ct_slice,  # Shape: (slice_range, H, W)
                gts=seg_slice   # Shape: (slice_range, H, W)
            )
            samples_saved += 1
            print(f"Saved sample to {output_path}")
        except Exception as e:
            print(f"Error saving sample {sample_id}: {e}")
    
    return samples_saved

def visualize_alignment(ct_slice, seg_slice, output_path=None):
    """Visualize CT slice with segmentation overlay to check alignment"""
    plt.figure(figsize=(10, 8))
    
    # Show CT slice in grayscale
    plt.imshow(ct_slice, cmap='gray')
    
    # Create colored overlay for segmentation
    seg_overlay = np.zeros((*ct_slice.shape, 4))
    seg_overlay[seg_slice > 0] = [1, 0, 0, 0.5]  # Red with 50% opacity
    
    plt.imshow(seg_overlay)
    plt.title("CT with Segmentation Overlay")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def check_orientation(ct_np, seg_np, patient_id, output_dir="orientation_check"):
    """Test different orientations to find the correct alignment"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select a slice from the middle that contains segmentation
    for z in range(seg_np.shape[0]):
        if np.sum(seg_np[z]) > 0:
            ct_slice = ct_np[z]
            seg_slice = seg_np[z]
            
            # Original orientation
            visualize_alignment(ct_slice, seg_slice, 
                               os.path.join(output_dir, f"{patient_id}_original.png"))
            
            # Vertical flip (up-down)
            seg_vflip = np.flipud(seg_slice)
            visualize_alignment(ct_slice, seg_vflip,
                               os.path.join(output_dir, f"{patient_id}_vflip.png"))
            
            # Horizontal flip (left-right)
            seg_hflip = np.fliplr(seg_slice)
            visualize_alignment(ct_slice, seg_hflip,
                               os.path.join(output_dir, f"{patient_id}_hflip.png"))
            
            # Both flips
            seg_both = np.flipud(np.fliplr(seg_slice))
            visualize_alignment(ct_slice, seg_both,
                               os.path.join(output_dir, f"{patient_id}_both_flips.png"))
            
            print(f"Created orientation visualizations in {output_dir}")
            
            # Create a 2x2 grid for comparison
            fig, axs = plt.subplots(2, 2, figsize=(16, 16))
            titles = ['Original', 'Vertical Flip', 'Horizontal Flip', 'Both Flips']
            seg_variants = [seg_slice, seg_vflip, seg_hflip, seg_both]
            
            for i, (ax, title, seg) in enumerate(zip(axs.flatten(), titles, seg_variants)):
                ax.imshow(ct_slice, cmap='gray')
                
                # Create colored overlay
                seg_overlay = np.zeros((*ct_slice.shape, 4))
                seg_overlay[seg > 0] = [1, 0, 0, 0.5]  # Red with 50% opacity
                
                ax.imshow(seg_overlay)
                ax.set_title(title)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{patient_id}_comparison.png"))
            plt.close()
            
            break

def process_spinemetsCT_dataset(base_dir, output_dir, check_orientations=True):
    """Process the entire SpineMetsCT dataset using SimpleITK"""
    # Get all patient directories
    patient_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) 
                   if os.path.isdir(d) and not d.endswith("LICENSE")]
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    total_samples = 0
    
    for patient_dir in tqdm(patient_dirs[:5]):  # Limit to first 5 patients for testing
        patient_id = os.path.basename(patient_dir)
        print(f"\nProcessing patient: {patient_id}")
        
        # Find study directories
        study_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]
        
        for study_dir in study_dirs:
            study_name = os.path.basename(study_dir)
            print(f"  Study: {study_name}")
            
            # Find image and segmentation directories
            img_dirs = [d for d in glob.glob(os.path.join(study_dir, "*")) 
                       if os.path.isdir(d) and "Segmentation" not in d]
            seg_dirs = glob.glob(os.path.join(study_dir, "*Segmentation*"))
            
            if not img_dirs or not seg_dirs:
                print(f"  Skipping study {study_name}: Missing image or segmentation directories")
                continue
            
            # Process each image-segmentation pair
            for img_dir in img_dirs:
                img_name = os.path.basename(img_dir)
                print(f"    Processing image: {img_name}")
                
                try:
                    # Load CT volume
                    ct_img = load_ct_volume(img_dir)
                    print(f"    CT volume loaded: {ct_img.GetSize()}")
                    
                    # Normalize CT
                    ct_np = normalize_ct_volume(ct_img)
                    print(f"    CT volume normalized: {ct_np.shape}")
                    
                    # Process each segmentation
                    for seg_dir in seg_dirs:
                        seg_name = os.path.basename(seg_dir)
                        print(f"      Processing segmentation: {seg_name}")
                        
                        # Find segmentation DICOM files
                        seg_files = glob.glob(os.path.join(seg_dir, "*.dcm"))
                        if not seg_files:
                            print(f"      No segmentation files found in {seg_name}")
                            continue
                        
                        # For multi-frame segmentations, there is typically only one file
                        seg_file = seg_files[0]
                        print(f"      Using segmentation file: {os.path.basename(seg_file)}")
                        
                        try:
                            # Load and resample segmentation to match CT dimensions
                            seg_np = load_and_resample_seg(seg_file, ct_img)
                            print(f"      Segmentation loaded and resampled: {seg_np.shape}")
                            
                            # Apply orientation fixes - IMPORTANT
                            # Based on your image, we need to apply both vertical and horizontal flips
                            # to correctly align the segmentation mask with the CT volume
                            fixed_seg_np = np.zeros_like(seg_np)
                            for i in range(seg_np.shape[0]):
                                # Apply BOTH vertical and horizontal flips (adjust as needed)
                                fixed_seg_np[i] = np.flipud(np.fliplr(seg_np[i]))
                            
                            print(f"      Applied orientation correction to segmentation")
                            
                            # Check orientations if requested
                            if check_orientations:
                                check_orientation(ct_np, seg_np, f"{patient_id}_{study_name}")
                            
                            # Save as NPZ files
                            samples = save_as_npz(
                                patient_id, study_name, img_name, seg_name,
                                ct_np, fixed_seg_np, output_dir, slice_range=8
                            )
                            
                            total_samples += samples
                            print(f"      Saved {samples} samples for this segmentation")
                            
                        except Exception as e:
                            print(f"      Error processing segmentation {seg_name}: {e}")
                
                except Exception as e:
                    print(f"    Error processing image {img_name}: {e}")
    
    print(f"\nTotal samples saved: {total_samples}")
    return total_samples

def main():
    parser = argparse.ArgumentParser(description='Process SpineMetsCT dataset using SimpleITK')
    parser.add_argument('--input_dir', type=str, 
                        default='SpineMetsCT_Data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG',
                        help='Input directory containing the SpineMetsCT dataset')
    parser.add_argument('--output_dir', type=str, 
                        default='processed_data/SpineMetsCT_sitk_npz',
                        help='Output directory for NPZ files')
    parser.add_argument('--check_orientations', action='store_true',
                        help='Generate orientation test images')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the dataset
    process_spinemetsCT_dataset(args.input_dir, args.output_dir, args.check_orientations)

if __name__ == "__main__":
    main()
