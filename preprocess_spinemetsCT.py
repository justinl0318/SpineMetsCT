#!/usr/bin/env python
# Script to convert SpineMetsCT DICOM data to NPZ format for MedSAM2 fine-tuning

import os
import glob
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
from tqdm import tqdm
import argparse
import random
import multiprocessing
from functools import partial
import gc

def read_dicom_series(directory):
    """Read all DICOM files in the directory and return as a 3D volume"""
    dicom_files = sorted(glob.glob(os.path.join(directory, "*.dcm")))
    if not dicom_files:
        return None

    # Read the first slice to get metadata
    first_slice = pydicom.dcmread(dicom_files[0])
    
    # Read all slices
    slices = []
    for file in dicom_files:
        try:
            slice = pydicom.dcmread(file)
            slices.append(slice)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Sort slices by slice location
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else 0)
    
    # Extract pixel arrays from each slice
    vol_data = []
    for slice in slices:
        try:
            # Convert to Hounsfield Units if modality is CT
            if hasattr(slice, 'Modality') and slice.Modality == 'CT':
                pixel_array = slice.pixel_array.astype(np.int16)
                # Convert to Hounsfield Units (HU)
                intercept = slice.RescaleIntercept if hasattr(slice, 'RescaleIntercept') else 0
                slope = slice.RescaleSlope if hasattr(slice, 'RescaleSlope') else 1
                pixel_array = pixel_array * slope + intercept
            else:
                # For other modalities, use VOI LUT transformation
                pixel_array = apply_voi_lut(slice.pixel_array, slice)
            
            vol_data.append(pixel_array)
        except Exception as e:
            print(f"Error processing slice: {e}")
    
    if not vol_data:
        return None
    
    # Stack slices to form a 3D volume
    volume = np.stack(vol_data)
    return volume

def read_segmentation_series(directory):
    """Read segmentation DICOM files and return as a 3D binary mask"""
    dicom_files = sorted(glob.glob(os.path.join(directory, "*.dcm")))
    if not dicom_files:
        return None
    
    # Read all slices
    slices = []
    for file in dicom_files:
        try:
            slice = pydicom.dcmread(file)
            slices.append(slice)
        except Exception as e:
            print(f"Error reading segmentation {file}: {e}")
    
    # Sort slices by slice location if available
    if slices and hasattr(slices[0], 'ImagePositionPatient'):
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Extract segmentation masks
    mask_data = []
    for slice in slices:
        try:
            # For segmentation, we just need binary masks
            # Handle the case where pixel array might have extra dimensions
            pixel_array = slice.pixel_array
            # Remove extra dimensions if present (like for multi-frame or 3D objects)
            if len(pixel_array.shape) > 2:
                print(f"      Found multi-dimensional mask with shape {pixel_array.shape}, flattening")
                # Take all frames if it's a multi-frame image
                for i in range(pixel_array.shape[0]):
                    mask = pixel_array[i].astype(bool).astype(np.uint8)
                    mask_data.append(mask)
            else:
                mask = pixel_array.astype(bool).astype(np.uint8)
                mask_data.append(mask)
        except Exception as e:
            print(f"Error processing segmentation slice: {e}")
    
    if not mask_data:
        return None
    
    # Stack slices to form a 3D segmentation mask
    mask_volume = np.stack(mask_data)
    return mask_volume

def normalize_ct(volume, window_center=-600, window_width=1500):
    """Apply windowing to CT volumes and normalize to 0-255"""
    # Default windowing for lung visualization
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    
    # Apply windowing
    volume = np.clip(volume, min_value, max_value)
    
    # Normalize to 0-255
    volume = ((volume - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    
    return volume

def process_patient_data(patient_dir, output_dir, slice_range=8):
    """Process a single patient directory and convert to NPZ format"""
    patient_id = os.path.basename(patient_dir)
    print(f"Processing patient: {patient_id}")
    study_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]
    print(f"Found {len(study_dirs)} study directories")
    
    total_samples_saved = 0
    
    for study_dir in study_dirs:
        study_name = os.path.basename(study_dir)
        seg_dirs = glob.glob(os.path.join(study_dir, "*Segmentation*"))
        img_dirs = [d for d in glob.glob(os.path.join(study_dir, "*")) 
                    if os.path.isdir(d) and "Segmentation" not in d]
        
        print(f"  Study: {study_name}, Found {len(img_dirs)} image dirs and {len(seg_dirs)} segmentation dirs")
        
        # Check if we have both image and segmentation data
        if not seg_dirs or not img_dirs:
            print(f"  Skipping study {study_name}: Missing image or segmentation data")
            continue
        
        # Process each image-segmentation pair
        for img_dir in img_dirs:
            img_name = os.path.basename(img_dir)
            print(f"    Processing image: {img_name}")
            
            # Read the CT volume
            volume = read_dicom_series(img_dir)
            if volume is None:
                print(f"    Skipping image {img_name}: Failed to read DICOM series")
                continue
            print(f"    Read volume with shape: {volume.shape}")
            
            # Process each segmentation for this volume
            for seg_dir in seg_dirs:
                seg_name = os.path.basename(seg_dir)
                print(f"      Processing segmentation: {seg_name}")
                
                # Read the segmentation masks
                masks = read_segmentation_series(seg_dir)
                if masks is None:
                    print(f"      Skipping segmentation {seg_name}: Failed to read masks")
                    continue
                print(f"      Read masks with shape: {masks.shape}")
                
                # Fix the mask orientation by flipping vertically
                # This corrects the up-down orientation issue in the segmentation masks
                fixed_masks = np.zeros_like(masks)
                for i in range(masks.shape[0]):
                    fixed_masks[i] = np.flipud(masks[i])  # Flip vertically (up-down)
                masks = fixed_masks
                print(f"      Applied vertical flip to fix mask orientation")
                
                # Normalize the CT volume
                norm_volume = normalize_ct(volume)
                
                # Ensure the volume and masks have the same dimensions
                if volume.shape != masks.shape:
                    print(f"      Shape mismatch: volume {volume.shape}, masks {masks.shape}")
                    
                    # Handle the case where the number of slices differs
                    vol_slices = volume.shape[0]
                    mask_slices = masks.shape[0]
                    
                    # Resample the masks to match the volume dimensions
                    if vol_slices != mask_slices:
                        print(f"      Resampling masks from {mask_slices} slices to {vol_slices} slices")
                        
                        # Create empty mask volume with the target size
                        resampled_masks = np.zeros((vol_slices, volume.shape[1], volume.shape[2]), dtype=np.uint8)
                        
                        # Calculate scaling factor for z-axis
                        scale_z = mask_slices / vol_slices
                        
                        # Resample each slice of the target volume
                        for i in range(vol_slices):
                            # Find the corresponding slice in the mask (with interpolation)
                            mask_idx = min(int(i * scale_z), mask_slices - 1)
                            
                            # Resize in x-y plane if needed
                            if masks.shape[1] != volume.shape[1] or masks.shape[2] != volume.shape[2]:
                                resampled_masks[i] = cv2.resize(
                                    masks[mask_idx],
                                    (volume.shape[2], volume.shape[1]),
                                    interpolation=cv2.INTER_NEAREST
                                )
                            else:
                                resampled_masks[i] = masks[mask_idx]
                        
                        masks = resampled_masks
                        print(f"      Resampled masks shape: {masks.shape}")
                    else:
                        # Only resize in x-y plane
                        resized_masks = np.zeros_like(volume, dtype=np.uint8)
                        for i in range(len(masks)):
                            resized_masks[i] = cv2.resize(
                                masks[i],
                                (volume.shape[2], volume.shape[1]),
                                interpolation=cv2.INTER_NEAREST
                            )
                        masks = resized_masks
                        print(f"      Resized masks to match volume: {masks.shape}")
                
                # Create sliding windows of slices (3D to pseudo-3D conversion)
                os.makedirs(output_dir, exist_ok=True)
                
                samples_saved = 0
                # Generate samples with sliding window
                for i in range(0, len(volume) - slice_range + 1, slice_range // 2):  # 50% overlap
                    # Get a range of slices
                    vol_slice = norm_volume[i:i+slice_range]
                    mask_slice = masks[i:i+slice_range]
                    
                    # Skip if no segmentation in this slice range
                    if np.sum(mask_slice) == 0:
                        continue
                    
                    # Create a unique ID for this sample
                    sample_id = f"{patient_id}_{study_name}_{img_name}_{seg_name}_{i}"
                    output_path = os.path.join(output_dir, f"{sample_id}.npz")
                    
                    # Save as NPZ file
                    try:
                        np.savez_compressed(
                            output_path,
                            imgs=vol_slice,  # Shape: (slice_range, H, W)
                            gts=mask_slice    # Shape: (slice_range, H, W)
                        )
                        samples_saved += 1
                        print(f"      Saved sample to {output_path}")
                    except Exception as e:
                        print(f"      Error saving sample {sample_id}: {e}")
                
                print(f"      Saved {samples_saved} samples for this segmentation")
                gc.collect()
                total_samples_saved += samples_saved
    
    return total_samples_saved

# New function to handle processing in parallel
def process_patient_batch(patient_dirs, output_dir, slice_range):
    """Process a batch of patient directories in parallel"""
    total_samples = 0
    for patient_dir in patient_dirs:
        try:
            samples = process_patient_data(patient_dir, output_dir, slice_range)
            total_samples += samples
        except Exception as e:
            print(f"Error processing {patient_dir}: {e}")
    return total_samples

def main():
    parser = argparse.ArgumentParser(description='Convert SpineMetsCT DICOM data to NPZ format for MedSAM2')
    parser.add_argument('--input_dir', type=str, default='SpineMetsCT_Data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG',
                        help='Input directory containing the SpineMetsCT dataset')
    parser.add_argument('--output_dir', type=str, default='processed_data/SpineMetsCT_npz',
                        help='Output directory for NPZ files')
    parser.add_argument('--slice_range', type=int, default=8,
                        help='Number of consecutive slices to include in each NPZ file')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers for processing (default: number of CPU cores)')
    
    args = parser.parse_args()
    
    # Create output directories
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all patient directories
    patient_dirs = [d for d in glob.glob(os.path.join(args.input_dir, "*")) 
                   if os.path.isdir(d) and not d.endswith("LICENSE")]
    
    # Shuffle and split into train/val sets (80/20 split)
    random.seed(42)  # For reproducibility
    random.shuffle(patient_dirs)
    split_idx = int(len(patient_dirs) * 0.8)
    train_patients = patient_dirs[:split_idx]
    val_patients = patient_dirs[split_idx:]
    
    print(f"Processing {len(train_patients)} patients for training set using {args.num_workers} workers...")
    
    # Set up multiprocessing
    if args.num_workers <= 1:
        # Process sequentially if only one worker
        train_samples = 0
        for patient_dir in tqdm(train_patients):
            train_samples += process_patient_data(patient_dir, train_dir, args.slice_range)
    else:
        # Process in parallel
        # Split patients into batches for each worker
        batch_size = max(1, len(train_patients) // args.num_workers)
        train_batches = [train_patients[i:i+batch_size] for i in range(0, len(train_patients), batch_size)]
        
        # Create a processing pool
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            # Process each batch in parallel
            args_list = [(batch, train_dir, args.slice_range) for batch in train_batches]
            train_results = pool.starmap(process_patient_batch, args_list)
            train_samples = sum(train_results)
    
    print(f"Processed {train_samples} training samples")
    
    print(f"Processing {len(val_patients)} patients for validation set using {args.num_workers} workers...")
    
    # Process validation set with the same approach
    if args.num_workers <= 1:
        # Process sequentially if only one worker
        val_samples = 0
        for patient_dir in tqdm(val_patients):
            val_samples += process_patient_data(patient_dir, val_dir, args.slice_range)
    else:
        # Process in parallel
        batch_size = max(1, len(val_patients) // args.num_workers)
        val_batches = [val_patients[i:i+batch_size] for i in range(0, len(val_patients), batch_size)]
        
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            val_results = pool.map(
                partial(process_patient_batch, output_dir=val_dir, slice_range=args.slice_range),
                val_batches
            )
            val_samples = sum(val_results)
    
    print(f"Processed {val_samples} validation samples")
    print(f"Total samples: {train_samples + val_samples}")
    print("Done!")

if __name__ == "__main__":
    main()