#!/usr/bin/env python
# Script to inspect DICOM orientation and verify axial plane alignment

import os
import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import cv2

def get_dicom_orientation(directory):
    """Get orientation information from DICOM files"""
    dicom_files = sorted(glob.glob(os.path.join(directory, "*.dcm")))
    if not dicom_files:
        return None, None
    
    # Read a random sample of slices
    orientation_info = []
    patient_positions = []
    
    # Take up to 5 files to check
    sample_files = dicom_files[:min(5, len(dicom_files))]
    
    for file in sample_files:
        try:
            dcm = pydicom.dcmread(file)
            
            # Get orientation information
            if hasattr(dcm, 'ImageOrientationPatient'):
                orientation_info.append(dcm.ImageOrientationPatient)
            
            # Get patient position
            if hasattr(dcm, 'PatientPosition'):
                patient_positions.append(dcm.PatientPosition)
            
            # Print additional information that might be useful
            print(f"File: {os.path.basename(file)}")
            if hasattr(dcm, 'ImageOrientationPatient'):
                print(f"  ImageOrientationPatient: {dcm.ImageOrientationPatient}")
            if hasattr(dcm, 'PatientPosition'):
                print(f"  PatientPosition: {dcm.PatientPosition}")
            if hasattr(dcm, 'ImagePositionPatient'):
                print(f"  ImagePositionPatient: {dcm.ImagePositionPatient}")
            if hasattr(dcm, 'PixelSpacing'):
                print(f"  PixelSpacing: {dcm.PixelSpacing}")
            if hasattr(dcm, 'SliceThickness'):
                print(f"  SliceThickness: {dcm.SliceThickness}")
                
            # Check for axial plane
            if hasattr(dcm, 'ImageOrientationPatient'):
                orientation = dcm.ImageOrientationPatient
                # Calculate the cross product of the row and column vectors
                row_vector = orientation[:3]
                col_vector = orientation[3:]
                cross_product = np.cross(row_vector, col_vector)
                
                # For axial images, the cross product should be close to [0, 0, -1] or [0, 0, 1]
                if abs(cross_product[0]) < 0.1 and abs(cross_product[1]) < 0.1 and abs(abs(cross_product[2]) - 1) < 0.1:
                    print("  This appears to be an AXIAL image.")
                elif abs(cross_product[0]) < 0.1 and abs(abs(cross_product[1]) - 1) < 0.1 and abs(cross_product[2]) < 0.1:
                    print("  This appears to be a CORONAL image.")
                elif abs(abs(cross_product[0]) - 1) < 0.1 and abs(cross_product[1]) < 0.1 and abs(cross_product[2]) < 0.1:
                    print("  This appears to be a SAGITTAL image.")
                else:
                    print(f"  This appears to be an OBLIQUE image. Cross product: {cross_product}")
            
            print("")
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    return orientation_info, patient_positions

def visualize_with_multiple_orientations(npz_file):
    """Visualize a single NPZ file with different orientation corrections"""
    if not os.path.exists(npz_file):
        print(f"File {npz_file} not found")
        return
        
    try:
        data = np.load(npz_file)
        
        # Get images and masks
        if 'imgs' in data:
            images = data['imgs']
            img_key = 'imgs'
        elif 'img' in data:
            images = data['img']
            img_key = 'img'
        else:
            print(f"No image data found in {npz_file}")
            return
            
        if 'gts' in data:
            masks = data['gts']
            mask_key = 'gts'
        elif 'mask' in data:
            masks = data['mask']
            mask_key = 'mask'
        else:
            print(f"No mask data found in {npz_file}")
            return
        
        # Find a slice with significant mask content
        slice_areas = [np.sum(masks[i]) for i in range(masks.shape[0])]
        if sum(slice_areas) == 0:
            print("No mask content found")
            return
        
        best_slice_idx = np.argmax(slice_areas)
        
        # Create different versions of the mask
        mask_original = masks[best_slice_idx]
        mask_vertical_flip = np.flipud(mask_original)
        mask_horizontal_flip = np.fliplr(mask_original)
        mask_both_flips = np.fliplr(np.flipud(mask_original))
        mask_transpose = np.transpose(mask_original)
        mask_transpose_vertical = np.flipud(np.transpose(mask_original))
        
        # Set up plot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Normalize image for better visualization
        img = images[best_slice_idx]
        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99)
        
        # Original image
        axes[0, 0].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Original mask overlay
        axes[0, 1].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        mask_rgb = np.zeros((*mask_original.shape, 4))
        mask_rgb[mask_original > 0] = [1, 0, 0, 0.5]  # Red with alpha=0.5
        axes[0, 1].imshow(mask_rgb)
        axes[0, 1].set_title("Original Mask")
        axes[0, 1].axis('off')
        
        # Vertical flip
        axes[0, 2].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        mask_rgb = np.zeros((*mask_vertical_flip.shape, 4))
        mask_rgb[mask_vertical_flip > 0] = [0, 1, 0, 0.5]  # Green with alpha=0.5
        axes[0, 2].imshow(mask_rgb)
        axes[0, 2].set_title("Vertical Flip")
        axes[0, 2].axis('off')
        
        # Horizontal flip
        axes[0, 3].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        mask_rgb = np.zeros((*mask_horizontal_flip.shape, 4))
        mask_rgb[mask_horizontal_flip > 0] = [0, 0, 1, 0.5]  # Blue with alpha=0.5
        axes[0, 3].imshow(mask_rgb)
        axes[0, 3].set_title("Horizontal Flip")
        axes[0, 3].axis('off')
        
        # Both flips
        axes[1, 0].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        mask_rgb = np.zeros((*mask_both_flips.shape, 4))
        mask_rgb[mask_both_flips > 0] = [1, 1, 0, 0.5]  # Yellow with alpha=0.5
        axes[1, 0].imshow(mask_rgb)
        axes[1, 0].set_title("Vertical + Horizontal Flip")
        axes[1, 0].axis('off')
        
        # Transpose
        axes[1, 1].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        mask_rgb = np.zeros((*mask_transpose.shape, 4))
        mask_rgb[mask_transpose > 0] = [1, 0, 1, 0.5]  # Purple with alpha=0.5
        axes[1, 1].imshow(mask_rgb)
        axes[1, 1].set_title("Transpose")
        axes[1, 1].axis('off')
        
        # Transpose + Vertical flip
        axes[1, 2].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        mask_rgb = np.zeros((*mask_transpose_vertical.shape, 4))
        mask_rgb[mask_transpose_vertical > 0] = [0, 1, 1, 0.5]  # Cyan with alpha=0.5
        axes[1, 2].imshow(mask_rgb)
        axes[1, 2].set_title("Transpose + Vertical Flip")
        axes[1, 2].axis('off')
        
        # No content in last plot
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        output_path = f"orientation_test_{os.path.basename(npz_file).split('.')[0]}.png"
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
        plt.show()
        
    except Exception as e:
        print(f"Error processing {npz_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Inspect DICOM orientation and verify axial plane alignment')
    parser.add_argument('--dicom_dir', type=str, help='Directory containing DICOM files to check orientation')
    parser.add_argument('--npz_file', type=str, help='NPZ file to visualize with different orientations')
    
    args = parser.parse_args()
    
    if args.dicom_dir:
        print(f"Checking orientation for DICOM files in {args.dicom_dir}")
        get_dicom_orientation(args.dicom_dir)
    
    if args.npz_file:
        print(f"Visualizing orientations for {args.npz_file}")
        visualize_with_multiple_orientations(args.npz_file)
    
    if not args.dicom_dir and not args.npz_file:
        print("No action specified. Please provide --dicom_dir or --npz_file")

if __name__ == "__main__":
    main()
