#!/usr/bin/env python
# Script to test and fix mask orientation issues in preprocessed data

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

def visualize_sample(data, correction_type=None):
    """Visualize a sample with original and potentially corrected masks"""
    # Get data
    if 'imgs' in data:
        images = data['imgs']
    elif 'img' in data:
        images = data['img']
    else:
        print("No image data found in keys:", list(data.keys()))
        return

    if 'gts' in data:
        masks = data['gts']
    elif 'mask' in data:
        masks = data['mask']
    else:
        print("No mask data found in keys:", list(data.keys()))
        return
    
    # Find a slice with significant mask content
    slice_areas = [np.sum(masks[i]) for i in range(masks.shape[0])]
    if sum(slice_areas) == 0:
        print("No mask content found")
        return
    
    best_slice_idx = np.argmax(slice_areas)
    
    # Create corrected masks based on correction type
    corrected_masks = np.copy(masks)
    
    if correction_type == 'flip_vertical':
        for i in range(corrected_masks.shape[0]):
            corrected_masks[i] = np.flipud(corrected_masks[i])
        title = "Vertical Flip Correction"
    elif correction_type == 'flip_horizontal':
        for i in range(corrected_masks.shape[0]):
            corrected_masks[i] = np.fliplr(corrected_masks[i])
        title = "Horizontal Flip Correction"
    elif correction_type == 'transpose':
        for i in range(corrected_masks.shape[0]):
            corrected_masks[i] = np.transpose(corrected_masks[i])
        title = "Transpose Correction"
    else:
        title = "No Correction"
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Normalize for better visualization
    vmin = np.percentile(images, 1)
    vmax = np.percentile(images, 99)
    
    # Original image
    axes[0].imshow(images[best_slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Original Image (Slice {best_slice_idx})")
    axes[0].axis('off')
    
    # Original mask overlay
    axes[1].imshow(images[best_slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
    mask_rgb = np.zeros((*masks[best_slice_idx].shape, 4))
    mask_rgb[masks[best_slice_idx] > 0] = [1, 0, 0, 0.5]  # Red with alpha=0.5
    axes[1].imshow(mask_rgb)
    axes[1].set_title("Original Mask Overlay")
    axes[1].axis('off')
    
    # Corrected mask overlay
    axes[2].imshow(images[best_slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
    mask_rgb = np.zeros((*corrected_masks[best_slice_idx].shape, 4))
    mask_rgb[corrected_masks[best_slice_idx] > 0] = [0, 0, 1, 0.5]  # Blue with alpha=0.5
    axes[2].imshow(mask_rgb)
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"visualization_{title.replace(' ', '_')}.png")
    plt.show()
    
    return images, masks, corrected_masks, best_slice_idx

def process_and_fix_npz(file_path, output_dir, correction_type='flip_vertical', dry_run=True):
    """Process an NPZ file and potentially fix the orientation issue"""
    try:
        # Load the data
        data = np.load(file_path)
        
        # Get images and masks
        if 'imgs' in data:
            images = data['imgs']
            img_key = 'imgs'
        elif 'img' in data:
            images = data['img']
            img_key = 'img'
        else:
            print(f"No image data found in {file_path}")
            return False
            
        if 'gts' in data:
            masks = data['gts']
            mask_key = 'gts'
        elif 'mask' in data:
            masks = data['mask']
            mask_key = 'mask'
        else:
            print(f"No mask data found in {file_path}")
            return False
        
        # Apply correction
        corrected_masks = np.copy(masks)
        
        if correction_type == 'flip_vertical':
            for i in range(corrected_masks.shape[0]):
                corrected_masks[i] = np.flipud(corrected_masks[i])
        elif correction_type == 'flip_horizontal':
            for i in range(corrected_masks.shape[0]):
                corrected_masks[i] = np.fliplr(corrected_masks[i])
        elif correction_type == 'transpose':
            for i in range(corrected_masks.shape[0]):
                corrected_masks[i] = np.transpose(corrected_masks[i])
                
        # Create new filename
        base_filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, base_filename)
        
        # Save the file with corrected masks
        if not dry_run:
            os.makedirs(output_dir, exist_ok=True)
            # Create dict of arrays to save
            save_dict = {}
            for key in data:
                if key == mask_key:
                    save_dict[key] = corrected_masks
                else:
                    save_dict[key] = data[key]
            
            np.savez_compressed(output_path, **save_dict)
            print(f"Saved corrected file to {output_path}")
        else:
            print(f"Would save corrected file to {output_path} (dry run)")
            
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test and fix mask orientation issues in NPZ files')
    parser.add_argument('--input_dir', type=str, default='processed_data/SpineMetsCT_npz',
                        help='Input directory containing NPZ files')
    parser.add_argument('--output_dir', type=str, default='processed_data/SpineMetsCT_npz_fixed',
                        help='Output directory for fixed NPZ files')
    parser.add_argument('--correction', type=str, default='flip_vertical',
                        choices=['flip_vertical', 'flip_horizontal', 'transpose'],
                        help='Type of correction to apply')
    parser.add_argument('--dry_run', action='store_true',
                        help='Run without saving any files')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Only visualize samples without fixing')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Find all NPZ files
    npz_files = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.input_dir, split)
        if os.path.exists(split_dir):
            npz_files.extend(glob.glob(os.path.join(split_dir, '*.npz')))
    
    # If no files found in split directories, search the input directory directly
    if not npz_files:
        npz_files = glob.glob(os.path.join(args.input_dir, '**/*.npz'), recursive=True)
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # Only visualize if requested
    if args.visualize_only:
        # Select random samples to visualize
        if len(npz_files) > 0:
            sample_files = random.sample(npz_files, min(args.num_samples, len(npz_files)))
            for file in sample_files:
                print(f"\nVisualizing {os.path.basename(file)}")
                data = np.load(file)
                visualize_sample(data, args.correction)
        return
    
    # # Process files
    # output_dir = args.output_dir
    # processed_count = 0
    
    # for file in npz_files:
    #     # Determine the output directory structure to maintain the same organization
    #     rel_path = os.path.relpath(file, args.input_dir)
    #     output_subdir = os.path.dirname(rel_path)
    #     full_output_dir = os.path.join(output_dir, output_subdir)
        
    #     # Process the file
    #     success = process_and_fix_npz(file, full_output_dir, args.correction, args.dry_run)
    #     if success:
    #         processed_count += 1
    
    # print(f"Processed {processed_count} of {len(npz_files)} files")
    
    # # Visualize some corrected files
    # if not args.dry_run and processed_count > 0:
    #     print("\nVisualizing some corrected files:")
    #     corrected_files = glob.glob(os.path.join(output_dir, '**/*.npz'), recursive=True)
    #     if len(corrected_files) > 0:
    #         sample_files = random.sample(corrected_files, min(3, len(corrected_files)))
    #         for file in sample_files:
    #             print(f"\nVisualizing corrected file: {os.path.basename(file)}")
    #             data = np.load(file)
    #             visualize_sample(data)

if __name__ == "__main__":
    main()
