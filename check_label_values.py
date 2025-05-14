import os
import glob
import numpy as np
import pydicom
from tqdm import tqdm

def read_segmentation_series(directory):
    dicom_files = sorted(glob.glob(os.path.join(directory, "*.dcm")))
    if not dicom_files:
        return None

    masks = []
    metadata = []
    
    for file in dicom_files:
        try:
            dcm = pydicom.dcmread(file)
            pixel_array = dcm.pixel_array
            
            # Also collect metadata
            metadata.append(dcm)
            
            if len(pixel_array.shape) > 2:
                print(f"the file is 3D")
                for i in range(pixel_array.shape[0]):
                    masks.append(pixel_array[i].astype(np.uint8))
            else:
                print(f"the file is 2D")
                masks.append(pixel_array.astype(np.uint8))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not masks:
        return None, None
    return np.stack(masks), metadata

def collect_unique_labels(root_dir):
    patient_dirs = [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
    label_set = set()
    
    # Also check for segmentation metadata
    segmentation_descriptions = {}
    segment_types = set()

    for patient_dir in tqdm(patient_dirs):
        for study_dir in glob.glob(os.path.join(patient_dir, "*")):
            if not os.path.isdir(study_dir): continue
            seg_dirs = glob.glob(os.path.join(study_dir, "*Segmentation*"))
            for seg_dir in seg_dirs:
                masks, metadata = read_segmentation_series(seg_dir)
                if masks is not None:
                    unique = np.unique(masks)
                    label_set.update(unique)
                    
                    # Check metadata for segmentation information
                    if metadata:
                        for dcm in metadata:
                            if hasattr(dcm, 'SegmentSequence'):
                                for i, segment in enumerate(dcm.SegmentSequence):
                                    segment_number = segment.SegmentNumber
                                    segment_desc = getattr(segment, 'SegmentDescription', 'Unknown')
                                    segment_types.add(segment_desc)
                                    segmentation_descriptions[segment_number] = segment_desc

    return sorted(label_set), segmentation_descriptions, segment_types

def check_label_values():
    """Check the unique pixel values in the DICOM SEG files"""
    dataset_root = "SpineMetsCT_Data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG"
    
    # Find all DICOM SEG files
    seg_files = []
    for root, _, files in os.walk(dataset_root):
        if "Spine Segmentation" in root:
            for file in files:
                if file.endswith('.dcm'):
                    seg_files.append(os.path.join(root, file))
    
    print(f"Found {len(seg_files)} DICOM SEG files")
    
    # Also check if there are .npz files in the processed directory
    processed_dir = "/tmp2/b10902078/MEDSAM/processed_data/SpineMetsCT_npz"
    npz_files = []
    if os.path.exists(processed_dir):
        for root, _, files in os.walk(processed_dir):
            for file in files:
                if file.endswith('.npz'):
                    npz_files.append(os.path.join(root, file))
        
        print(f"Found {len(npz_files)} NPZ files in processed directory")
    
    # Analyze unique values in DICOM SEG files
    dicom_values = {}
    
    for file_path in tqdm(seg_files[:10], desc="Analyzing DICOM SEG files"):  # Check first 10 for speed
        try:
            dcm = pydicom.dcmread(file_path)
            
            # Get pixel data
            if hasattr(dcm, 'PixelData'):
                # Extract pixel data and check unique values
                try:
                    pixel_array = dcm.pixel_array
                    unique_vals = np.unique(pixel_array)
                    
                    patient_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
                    
                    dicom_values[file_path] = {
                        'patient_id': patient_id,
                        'unique_values': unique_vals.tolist(),
                        'shape': pixel_array.shape,
                        'min': float(np.min(pixel_array)),
                        'max': float(np.max(pixel_array))
                    }
                except Exception as e:
                    print(f"Error extracting pixel data from {file_path}: {e}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Analyze unique values in NPZ files
    npz_values = {}
    
    for file_path in tqdm(npz_files[:10], desc="Analyzing NPZ files"):  # Check first 10 for speed
        try:
            data = np.load(file_path)
            
            # Check if 'label' key exists (for segmentation masks)
            if 'label' in data:
                label = data['label']
                unique_vals = np.unique(label)
                
                npz_values[file_path] = {
                    'unique_values': unique_vals.tolist(),
                    'shape': label.shape,
                    'min': float(np.min(label)),
                    'max': float(np.max(label))
                }
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Print findings
    print("\n===== DICOM SEG FILES =====")
    if dicom_values:
        print(f"Analyzed {len(dicom_values)} DICOM SEG files")
        
        # Summarize unique values across all files
        all_unique_vals = set()
        for info in dicom_values.values():
            all_unique_vals.update(info['unique_values'])
        
        print(f"All unique pixel values found: {sorted(all_unique_vals)}")
        
        # Print details for each file
        for file_path, info in dicom_values.items():
            print(f"\nFile: {os.path.basename(file_path)}")
            print(f"  Patient ID: {info['patient_id']}")
            print(f"  Shape: {info['shape']}")
            print(f"  Unique values: {info['unique_values']}")
            print(f"  Min: {info['min']}, Max: {info['max']}")
    else:
        print("No DICOM SEG files analyzed successfully.")
    
    # Print NPZ findings
    if npz_values:
        print("\n===== NPZ FILES =====")
        print(f"Analyzed {len(npz_values)} NPZ files")
        
        # Summarize unique values across all files
        all_unique_vals = set()
        for info in npz_values.values():
            all_unique_vals.update(info['unique_values'])
        
        print(f"All unique pixel values found: {sorted(all_unique_vals)}")
        
        # Print details for each file
        for file_path, info in npz_values.items():
            print(f"\nFile: {os.path.basename(file_path)}")
            print(f"  Shape: {info['shape']}")
            print(f"  Unique values: {info['unique_values']}")
            print(f"  Min: {info['min']}, Max: {info['max']}")
    
    # Conclusion
    if len(all_unique_vals) > 2:
        print("\n===== CONCLUSION =====")
        print(f"Found {len(all_unique_vals)} unique values in the segmentation files: {sorted(all_unique_vals)}")
        print("This suggests the possibility that different values might represent different lesion types:")
        print("- Value 0: Background")
        for i, val in enumerate(sorted(all_unique_vals - {0}), 1):
            lesion_types = ["Normal", "Osteolytic", "Osteosclerotic", "Mixed"]
            if i <= len(lesion_types):
                print(f"- Value {val}: Potentially {lesion_types[i-1]} lesions")
            else:
                print(f"- Value {val}: Unknown classification")
        print("\nRecommendation: Contact the dataset authors to confirm this interpretation.")
    else:
        print("\n===== CONCLUSION =====")
        print(f"Found only {len(all_unique_vals)} unique values in the segmentation files: {sorted(all_unique_vals)}")
        print("This suggests the segmentation is binary (0=background, 1=lesion) with no embedded lesion type classification.")
        print("\nPossible explanations:")
        print("1. The lesion classification is not included in this dataset")
        print("2. The classification was meant to be in a separate file not provided")
        print("3. The dataset description on the website might be inaccurate")
        print("\nRecommendation: Contact the dataset authors via https://doi.org/10.7937/kh36-ds04 for clarification")

if __name__ == "__main__":
    dataset_root = "SpineMetsCT_Data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG"
    labels, descriptions, segment_types = collect_unique_labels(dataset_root)
    print("Unique labels found in segmentation masks:", labels)
    
    print("\nSegmentation metadata:")
    if descriptions:
        for segment_num, desc in descriptions.items():
            print(f"Segment {segment_num}: {desc}")
    else:
        print("No segmentation descriptions found in DICOM metadata")
    
    print("\nUnique segment types found in metadata:", list(segment_types))
    
    # Look for specific files to examine in detail
    print("\nLooking for a sample segmentation file to examine metadata...")
    sample_file = None
    for path, _, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.dcm') and 'Segmentation' in path:
                sample_file = os.path.join(path, file)
                break
        if sample_file:
            break
            
    if sample_file:
        print(f"Examining sample file: {sample_file}")
        try:
            dcm = pydicom.dcmread(sample_file)
            # Print some specific metadata that might contain lesion type information
            print("\nSelected DICOM metadata fields:")
            for attr in ['SeriesDescription', 'StudyDescription', 'ImageType']:
                if hasattr(dcm, attr):
                    print(f"{attr}: {getattr(dcm, attr)}")
            
            # Print the first few metadata elements to find where lesion type might be stored
            print("\nFirst 20 metadata elements:")
            for i, elem in enumerate(dcm):
                if i >= 20: break
                print(f"{elem.tag}: {elem.name} = {elem.value}")
        except Exception as e:
            print(f"Error examining sample file: {e}")
    check_label_values()
