import os
import glob
import pydicom
from tqdm import tqdm

def deep_inspect_seg_file(file_path):
    """Thoroughly examine a DICOM SEG file for any lesion type information"""
    try:
        dcm = pydicom.dcmread(file_path)
        
        # Dictionary to store findings
        info = {
            'file': os.path.basename(file_path),
            'lesion_types': [],
            'segment_descriptions': [],
            'category_descriptions': []
        }
        
        # Check for ContentSequence which can contain classification codes
        if hasattr(dcm, 'ContentSequence'):
            for item in dcm.ContentSequence:
                if hasattr(item, 'ConceptNameCodeSequence'):
                    for code in item.ConceptNameCodeSequence:
                        if hasattr(code, 'CodeMeaning'):
                            info['category_descriptions'].append(code.CodeMeaning)
                
                # Look for nested content items which might contain category info
                if hasattr(item, 'ContentSequence'):
                    for subitem in item.ContentSequence:
                        if hasattr(subitem, 'ConceptNameCodeSequence'):
                            for code in subitem.ConceptNameCodeSequence:
                                if hasattr(code, 'CodeMeaning'):
                                    info['category_descriptions'].append(code.CodeMeaning)
                                    
                                    # Check for specific terms related to lesion types
                                    code_meaning = code.CodeMeaning.lower()
                                    if any(term in code_meaning for term in ['lytic', 'osteolytic', 'blastic', 'osteosclerotic', 'mixed']):
                                        info['lesion_types'].append(f"Found in ContentSequence: {code.CodeMeaning}")
                
        # Check for SegmentSequence which should contain segment info
        if hasattr(dcm, 'SegmentSequence'):
            for segment in dcm.SegmentSequence:
                # Get segment number
                segment_number = getattr(segment, 'SegmentNumber', 'Unknown')
                
                # Look for the segment description
                segment_desc = getattr(segment, 'SegmentDescription', 'Unknown')
                info['segment_descriptions'].append(f"Segment {segment_number}: {segment_desc}")
                
                # Check for lesion classification in description
                desc_lower = segment_desc.lower()
                if any(lesion_type in desc_lower for lesion_type in ['lytic', 'osteolytic', 'blastic', 'osteosclerotic', 'mixed', 'normal']):
                    # Found a lesion type!
                    for lesion_type in ['lytic', 'osteolytic', 'blastic', 'osteosclerotic', 'mixed', 'normal']:
                        if lesion_type in desc_lower:
                            info['lesion_types'].append(
                                f"Segment {segment_number}: {lesion_type.capitalize()}"
                            )
                
                # Check for SegmentedPropertyCategoryCodeSequence which can specify lesion type
                if hasattr(segment, 'SegmentedPropertyCategoryCodeSequence'):
                    for code_seq in segment.SegmentedPropertyCategoryCodeSequence:
                        code_meaning = getattr(code_seq, 'CodeMeaning', 'Unknown')
                        info['category_descriptions'].append(
                            f"Segment {segment_number} Category: {code_meaning}"
                        )
                        
                        # Check if code meaning contains lesion type
                        meaning_lower = code_meaning.lower()
                        if any(term in meaning_lower for term in ['lytic', 'osteolytic', 'blastic', 'osteosclerotic', 'mixed']):
                            info['lesion_types'].append(
                                f"Segment {segment_number} Category: {code_meaning}"
                            )
                
                # Check for SegmentedPropertyTypeCodeSequence which can specify lesion type
                if hasattr(segment, 'SegmentedPropertyTypeCodeSequence'):
                    for code_seq in segment.SegmentedPropertyTypeCodeSequence:
                        code_meaning = getattr(code_seq, 'CodeMeaning', 'Unknown')
                        info['category_descriptions'].append(
                            f"Segment {segment_number} Type: {code_meaning}"
                        )
                        
                        # Check if code meaning contains lesion type
                        meaning_lower = code_meaning.lower()
                        if any(term in meaning_lower for term in ['lytic', 'osteolytic', 'blastic', 'osteosclerotic', 'mixed']):
                            info['lesion_types'].append(
                                f"Segment {segment_number} Type: {code_meaning}"
                            )
        
        # Full dump of private tags
        info['private_tags'] = []
        for elem in dcm:
            if elem.tag.is_private:
                try:
                    # Try to get the value as a string
                    value = str(elem.value)
                    # Check if the value contains lesion type keywords
                    if any(term in value.lower() for term in ['lytic', 'osteolytic', 'blastic', 'osteosclerotic', 'mixed']):
                        info['lesion_types'].append(f"Found in private tag {elem.tag}: {value}")
                    info['private_tags'].append(f"{elem.tag}: {elem.name} = {value[:100]}...")
                except:
                    info['private_tags'].append(f"{elem.tag}: {elem.name} = [Unable to display value]")
        
        # Check for ReferencedSeriesSequence which might contain lesion info
        if hasattr(dcm, 'ReferencedSeriesSequence'):
            for ref_series in dcm.ReferencedSeriesSequence:
                if hasattr(ref_series, 'SeriesDescription'):
                    desc = ref_series.SeriesDescription
                    if any(term in desc.lower() for term in ['lytic', 'osteolytic', 'blastic', 'osteosclerotic', 'mixed']):
                        info['lesion_types'].append(f"From ReferencedSeriesSequence: {desc}")
                        
        # Also check SeriesDescription directly
        if hasattr(dcm, 'SeriesDescription'):
            desc = dcm.SeriesDescription
            if any(term in desc.lower() for term in ['lytic', 'osteolytic', 'blastic', 'osteosclerotic', 'mixed']):
                info['lesion_types'].append(f"From SeriesDescription: {desc}")
        
        return info
    
    except Exception as e:
        return {'file': os.path.basename(file_path), 'error': str(e)}

def find_lesion_classifications():
    dataset_root = "SpineMetsCT_Data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG"
    
    # Find all DICOM SEG files
    seg_files = []
    for root, _, files in os.walk(dataset_root):
        if "Spine Segmentation" in root:
            for file in files:
                if file.endswith('.dcm'):
                    seg_files.append(os.path.join(root, file))
    
    print(f"Found {len(seg_files)} DICOM SEG files for examination")
    
    # Also check non-segmentation DICOM files which might contain lesion info
    # This can capture CT series descriptions that might have lesion info
    ct_files = []
    for root, _, files in os.walk(dataset_root):
        if "Spine Segmentation" not in root:  # Only non-segmentation folders
            for file in files:
                if file.endswith('.dcm'):
                    ct_files.append(os.path.join(root, file))
                    break  # Just get the first DICOM file from each series
    
    print(f"Found {len(ct_files)} DICOM CT series for examination")
    
    # Examine each SEG file for lesion type information
    findings = []
    for file_path in tqdm(seg_files, desc="Examining SEG files"):
        info = deep_inspect_seg_file(file_path)
        findings.append(info)
    
    # Also examine CT files as they might contain lesion type info in series descriptions
    for file_path in tqdm(ct_files, desc="Examining CT files"):
        try:
            dcm = pydicom.dcmread(file_path)
            info = {'file': os.path.basename(file_path), 'lesion_types': []}
            
            # Check SeriesDescription
            if hasattr(dcm, 'SeriesDescription'):
                desc = dcm.SeriesDescription
                if any(term in desc.lower() for term in ['lytic', 'osteolytic', 'blastic', 'osteosclerotic', 'mixed']):
                    info['lesion_types'].append(f"From SeriesDescription: {desc}")
            
            # Check StudyDescription
            if hasattr(dcm, 'StudyDescription'):
                desc = dcm.StudyDescription
                if any(term in desc.lower() for term in ['lytic', 'osteolytic', 'blastic', 'osteosclerotic', 'mixed']):
                    info['lesion_types'].append(f"From StudyDescription: {desc}")
            
            if info['lesion_types']:
                findings.append(info)
        except:
            # Skip files that can't be read
            pass
    
    # Report findings
    print("\n===== FINDINGS =====")
    
    lesion_types_found = [f for f in findings if f.get('lesion_types')]
    if lesion_types_found:
        print(f"\nFound explicit lesion types in {len(lesion_types_found)} files:")
        for finding in lesion_types_found:
            print(f"File: {finding['file']}")
            for lesion_type in finding['lesion_types']:
                print(f"  {lesion_type}")
    else:
        print("\nNo explicit lesion type classifications (Lytic/Blastic/Mixed) found in file descriptions")
    
    # Report segment descriptions
    all_segment_descriptions = set()
    for finding in findings:
        for desc in finding.get('segment_descriptions', []):
            all_segment_descriptions.add(desc)
    
    if all_segment_descriptions:
        print(f"\nFound {len(all_segment_descriptions)} unique segment descriptions:")
        for desc in sorted(all_segment_descriptions):
            print(f"  {desc}")
    
    # Report category descriptions
    all_category_descriptions = set()
    for finding in findings:
        for desc in finding.get('category_descriptions', []):
            all_category_descriptions.add(desc)
    
    if all_category_descriptions:
        print(f"\nFound {len(all_category_descriptions)} unique category descriptions:")
        for desc in sorted(all_category_descriptions):
            print(f"  {desc}")
    
    # Check for structured content (which might contain lesion info)
    structured_content_files = [f for f in findings if f.get('structured_content')]
    if structured_content_files:
        print("\nFound structured content which might contain lesion information:")
        for finding in structured_content_files:
            print(f"File: {finding['file']}")
            print(f"  {finding['structured_content']}")
    
    # Check the first 5 files for private tags
    private_tag_files = [f for f in findings[:5] if f.get('private_tags')]
    if private_tag_files:
        print("\nFound private tags in DICOM files which might contain lesion information:")
        for finding in private_tag_files:
            print(f"File: {finding['file']}")
            for tag in finding.get('private_tags', [])[:10]:  # Show first 10 private tags
                print(f"  {tag}")
    
    # Check if there are any files in the processed data directory that might contain classifications
    processed_data_dir = "/tmp2/b10902078/MEDSAM/processed_data/SpineMetsCT_npz"
    if os.path.exists(processed_data_dir):
        print("\nChecking processed data directory for classification information...")
        for root, _, files in os.walk(processed_data_dir):
            for file in files:
                if "label" in file.lower() or "class" in file.lower():
                    print(f"Found potentially relevant file: {os.path.join(root, file)}")
    
    print("\n===== CONCLUSION =====")
    if not lesion_types_found and not any('lytic' in desc.lower() or 'blastic' in desc.lower() or 'mixed' in desc.lower() 
                                         for desc in all_segment_descriptions.union(all_category_descriptions)):
        print("No explicit lesion type classifications (Lytic/Blastic/Mixed) found in the DICOM files.")
        print("The segmentation data appears to be binary (0=background, 1=lesion) without lesion type differentiation.")
        print("\nPossible reasons:")
        print("1. The lesion type information might not be stored in the DICOM files themselves")
        print("2. The information might be in a separate file not included in your dataset")
        print("3. The dataset description might not match the actual content of the dataset files")
        print("4. The information might be encoded in the pixel data values themselves (different values for different lesion types)")
        print("\nRecommendation:")
        print("Contact the dataset authors via https://doi.org/10.7937/kh36-ds04 for clarification")
        print("or check if the segmentation maps use different pixel values for different lesion types")

if __name__ == "__main__":
    find_lesion_classifications()