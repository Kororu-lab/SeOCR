import os
import json
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import Config

def verify_json_structure(json_path):
    """Verify that JSON file has all required fields"""
    required_fields = [
        'Typeface', 'Info_id', 'Category_name', 'Info_name',
        'Period', 'Author', 'Difficulty', 'License_name',
        'Info_Data_created', 'Image_id', 'Image_filename',
        'Image_Data_captured', 'Image_Width', 'Image_Height',
        'Image_dpi', 'Image_Char_no', 'Text_Coord'
    ]
    
    try:
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return False, f"Missing fields: {', '.join(missing_fields)}"
            
        if not isinstance(data['Text_Coord'], list):
            return False, "Text_Coord is not a list"
            
        for coord in data['Text_Coord']:
            if 'bbox' not in coord or 'annotate' not in coord:
                return False, "Invalid Text_Coord structure"
                
        return True, "Valid JSON structure"
    except Exception as e:
        return False, str(e)

def verify_image(image_path):
    """Verify that image file is valid and can be opened"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, "Valid image file"
    except Exception as e:
        return False, str(e)

def verify_data_directory(data_dir):
    """Verify all data in the directory"""
    results = {
        'total_files': 0,
        'valid_json': 0,
        'invalid_json': 0,
        'valid_images': 0,
        'invalid_images': 0,
        'missing_pairs': 0,
        'errors': []
    }
    
    for data_type in ['manuscripts', 'woodblocks', 'movable_types']:
        type_dir = os.path.join(data_dir, data_type)
        if not os.path.exists(type_dir):
            results['errors'].append(f"Directory not found: {type_dir}")
            continue
            
        for root, _, files in os.walk(type_dir):
            json_files = {f[:-5] for f in files if f.endswith('.json')}
            png_files = {f[:-4] for f in files if f.endswith('.png')}
            
            # Check for missing pairs
            missing_images = json_files - png_files
            missing_json = png_files - json_files
            results['missing_pairs'] += len(missing_images) + len(missing_json)
            
            for file in files:
                results['total_files'] += 1
                file_path = os.path.join(root, file)
                
                if file.endswith('.json'):
                    valid, message = verify_json_structure(file_path)
                    if valid:
                        results['valid_json'] += 1
                    else:
                        results['invalid_json'] += 1
                        results['errors'].append(f"Invalid JSON {file}: {message}")
                
                elif file.endswith('.png'):
                    valid, message = verify_image(file_path)
                    if valid:
                        results['valid_images'] += 1
                    else:
                        results['invalid_images'] += 1
                        results['errors'].append(f"Invalid image {file}: {message}")
    
    return results

def print_results(results):
    """Print verification results in a formatted way"""
    print("\n=== Data Verification Results ===")
    print(f"Total files checked: {results['total_files']}")
    print(f"\nJSON Files:")
    print(f"  Valid: {results['valid_json']}")
    print(f"  Invalid: {results['invalid_json']}")
    print(f"\nImage Files:")
    print(f"  Valid: {results['valid_images']}")
    print(f"  Invalid: {results['invalid_images']}")
    print(f"\nMissing pairs: {results['missing_pairs']}")
    
    if results['errors']:
        print("\nErrors found:")
        for error in results['errors']:
            print(f"  - {error}")
    else:
        print("\nNo errors found!")

if __name__ == "__main__":
    data_dir = Config.DATA_ROOT
    print(f"Verifying data in: {data_dir}")
    results = verify_data_directory(data_dir)
    print_results(results) 