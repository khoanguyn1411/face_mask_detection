import os
import shutil
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# Configuration for original face-mask-detection dataset
FACE_MASK_SOURCE_DIR = Path("./datasets/face-mask-detection")
FACE_MASK_ANNOTATIONS_DIR = FACE_MASK_SOURCE_DIR / "annotations"
FACE_MASK_IMAGES_DIR = FACE_MASK_SOURCE_DIR / "images"

# Configuration for medical-mask-detection dataset
MEDICAL_MASK_SOURCE_DIR = Path("./datasets/medical-mask-detection")
MEDICAL_MASK_IMAGES_DIR = MEDICAL_MASK_SOURCE_DIR / "Medical mask" / "Medical Mask" / "images"
MEDICAL_MASK_ANNOTATIONS_DIR = MEDICAL_MASK_SOURCE_DIR / "Medical mask" / "Medical Mask" / "annotations"

# Output directory (same for both)
OUTPUT_DIR = Path("./datasets/face-mask-detection-processed")

# Class mapping for face mask detection
CLASS_MAPPING = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

# Medical mask class mapping to face mask classes
MEDICAL_MASK_CLASS_MAPPING = {
    "face_other_covering": 0,  # Maps to "with_mask"
    "hat": 1,                  # Maps to "without_mask"
    "hood": 1                  # Maps to "without_mask"
}

# Train/val/test split ratio
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


def create_output_structure():
    """Create the YOLOv8 directory structure."""
    subdirs = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    for subdir in subdirs:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created output directory structure at {OUTPUT_DIR}")


def parse_xml_annotation(xml_path):
    """
    Parse XML annotation file and extract bounding boxes.
    Returns: (image_filename, image_width, image_height, objects_list)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    
    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        # Convert to YOLO format (normalized center_x, center_y, width, height)
        center_x = ((xmin + xmax) / 2) / width
        center_y = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        
        # Get class id
        class_id = CLASS_MAPPING.get(class_name, 1)
        
        objects.append({
            "class_id": class_id,
            "center_x": center_x,
            "center_y": center_y,
            "width": box_width,
            "height": box_height
        })
    
    return filename, width, height, objects


def parse_json_annotation(json_path, images_dir):
    """
    Parse JSON annotation file from medical-mask-detection dataset.
    Returns: (image_filename, image_width, image_height, objects_list)
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return None, None, None, []
    
    filename = data.get("FileName", "")
    
    # Try to find the image file with different extensions
    base_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
    possible_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    image_path = None
    for ext in possible_extensions:
        candidate_path = images_dir / (base_filename + ext)
        if candidate_path.exists():
            image_path = candidate_path
            filename = base_filename + ext
            break
    
    if image_path is None or not image_path.exists():
        print(f"Error: Image file not found for {base_filename} in {images_dir}")
        return None, None, None, []
    
    # Get image dimensions using PIL
    try:
        from PIL import Image
        img = Image.open(image_path)
        width, height = img.size
    except Exception as e:
        print(f"Error reading image dimensions from {image_path}: {e}")
        return None, None, None, []
    
    objects = []
    annotations = data.get("Annotations", [])
    
    for ann in annotations:
        bbox = ann.get("BoundingBox", [])
        if len(bbox) != 4:
            continue
        
        xmin, ymin, xmax, ymax = bbox
        
        # Convert to YOLO format (normalized center_x, center_y, width, height)
        center_x = ((xmin + xmax) / 2) / width
        center_y = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        
        # Get class name and map to face mask classes
        class_name = ann.get("classname", "without_mask")
        class_id = MEDICAL_MASK_CLASS_MAPPING.get(class_name, 1)
        
        objects.append({
            "class_id": class_id,
            "center_x": center_x,
            "center_y": center_y,
            "width": box_width,
            "height": box_height
        })
    
    return filename, width, height, objects



def convert_annotations():
    """Convert all XML and JSON annotations to YOLO format from both datasets."""
    yolo_annotations = []
    
    # Process face-mask-detection dataset (XML format)
    if FACE_MASK_ANNOTATIONS_DIR.exists():
        xml_files = sorted(FACE_MASK_ANNOTATIONS_DIR.glob("*.xml"))
        print(f"Found {len(xml_files)} XML annotation files (face-mask-detection)")
        
        for xml_file in tqdm(xml_files, desc="Converting XML annotations"):
            try:
                filename, width, height, objects = parse_xml_annotation(xml_file)
                
                if objects and filename:  # Only include images with annotations
                    yolo_annotations.append({
                        "filename": filename,
                        "objects": objects,
                        "width": width,
                        "height": height,
                        "source": "face-mask"
                    })
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
        
        print(f"✓ Converted {len([a for a in yolo_annotations if a.get('source') == 'face-mask'])} XML annotations")
    else:
        print(f"Warning: Face-mask-detection dataset not found at {FACE_MASK_ANNOTATIONS_DIR}")
    
    # Process medical-mask-detection dataset (JSON format)
    if MEDICAL_MASK_ANNOTATIONS_DIR.exists():
        json_files = sorted(MEDICAL_MASK_ANNOTATIONS_DIR.glob("*.json"))
        print(f"Found {len(json_files)} JSON annotation files (medical-mask-detection)")
        
        initial_count = len(yolo_annotations)
        for json_file in tqdm(json_files, desc="Converting JSON annotations"):
            try:
                filename, width, height, objects = parse_json_annotation(json_file, MEDICAL_MASK_IMAGES_DIR)
                
                if objects and filename:  # Only include images with annotations
                    yolo_annotations.append({
                        "filename": filename,
                        "objects": objects,
                        "width": width,
                        "height": height,
                        "source": "medical-mask"
                    })
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        medical_count = len(yolo_annotations) - initial_count
        print(f"✓ Converted {medical_count} JSON annotations")
    else:
        print(f"Warning: Medical-mask-detection dataset not found at {MEDICAL_MASK_ANNOTATIONS_DIR}")
    
    print(f"✓ Total annotations converted: {len(yolo_annotations)}")
    return yolo_annotations


def split_dataset(yolo_annotations):
    """Split dataset into train, val, test sets."""
    indices = list(range(len(yolo_annotations)))
    
    # First split: train+val vs test
    train_val_indices, test_indices = train_test_split(
        indices, 
        test_size=TEST_RATIO, 
        random_state=42
    )
    
    # Second split: train vs val
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=VAL_RATIO / (1 - TEST_RATIO),
        random_state=42
    )
    
    splits = {
        "train": [yolo_annotations[i] for i in train_indices],
        "val": [yolo_annotations[i] for i in val_indices],
        "test": [yolo_annotations[i] for i in test_indices]
    }
    
    print(f"✓ Dataset split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits


def save_yolo_format(split_name, annotations):
    """Save images and labels in YOLO format from both datasets."""
    for ann in tqdm(annotations, desc=f"Processing {split_name} set"):
        # Determine source and image path
        source = ann.get("source", "face-mask")
        if source == "medical-mask":
            src_image_path = MEDICAL_MASK_IMAGES_DIR / ann["filename"]
        else:
            src_image_path = FACE_MASK_IMAGES_DIR / ann["filename"]
        
        if not src_image_path.exists():
            print(f"Warning: Image not found {src_image_path}")
            continue
        
        # Destination image path
        dst_image_path = OUTPUT_DIR / "images" / split_name / ann["filename"]
        
        # Copy image
        shutil.copy2(src_image_path, dst_image_path)
        
        # Create label file (txt format)
        label_filename = ann["filename"].rsplit(".", 1)[0] + ".txt"
        label_path = OUTPUT_DIR / "labels" / split_name / label_filename
        
        with open(label_path, "w") as f:
            for obj in ann["objects"]:
                line = (f"{obj['class_id']} "
                       f"{obj['center_x']:.6f} "
                       f"{obj['center_y']:.6f} "
                       f"{obj['width']:.6f} "
                       f"{obj['height']:.6f}\n")
                f.write(line)


def create_yaml_config():
    """Create dataset.yaml for YOLOv8."""
    yaml_content = {
        "path": str(OUTPUT_DIR.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_MAPPING),
        "names": {v: k.replace("_", " ").title() for k, v in CLASS_MAPPING.items()}
    }
    
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"✓ Created dataset.yaml at {yaml_path}")


def main():
    """Main preprocessing pipeline for both face-mask and medical-mask datasets."""
    print("=" * 60)
    print("🎯 Face & Medical Mask Detection - YOLOv8 Data Preprocessing")
    print("=" * 60)
    
    # Step 1: Create output structure
    print("\n[1/5] Creating output directory structure...")
    create_output_structure()
    
    # Step 2: Convert annotations
    print("\n[2/5] Converting XML and JSON annotations to YOLO format...")
    yolo_annotations = convert_annotations()
    
    if not yolo_annotations:
        print("ERROR: No annotations found in either dataset!")
        return
    
    # Step 3: Split dataset
    print("\n[3/5] Splitting dataset into train/val/test...")
    splits = split_dataset(yolo_annotations)
    
    # Step 4: Save to YOLO format
    print("\n[4/5] Saving images and labels...")
    for split_name, annotations in splits.items():
        save_yolo_format(split_name, annotations)
    
    # Step 5: Create YAML config
    print("\n[5/5] Creating YAML configuration...")
    create_yaml_config()
    
    # Calculate statistics by source
    face_mask_count = sum(1 for a in yolo_annotations if a.get('source') == 'face-mask')
    medical_mask_count = sum(1 for a in yolo_annotations if a.get('source') == 'medical-mask')
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing completed successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Total dataset size: {len(yolo_annotations)} images")
    print(f"   - Face-mask-detection: {face_mask_count} images")
    print(f"   - Medical-mask-detection: {medical_mask_count} images")
    print(f"🏷️  Classes: {', '.join(CLASS_MAPPING.keys())}")
    print(f"📈 Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
