import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# Configuration
SOURCE_DIR = Path("./datasets/face-mask-detection")
OUTPUT_DIR = Path("./datasets/face-mask-detection-processed")
ANNOTATIONS_DIR = SOURCE_DIR / "annotations"
IMAGES_DIR = SOURCE_DIR / "images"

# Class mapping for face mask detection
CLASS_MAPPING = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
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


def convert_annotations():
    """Convert all XML annotations to YOLO format."""
    yolo_annotations = []
    
    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files")
    
    for xml_file in tqdm(xml_files, desc="Converting annotations"):
        try:
            filename, width, height, objects = parse_xml_annotation(xml_file)
            
            if objects:  # Only include images with annotations
                yolo_annotations.append({
                    "filename": filename,
                    "objects": objects,
                    "width": width,
                    "height": height
                })
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
    
    print(f"✓ Converted {len(yolo_annotations)} annotations")
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
    """Save images and labels in YOLO format."""
    for ann in tqdm(annotations, desc=f"Processing {split_name} set"):
        # Source image path
        src_image_path = IMAGES_DIR / ann["filename"]
        
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
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("🎯 Face Mask Detection - YOLOv8 Data Preprocessing")
    print("=" * 60)
    
    # Verify source directories
    if not ANNOTATIONS_DIR.exists() or not IMAGES_DIR.exists():
        print("Error: Source directories not found!")
        return
    
    # Step 1: Create output structure
    print("\n[1/5] Creating output directory structure...")
    create_output_structure()
    
    # Step 2: Convert annotations
    print("\n[2/5] Converting XML annotations to YOLO format...")
    yolo_annotations = convert_annotations()
    
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
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing completed successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Dataset size: {sum(len(anns) for anns in splits.values())} images")
    print(f"🏷️  Classes: {', '.join(CLASS_MAPPING.keys())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
