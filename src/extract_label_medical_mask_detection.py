import json
from pathlib import Path
from collections import defaultdict

# Configuration
MEDICAL_MASK_SOURCE_DIR = Path("./datasets/medical-mask-detection")
MEDICAL_MASK_ANNOTATIONS_DIR = MEDICAL_MASK_SOURCE_DIR / "Medical mask" / "Medical Mask" / "annotations"


def extract_all_classnames():
    """Extract all unique classnames and sample images from medical-mask-detection annotations."""
    
    if not MEDICAL_MASK_ANNOTATIONS_DIR.exists():
        print(f"❌ Error: Annotations directory not found at {MEDICAL_MASK_ANNOTATIONS_DIR}")
        return
    
    # Track classnames, frequencies, and sample images
    classname_counts = defaultdict(int)
    classname_images = defaultdict(list)  # Store image filenames for each class
    all_classnames = set()
    json_files = sorted(MEDICAL_MASK_ANNOTATIONS_DIR.glob("*.json"))
    
    print("=" * 80)
    print("📋 Extracting Classnames & Sample Images from Medical Mask Detection Dataset")
    print("=" * 80)
    print(f"\n🔍 Found {len(json_files)} JSON annotation files\n")
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            filename = data.get("FileName", "unknown")
            annotations = data.get("Annotations", [])
            for ann in annotations:
                classname = ann.get("classname", "unknown")
                classname_counts[classname] += 1
                all_classnames.add(classname)
                
                # Store image filename (limit to 5 per class)
                if len(classname_images[classname]) < 5:
                    classname_images[classname].append(filename)
        
        except Exception as e:
            print(f"⚠️  Error reading {json_file.name}: {e}")
    
    # Display results
    print("=" * 80)
    print("📊 UNIQUE CLASSNAMES & SAMPLE IMAGES:")
    print("=" * 80)
    
    sorted_classnames = sorted(all_classnames)
    for i, classname in enumerate(sorted_classnames, 1):
        count = classname_counts[classname]
        images = classname_images[classname]
        print(f"\n{i}. {classname:<30} - Count: {count}")
        print(f"   Sample images:")
        for j, img in enumerate(images, 1):
            print(f"      {j}. {img}")
    
    print("\n" + "=" * 80)
    print("📈 SUMMARY:")
    print("=" * 80)
    print(f"Total Unique Classnames: {len(all_classnames)}")
    print(f"Total Annotations: {sum(classname_counts.values())}")
    print(f"Total Files Processed: {len(json_files)}")
    print("=" * 80 + "\n")
    
    # Create mapping for reference
    print("🔗 CLASSNAME MAPPING:")
    print("-" * 80)
    classname_mapping = {classname: i for i, classname in enumerate(sorted_classnames)}
    for classname, idx in classname_mapping.items():
        print(f"  {classname}: {idx}")
    print("-" * 80 + "\n")
    
    return sorted(all_classnames), classname_counts


if __name__ == "__main__":
    classnames, counts = extract_all_classnames()
