import kagglehub

import os
import shutil
from pathlib import Path

# Create datasets directory if it doesn't exist
datasets_dir = Path("./datasets").resolve()
datasets_dir.mkdir(parents=True, exist_ok=True)

# Download dataset (this caches it by default)
cached_path = kagglehub.dataset_download("andrewmvd/face-mask-detection")

# Move the dataset to our datasets folder
dataset_name = "face-mask-detection"
target_path = datasets_dir / dataset_name

if target_path.exists():
    shutil.rmtree(target_path)

shutil.move(cached_path, str(target_path))

print("✓ Dataset moved successfully!")
print(f"Dataset location: {target_path}")