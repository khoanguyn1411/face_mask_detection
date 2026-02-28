"""
YOLOv8m Face Detection Only Training Script
============================================
This script trains a YOLOv8m model for face detection ONLY (single class: face).
This approach often improves face detection accuracy compared to simultaneous detection + classification.

Two-stage pipeline:
1. Face Detection (this model) - Detects all faces regardless of mask status
2. Mask Classification (separate model) - Classifies detected faces

Benefits:
- Better face detection recall (finds more faces)
- More robust to challenging conditions
- Focused learning on face localization
"""

import os
import yaml
import platform
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
import argparse

# Detect platform
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS_OR_LINUX = platform.system() in ["Windows", "Linux"]

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATASET_YAML = PROJECT_DIR / "datasets" / "face-mask-detection-processed" / "dataset_face_only.yaml"
RUNS_DIR = PROJECT_DIR / "runs"
MODELS_DIR = PROJECT_DIR / "models"

# Device detection and configuration
print("=" * 80)
print("🖥️  DEVICE DETECTION")
print("=" * 80)
print(f"Platform: {platform.system()} {platform.release()}")

DEVICE_INFO = {"type": "CPU", "is_gpu": False, "torch_device": "cpu"}

if IS_MAC:
    print("🍎 macOS detected - Using PyTorch with Metal Performance Shaders")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✓ Apple Metal Performance Shaders (MPS) available")
        DEVICE_INFO = {"type": "Apple Silicon (Metal)", "is_gpu": True, "torch_device": "mps"}
    else:
        print("⚠️  Metal GPU not available or not supported. Using CPU.")
        DEVICE_INFO = {"type": "CPU", "is_gpu": False, "torch_device": "cpu"}
    BATCH_SIZE_GPU = 16
    
elif IS_WINDOWS_OR_LINUX:
    print("💻 Windows/Linux detected - Using PyTorch with CUDA (NVIDIA GPU)")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ NVIDIA GPU detected: {gpu_name}")
        print(f"  Total Memory: {total_memory:.2f} GB")
        DEVICE_INFO = {"type": f"NVIDIA CUDA ({gpu_name})", "is_gpu": True, "torch_device": 0}
        
        if total_memory <= 4.5:
            BATCH_SIZE_GPU = 8
        elif total_memory <= 8:
            BATCH_SIZE_GPU = 16
        else:
            BATCH_SIZE_GPU = 32
    else:
        print("⚠️  CUDA not available. Training will use CPU (slower).")
        DEVICE_INFO = {"type": "CPU", "is_gpu": False, "torch_device": "cpu"}
        BATCH_SIZE_GPU = 4
else:
    print(f"⚠️  Unknown platform: {platform.system()}")
    BATCH_SIZE_GPU = 4

print(f"Selected Device: {DEVICE_INFO['type']}")
print("=" * 80 + "\n")

# Training hyperparameters
EPOCHS = 150
BATCH_SIZE = BATCH_SIZE_GPU
IMG_SIZE = 256
PATIENCE = 20
DEVICE = DEVICE_INFO['torch_device']

# Inference optimization parameters
CONF_THRESHOLD = 0.25  # Lower for face detection (want to find all faces)
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300

BASE_NAME = "face_detection_yolov8m"
PREDICTION_NAME = "predictions_face_yolov8m"


def create_face_only_dataset():
    """
    Create a face-only dataset from the existing multi-class dataset.
    Converts all mask classes (with_mask, without_mask, incorrect_mask) to a single 'face' class.
    """
    print("=" * 80)
    print("📦 Creating Face-Only Dataset")
    print("=" * 80)
    
    # Source dataset
    source_dataset_dir = PROJECT_DIR / "datasets" / "face-mask-detection-processed"
    source_yaml = source_dataset_dir / "dataset.yaml"
    
    if not source_yaml.exists():
        print(f"❌ Error: Source dataset not found at {source_yaml}")
        print("Please run preprocessing.py first to prepare the dataset.")
        return False
    
    # Load source dataset config
    with open(source_yaml, 'r') as f:
        source_config = yaml.safe_load(f)
    
    print(f"📊 Source Dataset: {source_config.get('nc')} classes")
    print(f"   Classes: {source_config.get('names')}")
    
    # Create face-only dataset config
    face_only_config = {
        'path': str(source_dataset_dir),
        'train': source_config['train'],
        'val': source_config['val'],
        'test': source_config['test'],
        'nc': 1,  # Single class: face
        'names': {0: 'face'}
    }
    
    # Save face-only dataset YAML
    with open(DATASET_YAML, 'w') as f:
        yaml.dump(face_only_config, f, default_flow_style=False)
    
    print(f"\n✓ Face-only dataset configuration created")
    print(f"  Path: {DATASET_YAML}")
    print(f"  Classes: 1 (face)")
    print(f"\n📝 Note: Label files will be automatically converted during training.")
    print(f"   All mask classes (0, 1, 2) will be treated as class 0 (face).")
    
    # Optional: Convert label files to single class
    convert_labels = input("\n🔄 Convert all label files now? (y/n, default=n): ").strip().lower()
    if convert_labels == 'y':
        convert_labels_to_single_class(source_dataset_dir)
    else:
        print("⚠️  Label conversion skipped. YOLOv8 will treat all classes as 'face' during training.")
        print("   For production deployment, consider converting labels permanently.")
    
    return True


def convert_labels_to_single_class(dataset_dir):
    """
    Convert all YOLO label files to single-class (face).
    Changes all class IDs (0, 1, 2) to 0 (face).
    """
    print("\n" + "=" * 80)
    print("🔄 Converting Labels to Single Class")
    print("=" * 80)
    
    splits = ['train', 'val', 'test']
    total_files = 0
    total_boxes = 0
    
    for split in splits:
        label_dir = dataset_dir / "labels" / split
        if not label_dir.exists():
            print(f"⚠️  Directory not found: {label_dir}")
            continue
        
        label_files = list(label_dir.glob("*.txt"))
        print(f"\n📁 Processing {split}: {len(label_files)} files")
        
        for label_file in label_files:
            # Read original labels
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Convert all class IDs to 0 (face)
            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Format: class_id x_center y_center width height
                    parts[0] = '0'  # Change class to 0 (face)
                    converted_lines.append(' '.join(parts) + '\n')
                    total_boxes += 1
            
            # Write converted labels
            with open(label_file, 'w') as f:
                f.writelines(converted_lines)
            
            total_files += 1
    
    print(f"\n✓ Conversion complete!")
    print(f"  Files processed: {total_files}")
    print(f"  Bounding boxes converted: {total_boxes}")
    print("=" * 80)


def verify_dataset():
    """Verify that the face-only dataset exists."""
    if not DATASET_YAML.exists():
        print(f"❌ Error: Face-only dataset YAML not found at {DATASET_YAML}")
        print("\nCreating face-only dataset...")
        if not create_face_only_dataset():
            return False
    
    # Verify the YAML content
    with open(DATASET_YAML, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print("📊 Face Detection Dataset Configuration:")
    print(f"  Path: {dataset_config.get('path')}")
    print(f"  Train: {dataset_config.get('train')}")
    print(f"  Val: {dataset_config.get('val')}")
    print(f"  Test: {dataset_config.get('test')}")
    print(f"  Classes: {dataset_config.get('nc')}")
    print(f"  Class names: {dataset_config.get('names')}")
    
    return True


def check_gpu_compatibility():
    """Check GPU compatibility and memory availability."""
    print("=" * 80)
    print("💻 GPU Information & Compatibility Check")
    print("=" * 80)
    
    if IS_MAC:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ Apple Metal Performance Shaders (MPS) Available")
            print(f"  Device Type: {DEVICE_INFO['type']}")
            print(f"  Batch Size: {BATCH_SIZE}")
            print("\n✓ Metal GPU Configuration:")
            print("  - FP16 Training: ENABLED (saves memory)")
        else:
            print("⚠️  Metal GPU NOT available. Using CPU (very slow).")
    
    elif IS_WINDOWS_OR_LINUX:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"✓ NVIDIA GPU Available")
            print(f"  Device: {gpu_name}")
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Batch Size: {BATCH_SIZE}")
            
            if total_memory <= 4.5:
                print("\n⚠️  LOW VRAM GPU DETECTED (≤4.5GB)")
                print("  Recommendations:")
                if BATCH_SIZE > 4:
                    print(f"    • If OOM errors: Reduce BATCH_SIZE to {max(2, BATCH_SIZE//2)}")
                print(f"    • FP16 Training: ENABLED")
        else:
            print("⚠️  CUDA not available. Using CPU.")
    
    print("=" * 80 + "\n")
    return True


def check_checkpoint():
    """Check if there's a previous checkpoint to resume from."""
    checkpoint_path = RUNS_DIR / BASE_NAME / "weights" / "last.pt"
    
    if checkpoint_path.exists():
        print("=" * 80)
        print("🔄 CHECKPOINT FOUND")
        print("=" * 80)
        print(f"📍 Found previous checkpoint at: {checkpoint_path}")
        print("\nYou can resume training by running with: --resume")
        print("Or start fresh with: --fresh")
        print("=" * 80 + "\n")
        return checkpoint_path
    
    return None


def train_model(resume=False):
    """Train YOLOv8m model on face detection only (single class)."""
    print("=" * 80)
    print("🎯 YOLOv8m Training - Face Detection Only (Single Class)")
    print("=" * 80)
    
    # Verify dataset
    if not verify_dataset():
        return None
    
    # Check GPU compatibility
    check_gpu_compatibility()
    
    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("🔧 Training Configuration:")
    print("=" * 80)
    print(f"  Model: YOLOv8m")
    print(f"  Task: Face Detection (Single Class)")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Compute Device: {DEVICE_INFO['type']}")
    print(f"  Early Stopping Patience: {PATIENCE}")
    print(f"  Dataset: {DATASET_YAML}")
    
    print(f"\n  Inference Optimization:")
    print(f"    - Confidence Threshold: {CONF_THRESHOLD} (lower for better recall)")
    print(f"    - IOU Threshold: {IOU_THRESHOLD}")
    print(f"    - Max Detections: {MAX_DETECTIONS}")
    
    print(f"\n💡 Training Strategy:")
    print(f"   This model focuses ONLY on detecting faces (not classifying masks).")
    print(f"   Expected benefits:")
    print(f"   ✓ Better face detection accuracy")
    print(f"   ✓ Higher recall (finds more faces)")
    print(f"   ✓ More robust to challenging conditions")
    
    # Load or create YOLOv8m model
    print("\n" + "=" * 80)
    if resume:
        print("📥 Resuming training from checkpoint...")
    else:
        print("📥 Loading YOLOv8m model...")
    print("=" * 80)
    
    try:
        if resume:
            checkpoint_path = RUNS_DIR / BASE_NAME / "weights" / "last.pt"
            if checkpoint_path.exists():
                model = YOLO(str(checkpoint_path))
                print(f"✓ Checkpoint loaded successfully: {checkpoint_path}")
            else:
                print(f"❌ Checkpoint not found at {checkpoint_path}")
                print("Starting fresh training instead...")
                # Check for local yolov8m.pt file first
                local_model = PROJECT_DIR / "yolov8m.pt"
                if local_model.exists():
                    model = YOLO(str(local_model))
                    print(f"✓ Using local model: {local_model}")
                else:
                    model = YOLO("yolov8m.pt")  # Will download if not found
                    print("✓ YOLOv8m model loaded (downloaded if needed)")
        else:
            # Check for local yolov8m.pt file first
            local_model = PROJECT_DIR / "yolov8m.pt"
            if local_model.exists():
                model = YOLO(str(local_model))
                print(f"✓ Using local model: {local_model}")
            else:
                model = YOLO("yolov8m.pt")  # Will download if not found
                print("✓ YOLOv8m model loaded (downloaded if needed)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Train the model
    print("\n" + "=" * 80)
    print("🚀 Starting training...")
    print("💡 Press Ctrl+C to pause training (checkpoint will be saved)")
    print("=" * 80)
    
    try:
        results = model.train(
            data=str(DATASET_YAML),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            patience=PATIENCE,
            save=True,
            close_mosaic=10,
            project=str(RUNS_DIR),
            name=BASE_NAME,
            exist_ok=True,
            verbose=True,
            resume=resume,  # Resume from checkpoint if True
            # Additional training parameters
            augment=True,
            mosaic=1.0,
            flipud=0.5,
            fliplr=0.5,
            degrees=10,
            translate=0.1,
            scale=0.5,
            warmup_epochs=3,
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            optimizer="SGD",
            # Validation parameters
            val=True,
            # Logging parameters
            plots=True,
            half=DEVICE_INFO['is_gpu'],  # Use FP16 if GPU available (CUDA or MPS)
        )
        
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        
        # Save model summary
        print("\n📈 Training Results:")
        print(f"  Best model path: {model.trainer.best.name if hasattr(model, 'trainer') else 'See runs directory'}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("⏸️  Training paused by user (Ctrl+C)")
        print("=" * 80)
        print("\n💾 Checkpoint has been saved automatically!")
        print(f"📍 Run with: python src/traning_model_yolov8m_face.py --resume")
        print("   to continue training from where you left off")
        print("=" * 80 + "\n")
        return None
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_model():
    """Validate the trained model."""
    print("\n" + "=" * 80)
    print("🔍 Validating model...")
    print("=" * 80)
    
    # Find the best model
    best_model_path = RUNS_DIR / BASE_NAME / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print(f"⚠️  Best model not found at {best_model_path}")
        return None
    
    try:
        model = YOLO(str(best_model_path))
        
        # Validate
        metrics = model.val(
            data=str(DATASET_YAML),
            device=DEVICE
        )
        
        print("\n📊 Validation Metrics:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return None


def test_model():
    """Test the model on test set."""
    print("\n" + "=" * 80)
    print("🧪 Testing model...")
    print("=" * 80)
    
    best_model_path = RUNS_DIR / BASE_NAME / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print(f"⚠️  Best model not found at {best_model_path}")
        return None
    
    try:
        model = YOLO(str(best_model_path))
        
        # Check if test set exists
        test_dir = PROJECT_DIR / "datasets" / "face-mask-detection-processed" / "images" / "test"
        
        if not test_dir.exists():
            print(f"⚠️  Test directory not found at {test_dir}")
            return None
        
        # Run prediction on test set
        results = model.predict(
            source=str(test_dir),
            device=DEVICE,
            save=True,
            project=str(RUNS_DIR),
            name=PREDICTION_NAME,
            conf=CONF_THRESHOLD,           # Optimized confidence threshold
            iou=IOU_THRESHOLD,              # Optimized IOU threshold
            max_det=MAX_DETECTIONS,         # Limit detections for faster NMS
            imgsz=IMG_SIZE,                 # Use same image size as training
            half=DEVICE_INFO['is_gpu'],     # Use FP16 if GPU available
            verbose=True
        )
        
        print(f"\n✓ Predictions saved to {RUNS_DIR / PREDICTION_NAME}")
        return results
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None


def export_model():
    """Export the model to different formats."""
    print("\n" + "=" * 80)
    print("📤 Exporting model...")
    print("=" * 80)
    
    best_model_path = RUNS_DIR / BASE_NAME / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print(f"⚠️  Best model not found at {best_model_path}")
        return None
    
    try:
        model = YOLO(str(best_model_path))
        
        # Export to multiple formats
        export_formats = ['onnx', 'torchscript', 'tflite']  # Add more if needed
        
        for fmt in export_formats:
            try:
                exported_path = model.export(format=fmt)
                print(f"✓ Model exported to {fmt}: {exported_path}")
            except Exception as e:
                print(f"⚠️  Could not export to {fmt}: {e}")
        
        # Also save the best.pt to models directory
        import shutil
        shutil.copy(best_model_path, MODELS_DIR / f"{BASE_NAME}_best.pt")
        print(f"✓ Best model saved to {MODELS_DIR / f'{BASE_NAME}_best.pt'}")
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        return None


def main():
    """Main training pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="YOLOv8m Training for Face Detection Only")
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--fresh', action='store_true', help='Start fresh training (ignore previous checkpoint)')
    parser.add_argument('--setup', action='store_true', help='Setup face-only dataset (convert from multi-class)')
    args = parser.parse_args()
    
    # If setup mode, only run dataset creation
    if args.setup:
        print("🔧 Setting up face-only dataset...")
        create_face_only_dataset()
        return
    
    # Check for checkpoint
    checkpoint = check_checkpoint()
    
    # Determine if we should resume
    resume = False
    if args.resume and checkpoint:
        resume = True
        print("▶️  Resuming training from checkpoint...\n")
    elif args.fresh:
        print("🆕 Starting fresh training (ignoring checkpoint)...\n")
    elif checkpoint and not args.resume:
        # Ask user interactively
        print("\n❓ How would you like to proceed?")
        print("   1. Resume from checkpoint (--resume)")
        print("   2. Start fresh (--fresh)")
        choice = input("\nEnter choice (1 or 2) [default=1]: ").strip() or "1"
        if choice == "1":
            resume = True
        print()
    
    # Train model
    train_results = train_model(resume=resume)
    
    if train_results is None:
        if resume:
            print("💡 Training paused. Run again with --resume to continue!")
        else:
            print("\n❌ Training failed or paused!")
        return
    
    # Validate model
    validate_model()
    
    # Test model
    test_model()
    
    # Export model
    export_model()
    
    print("\n" + "=" * 80)
    print("🎉 Training pipeline completed!")
    print("=" * 80)
    print(f"\n📁 Results saved to: {RUNS_DIR}")
    print(f"📁 Models saved to: {MODELS_DIR}")
    
    print("\n🎯 Next Steps - Two-Stage Pipeline:")
    print("  1. Use this model for face detection (finds all faces)")
    print("  2. Crop detected faces")
    print("  3. Use a separate CNN/classifier for mask classification")
    print("     (with_mask / without_mask / incorrect_mask)")


if __name__ == "__main__":
    main()
