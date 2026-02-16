import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATASET_YAML = PROJECT_DIR / "datasets" / "face-mask-detection-processed" / "dataset.yaml"
RUNS_DIR = PROJECT_DIR / "runs"
MODELS_DIR = PROJECT_DIR / "models"

# Training hyperparameters
# NOTE: Optimized for GTX 1650 (4GB VRAM)
# If you have a GTX 1650, these are safe settings. Adjust if needed:
# - More GPU RAM (8GB+): Increase BATCH_SIZE to 16-32
# - Less GPU RAM (<4GB): Reduce BATCH_SIZE to 2-4
EPOCHS = 100
BATCH_SIZE = 8  # GTX 1650 optimal: 8 (reduce to 4 if OOM errors)
IMG_SIZE = 416  # Can reduce to 320 for GTX 1650 if memory issues
PATIENCE = 20  # Early stopping patience
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # Use GPU if available


def verify_dataset():
    """Verify that the preprocessed dataset exists."""
    if not DATASET_YAML.exists():
        print(f"❌ Error: Dataset YAML not found at {DATASET_YAML}")
        print("Please run preprocessing.py first to prepare the dataset.")
        return False
    
    # Verify the YAML content
    with open(DATASET_YAML, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print("📊 Dataset Configuration:")
    print(f"  Path: {dataset_config.get('path')}")
    print(f"  Train: {dataset_config.get('train')}")
    print(f"  Val: {dataset_config.get('val')}")
    print(f"  Test: {dataset_config.get('test')}")
    print(f"  Classes: {dataset_config.get('nc')}")
    print(f"  Class names: {dataset_config.get('names')}")
    
    return True


def check_gpu_compatibility():
    """Check GPU compatibility and memory availability."""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Training will use CPU (very slow).")
        return True
    
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print("💻 GPU Information:")
    print(f"  Device: {gpu_name}")
    print(f"  Total Memory: {total_memory:.2f} GB")
    
    # Check if it's GTX 1650 or similar low-VRAM GPU
    if total_memory <= 4.5:
        print("\n⚠️  LOW VRAM GPU DETECTED (≤4.5GB)")
        print("  Recommendations:")
        print(f"    • Current Batch Size: {BATCH_SIZE}")
        if BATCH_SIZE > 4:
            print(f"    • Recommended Batch Size: 2-4 (reduce to {max(2, BATCH_SIZE//2)})")
        print(f"    • FP16 Training: ENABLED (saves ~50% memory) ✓")
        if IMG_SIZE > 320:
            print(f"    • Option: Reduce IMG_SIZE to 320 (currently {IMG_SIZE})")
        print("  For GTX 1650 specifically:")
        print("    ✓ Batch Size 8 should work")
        print("    ✓ Batch Size 4 is safer")
        print("    ✓ GPU memory will be mostly utilized (~80-90%)")
    elif total_memory <= 8:
        print(f"\n✓ GPU Memory OK for Batch Size {BATCH_SIZE}")
    else:
        print(f"\n✓ GPU Memory Good for any configuration")
    
    return True


def train_model():
    """Train YOLOv8m model on face mask detection dataset."""
    print("=" * 80)
    print("🎯 YOLOv8m Training - Face Mask Detection")
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
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Device: {'GPU' if DEVICE == 0 else 'CPU'}")
    print(f"  Early Stopping Patience: {PATIENCE}")
    print(f"  Dataset: {DATASET_YAML}")
    
    # Check CUDA availability
    print(f"\n💻 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load or create YOLOv8m model
    print("\n" + "=" * 80)
    print("📥 Loading YOLOv8m model...")
    print("=" * 80)
    
    try:
        model = YOLO("yolov8m.pt")  # Load pretrained YOLOv8m
        print("✓ YOLOv8m model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Train the model
    print("\n" + "=" * 80)
    print("🚀 Starting training...")
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
            name="face_mask_detection",
            exist_ok=True,
            verbose=True,
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
            half=torch.cuda.is_available(),  # Use FP16 if GPU available
        )
        
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        
        # Save model summary
        print("\n📈 Training Results:")
        print(f"  Best model path: {model.trainer.best.name if hasattr(model, 'trainer') else 'See runs directory'}")
        
        return results
        
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
    best_model_path = RUNS_DIR / "face_mask_detection" / "weights" / "best.pt"
    
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
    
    best_model_path = RUNS_DIR / "face_mask_detection" / "weights" / "best.pt"
    
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
            name="predictions",
            conf=0.5,
            iou=0.5
        )
        
        print(f"\n✓ Predictions saved to {RUNS_DIR / 'predictions'}")
        return results
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None


def export_model():
    """Export the model to different formats."""
    print("\n" + "=" * 80)
    print("📤 Exporting model...")
    print("=" * 80)
    
    best_model_path = RUNS_DIR / "face_mask_detection" / "weights" / "best.pt"
    
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
        shutil.copy(best_model_path, MODELS_DIR / "face_mask_detection_best.pt")
        print(f"✓ Best model saved to {MODELS_DIR / 'face_mask_detection_best.pt'}")
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        return None


def main():
    """Main training pipeline."""
    # Train model
    train_results = train_model()
    
    if train_results is None:
        print("\n❌ Training failed!")
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


if __name__ == "__main__":
    main()
