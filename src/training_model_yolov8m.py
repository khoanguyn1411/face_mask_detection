import os
import yaml
import platform
from pathlib import Path
from ultralytics import YOLO
import torch

# Detect platform
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS_OR_LINUX = platform.system() in ["Windows", "Linux"]

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATASET_YAML = PROJECT_DIR / "datasets" / "face-mask-detection-processed" / "dataset.yaml"
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
    # Check if MPS is available (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✓ Apple Metal Performance Shaders (MPS) available")
        DEVICE_INFO = {"type": "Apple Silicon (Metal)", "is_gpu": True, "torch_device": "mps"}
        # PyTorch will use MPS automatically
    else:
        print("⚠️  Metal GPU not available or not supported. Using CPU.")
        DEVICE_INFO = {"type": "CPU", "is_gpu": False, "torch_device": "cpu"}
    
    # Configure batch size for M4 Pro
    BATCH_SIZE_GPU = 16  # Reduced for faster training
    
elif IS_WINDOWS_OR_LINUX:
    print("💻 Windows/Linux detected - Using PyTorch with CUDA (NVIDIA GPU)")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ NVIDIA GPU detected: {gpu_name}")
        print(f"  Total Memory: {total_memory:.2f} GB")
        DEVICE_INFO = {"type": f"NVIDIA CUDA ({gpu_name})", "is_gpu": True, "torch_device": 0}
        
        # Configure batch size based on GPU memory
        if total_memory <= 4.5:
            BATCH_SIZE_GPU = 8  # GTX 1650 or similar
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
EPOCHS = 100  # Reduced from 150 for faster iteration
BATCH_SIZE = BATCH_SIZE_GPU  # Dynamically set based on device
IMG_SIZE = 192  # Reduced from 256 for faster training (4x faster per iteration)
PATIENCE = 15  # Reduced early stopping patience
DEVICE = DEVICE_INFO['torch_device']  # Use device from configuration above
NUM_WORKERS = 4  # Parallel data loading (adjust based on CPU cores)

# Inference optimization parameters
CONF_THRESHOLD = 0.45  # Confidence threshold (lower = fewer detections, faster NMS)
IOU_THRESHOLD = 0.45   # IOU threshold for NMS (lower = fewer overlapping boxes)
MAX_DETECTIONS = 300   # Maximum detections per image (reduces NMS workload)

BASE_NAME = "face_mask_detection_yolov8m_v2"
PREDICTION_NAME = "predictions_yolov8m_v2"


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
    print("=" * 80)
    print("💻 GPU Information & Compatibility Check")
    print("=" * 80)
    
    if IS_MAC:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ Apple Metal Performance Shaders (MPS) Available")
            print(f"  Device Type: {DEVICE_INFO['type']}")
            print(f"  Batch Size: {BATCH_SIZE} (optimized for M4 Pro)")
            print("\n✓ Metal GPU Configuration:")
            print("  - FP16 Training: ENABLED (saves memory)")
            print("  - Typical Training Speed: 15-25ms per iteration")
        else:
            print("⚠️  Metal GPU NOT available. Using CPU (very slow).")
            print(f"  Batch Size: {BATCH_SIZE}")
    
    elif IS_WINDOWS_OR_LINUX:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"✓ NVIDIA GPU Available")
            print(f"  Device: {gpu_name}")
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Batch Size: {BATCH_SIZE} (optimized for your GPU)")
            
            # Check if it's GTX 1650 or similar low-VRAM GPU
            if total_memory <= 4.5:
                print("\n⚠️  LOW VRAM GPU DETECTED (≤4.5GB)")
                print("  Recommendations:")
                if BATCH_SIZE > 4:
                    print(f"    • If OOM errors: Reduce BATCH_SIZE to {max(2, BATCH_SIZE//2)}")
                print(f"    • FP16 Training: ENABLED (saves ~50% memory)")
                if IMG_SIZE > 320:
                    print(f"    • Option: Reduce IMG_SIZE to 320 (currently {IMG_SIZE})")
                print("  For GTX 1650 specifically:")
                print("    ✓ Batch Size 8 should work")
                print("    ✓ Batch Size 4 is safer if OOM errors occur")
            elif total_memory <= 8:
                print(f"\n✓ GPU Memory OK for Batch Size {BATCH_SIZE}")
            else:
                print(f"\n✓ GPU Memory Good for any configuration")
        else:
            print("⚠️  CUDA not available. Training will use CPU (very slow).")
            print(f"  Batch Size: {BATCH_SIZE}")
    
    print("=" * 80 + "\n")
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
    print(f"  Compute Device: {DEVICE_INFO['type']}")
    print(f"  Early Stopping Patience: {PATIENCE}")
    print(f"  Dataset: {DATASET_YAML}")
    
    print(f"\n  Inference Optimization (prevents NMS timeout):")
    print(f"    - Confidence Threshold: {CONF_THRESHOLD}")
    print(f"    - IOU Threshold: {IOU_THRESHOLD}")
    print(f"    - Max Detections: {MAX_DETECTIONS}")
    
    # Show device details
    if IS_MAC and DEVICE_INFO['is_gpu']:
        print(f"\n🍎 Apple Silicon Configuration:")
        print(f"   Platform: macOS")
        print(f"   Expected Speed: 15-25ms per iteration")
    elif IS_WINDOWS_OR_LINUX and DEVICE_INFO['is_gpu']:
        print(f"\n💻 NVIDIA CUDA Configuration:")
        print(f"   Platform: {platform.system()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"\n💻 CPU Configuration:")
        print(f"   ⚠️  Training on CPU will be VERY SLOW")
    
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
            close_mosaic=8,  # Reduced from 10 for faster training
            project=str(RUNS_DIR),
            name=BASE_NAME,
            exist_ok=True,
            verbose=True,
            workers=NUM_WORKERS,  # Parallel data loading
            # Additional training parameters
            augment=True,
            mosaic=0.8,  # Reduced from 1.0 (mosaic less frequently)
            flipud=0.3,  # Reduced from 0.5
            fliplr=0.3,  # Reduced from 0.5
            degrees=5,   # Reduced from 10
            translate=0.05,  # Reduced from 0.1
            scale=0.3,   # Reduced from 0.5
            warmup_epochs=2,  # Reduced from 3
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
