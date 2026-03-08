import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import os
import platform

# These MUST be set BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Detect platform and configure appropriately
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS_OR_LINUX = platform.system() in ["Windows", "Linux"]

# Configure CUDA only for Windows/Linux with NVIDIA GPUs
if IS_WINDOWS_OR_LINUX:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU (GTX 1650)


# Set seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Device configuration and detection
print("=" * 80)
print("🖥️  DEVICE DETECTION")
print("=" * 80)
print(f"Platform: {platform.system()} {platform.release()}")

DEVICE_INFO = {"type": "CPU", "is_gpu": False}

if IS_MAC:
    print("🍎 macOS detected - Using Metal Performance Shaders (MPS)")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ Apple Metal GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for Apple Silicon")
            DEVICE_INFO = {"type": "Apple Silicon (Metal)", "is_gpu": True}
        else:
            print("⚠️  No Metal GPU detected. Using CPU (slower).")
    except Exception as e:
        print(f"⚠️  Error configuring Metal GPU: {e}")

elif IS_WINDOWS_OR_LINUX:
    print("💻 Windows/Linux detected - Using CUDA (NVIDIA GPU)")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ NVIDIA GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   - {gpu.name}")
            print(f"✓ GPU memory growth enabled for CUDA")
            DEVICE_INFO = {"type": "NVIDIA CUDA", "is_gpu": True}
        else:
            print("⚠️  No NVIDIA GPU detected. Using CPU (slower).")
            # Unset CUDA_VISIBLE_DEVICES if no GPU found
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    except RuntimeError as e:
        print(f"⚠️  Error configuring CUDA GPU: {e}")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    print(f"⚠️  Unknown platform: {platform.system()}")

print(f"Selected Device: {DEVICE_INFO['type']}")
print("=" * 80 + "\n")


# Configuration
PROJECT_DIR = Path(__file__).parent.parent
PREPROCESSED_DIR = PROJECT_DIR / "datasets" / "face-mask-detection-processed"
IMAGES_TRAIN_DIR = PREPROCESSED_DIR / "images" / "train"
IMAGES_VAL_DIR = PREPROCESSED_DIR / "images" / "val"
IMAGES_TEST_DIR = PREPROCESSED_DIR / "images" / "test"
LABELS_TRAIN_DIR = PREPROCESSED_DIR / "labels" / "train"
LABELS_VAL_DIR = PREPROCESSED_DIR / "labels" / "val"
LABELS_TEST_DIR = PREPROCESSED_DIR / "labels" / "test"
MODELS_DIR = PROJECT_DIR / "models"

# Training parameters - Optimized for both devices
IMG_SIZE = (192, 192)  # Balance between speed and accuracy - smaller is faster
EPOCHS = 150  # Extended for transfer learning fine-tuning
LEARNING_RATE = 0.0005  # Lower LR for fine-tuning pre-trained weights

# Batch size depends on device - optimized for memory usage
if DEVICE_INFO['is_gpu']:
    if IS_MAC:
        BATCH_SIZE = 32  # M4 Pro has good GPU bandwidth
    else:  # NVIDIA GTX 1650
        BATCH_SIZE = 16  # GTX 1650 has limited VRAM (~2-4GB)
else:
    BATCH_SIZE = 8  # CPU - use smaller batches

# Split ratios (same as preprocessing.py)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Class mapping
CLASS_MAPPING = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

# Reverse mapping for display
CLASS_NAMES = {v: k.replace("_", " ").title()
               for k, v in CLASS_MAPPING.items()}


def load_images_from_split(image_dir, labels_dir, yolo_model):
    """Load images and extract face crops using YOLOv26m detection instead of ground truth."""
    images = []
    classes = []

    print(f"  Using YOLOv26m for face detection (not ground truth)...")

    # Get all image files
    image_files = sorted(image_dir.glob(
        "*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))

    total_files = len(image_files)
    detected_faces = 0
    skipped_images = 0

    for idx, img_path in enumerate(image_files):
        if (idx + 1) % max(1, total_files // 10) == 0:
            print(
                f"    Progress: {idx + 1}/{total_files} images processed, {detected_faces} faces detected...")

        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                skipped_images += 1
                continue

            img_height, img_width = img.shape[:2]

            # Run YOLOv26m inference on image
            results = yolo_model.predict(source=img, conf=0.4, verbose=False)

            # Check if detections exist
            if results and len(results) > 0:
                detection_result = results[0]
                boxes = detection_result.boxes

                if boxes is not None and len(boxes) > 0:
                    # Extract face crops from detections
                    for box in boxes:
                        try:
                            # Get bounding box coordinates (in pixels)
                            x1, y1, x2, y2 = box.xyxy[0].cpu(
                            ).numpy().astype(int)
                            class_id = int(box.cls[0].cpu().numpy())
                            confidence = float(box.conf[0].cpu().numpy())

                            # Clip to image boundaries
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(img_width, x2)
                            y2 = min(img_height, y2)

                            # Extract face crop
                            face_crop = img[y1:y2, x1:x2]

                            # Skip if crop is too small or invalid
                            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                                continue

                            # Resize to standard size
                            img_resized = cv2.resize(face_crop, IMG_SIZE)
                            img_rgb = cv2.cvtColor(
                                img_resized, cv2.COLOR_BGR2RGB)

                            images.append(img_rgb)
                            classes.append(class_id)
                            detected_faces += 1

                        except Exception as e:
                            continue
            else:
                skipped_images += 1

        except Exception as e:
            skipped_images += 1
            continue

    print(
        f"  ✓ Extracted {detected_faces} face crops from {total_files} images")
    if skipped_images > 0:
        print(f"    (Skipped {skipped_images} images with no faces detected)")

    return np.array(images, dtype=np.uint8), np.array(classes)


def prepare_dataset():
    """Load preprocessed dataset and extract face crops using YOLOv26m detection."""
    print("=" * 80)
    print("📊 PREPARING DATASET - Extracting Face Crops with YOLOv26m Detection")
    print("   (Using YOLOv26m for face detection instead of ground truth annotations)")
    print("=" * 80)

    # Verify dataset exists
    if not PREPROCESSED_DIR.exists():
        print(f"❌ Error: Preprocessed dataset not found at {PREPROCESSED_DIR}")
        print("Please run preprocessing.py first to prepare the dataset.")
        return None

    # Load YOLOv26m model for face detection
    print("\n📥 Loading YOLOv26m model for face detection...")
    try:
        PROJECT_DIR = Path(__file__).parent.parent
        model_path = PROJECT_DIR / "runs" / \
            "face_mask_detection_yolo26m_v1" / "weights" / "best.pt"

        if not model_path.exists():
            print(f"⚠️  Warning: Model not found at {model_path}")
            print(f"   Trying alternative path...")
            # Try to use a default model if custom training not available
            # Use YOLOv26m segmentation model as fallback
            yolo_model = YOLO("yolo26m.pt")
            print(f"   ✓ Using default YOLOv26m model")
        else:
            yolo_model = YOLO(str(model_path))
            print(f"   ✓ Loaded trained YOLOv26m model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading YOLOv26m model: {e}")
        return None

    # Load train split
    print(f"\nExtracting train split face crops using YOLOv26m detection...")
    X_train, y_train = load_images_from_split(
        IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR, yolo_model)

    # Load validation split
    print(f"Extracting validation split face crops using YOLOv26m detection...")
    X_val, y_val = load_images_from_split(
        IMAGES_VAL_DIR, LABELS_VAL_DIR, yolo_model)

    # Load test split
    print(f"Extracting test split face crops using YOLOv26m detection...")
    X_test, y_test = load_images_from_split(
        IMAGES_TEST_DIR, LABELS_TEST_DIR, yolo_model)

    print(f"\n✓ Face Crops Extracted Successfully with YOLOv26m")
    print(f"\n📊 Extracted Face Regions:")
    print(f"  Total: {len(X_train) + len(X_val) + len(X_test)} face crops")

    # Print train split statistics
    print(f"\n  Train Split: {len(X_train)} images")
    print(f"    - With Mask:              {sum(1 for c in y_train if c == 0)}")
    print(f"    - Without Mask:           {sum(1 for c in y_train if c == 1)}")
    print(f"    - Mask Weared Incorrect:  {sum(1 for c in y_train if c == 2)}")

    # Print validation split statistics
    print(f"\n  Validation Split: {len(X_val)} images")
    print(f"    - With Mask:              {sum(1 for c in y_val if c == 0)}")
    print(f"    - Without Mask:           {sum(1 for c in y_val if c == 1)}")
    print(f"    - Mask Weared Incorrect:  {sum(1 for c in y_val if c == 2)}")

    # Print test split statistics
    print(f"\n  Test Split: {len(X_test)} images")
    print(f"    - With Mask:              {sum(1 for c in y_test if c == 0)}")
    print(f"    - Without Mask:           {sum(1 for c in y_test if c == 1)}")
    print(f"    - Mask Weared Incorrect:  {sum(1 for c in y_test if c == 2)}")

    print(f"\n📂 Dataset Split Ratio:")
    total = len(X_train) + len(X_val) + len(X_test)
    print(f"  Train: {len(X_train)} ({len(X_train)/total*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/total*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/total*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard-to-classify examples, especially minority classes.

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples). Default: 2
        alpha: Weighting factor for class balance. Default: 0.25
    """
    def focal_loss_fn(y_true, y_pred):
        # y_true shape: [batch_size] - class indices
        # y_pred shape: [batch_size, num_classes] - logits

        # Convert logits to probabilities
        y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Convert y_true to one-hot encoding
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)

        # Calculate cross entropy loss
        ce_loss = -y_true_one_hot * tf.math.log(y_pred)
        ce_loss = tf.reduce_sum(ce_loss, axis=-1)

        # Calculate focal weight: (1 - p_t)^gamma
        # Get the probability of the true class
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, gamma)

        # Apply focal weight to cross entropy loss
        focal_loss_value = alpha * focal_weight * ce_loss

        return tf.reduce_mean(focal_loss_value)

    return focal_loss_fn


def build_model(model_type="mobilenet"):
    """Build Transfer Learning model with MobileNetV3Large and Focal Loss."""
    print("\n" + "=" * 80)
    print("🔧 BUILDING TRANSFER LEARNING MODEL (MobileNetV3Large + Focal Loss)")
    print("=" * 80)
    print(f"Compute Device: {DEVICE_INFO['type']}")

    # Resize input to match IMG_SIZE (currently 192x192)
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

    # Load pre-trained MobileNetV3Large from ImageNet
    print("\n📥 Loading pre-trained MobileNetV3Large (ImageNet weights)...")
    base_model = keras.applications.MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze initial layers for faster training
    base_model.trainable = False
    print(f"   ✓ Froze {len(base_model.layers)} pre-trained layers")

    # Build the full model
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Pre-trained feature extractor
        base_model,

        # Custom classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Output layer - NO softmax (focal loss expects raw logits)
        layers.Dense(3, activation='linear')  # 3 classes
    ])

    # Compile with Focal Loss
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    focal_loss_fn = focal_loss(gamma=2.0, alpha=0.25)

    model.compile(
        optimizer=optimizer,
        loss=focal_loss_fn,
        metrics=['accuracy']
    )

    print(f"\n📊 Model Summary:")
    print(f"  Base Model Parameters: ~5.1M (MobileNetV3Large)")
    print(
        f"  Custom Head Parameters: {sum([np.prod(w.shape) for w in model.trainable_weights])}")
    print(f"  Total Parameters: {model.count_params():,}")
    print(f"\n🎯 Loss Function: Focal Loss (γ=2.0, α=0.25)")
    print(f"   → Focuses on hard-to-classify and minority class samples")
    print(f"   → Excellent for imbalanced datasets")

    return model


def train_model(model, X_train, X_val, y_train, y_val):
    """Train the Transfer Learning model with Focal Loss for imbalanced data."""
    print("\n" + "=" * 80)
    print("🚀 TRAINING TRANSFER LEARNING MODEL (MobileNetV2 + Focal Loss)")
    print("=" * 80)

    # Compute class weights to handle imbalanced dataset
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(y_train),
                                         y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    print(f"\n📊 Class Weights (combined with Focal Loss):")
    print(f"  With Mask:               {class_weight_dict.get(0, 1):.2f}x")
    print(f"  Without Mask:            {class_weight_dict.get(1, 1):.2f}x")
    print(f"  Mask Weared Incorrect:   {class_weight_dict.get(2, 1):.2f}x")

    # Enhanced callbacks with increased patience for transfer learning
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Increased for transfer learning convergence
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,  # More patience for stable learning
            min_lr=1e-7
        )
    ]

    print(f"\nTraining Configuration:")
    print(f"  Epochs:                  {EPOCHS}")
    print(
        f"  Batch Size:              {BATCH_SIZE} (optimized for {DEVICE_INFO['type']})")
    print(f"  Learning Rate:           {LEARNING_RATE}")
    print(f"  Early Stop Patience:     20 epochs")
    print(f"  Compute Device:          {DEVICE_INFO['type']}")
    print(f"  Loss Function:           Focal Loss (γ=2.0, α=0.25)")
    print(f"  Data Augmentation:       rotation ±20°, shifts ±15%, zoom ±20%, flip")

    # Enhanced data augmentation for better minority class learning
    train_datagen = ImageDataGenerator(
        rescale=1./255,                    # Normalize from uint8
        rotation_range=20,                 # Random rotations up to 20 degrees
        width_shift_range=0.15,            # Random horizontal shift
        height_shift_range=0.15,           # Random vertical shift
        zoom_range=0.2,                    # Random zoom
        horizontal_flip=True,              # Random horizontal flips
        fill_mode='nearest'                # Fill missing pixels after transforms
    )

    # Create generators for memory-efficient batch loading
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Validation data - rescale uint8 on the fly
    val_data = X_val.astype(np.float32) / 255.0

    # Calculate steps per epoch (ceiling division to ensure full coverage)
    steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=(val_data, y_val),
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight_dict,  # Apply class weights with focal loss
        steps_per_epoch=steps_per_epoch  # Explicitly set steps per epoch
    )

    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    print("\n" + "=" * 80)
    print("📊 EVALUATING MODEL")
    print("=" * 80)

    # Convert test data to float32 for evaluation
    X_test_float = X_test.astype(np.float32) / 255.0
    test_loss, test_accuracy = model.evaluate(
        X_test_float, y_test, verbose=0, batch_size=BATCH_SIZE)

    print(f"\n🎯 Test Results:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model.predict(X_test_float, verbose=0, batch_size=BATCH_SIZE)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_classes,
          target_names=list(CLASS_NAMES.values()),
          labels=[0, 1, 2]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes, labels=[0, 1, 2])
    print(f"\n🔢 Confusion Matrix:")
    print(cm)

    return test_accuracy, y_pred_classes


def visualize_training(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'],
                 label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'],
                 label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_model(model):
    """Save trained model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "face_mask_mobilenet_yolo26m_best.h5"
    model.save(str(model_path))

    print(f"\n✓ Model saved to {model_path}")


def main():
    """Main training pipeline with Transfer Learning + Focal Loss."""
    print("\n" + "=" * 80)
    print("🎯 Face Mask Detection - Transfer Learning Training")
    print("   Architecture: MobileNetV2 + Focal Loss")
    print("   Datasets: Face-Mask-Detection + Medical-Mask-Detection (Combined)")
    print("   Face Detection: YOLOv26m (not ground truth annotations)")
    print("   Goal: Detect all mask classes with focus on minority classes")
    print("=" * 80)

    # Step 1: Prepare dataset
    print("\n[1/4] Preparing dataset...")
    dataset = prepare_dataset()

    if dataset is None:
        print("\n❌ Failed to prepare dataset!")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = dataset

    # Step 2: Build model
    print("\n[2/4] Building model...")
    model = build_model(model_type="mobilenet")

    # Step 3: Train model
    print("\n[3/4] Training model...")
    history = train_model(model, X_train, X_val, y_train, y_val)

    # Save model
    save_model(model)

    # Step 4: Evaluate model
    print("\n[4/4] Evaluating model...")
    test_accuracy, y_pred = evaluate_model(model, X_test, y_test)

    # Visualize training
    print("\n📈 Visualizing training history...")
    visualize_training(history)

    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\n📊 Final Results:")
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"  Architecture: MobileNetV2 (Transfer Learning)")
    print(f"  Loss Function: Focal Loss (focused on minority classes)")
    print(f"  Expected Inference Speed: 5-10ms per image")
    print(f"  Memory Usage: ~100MB")
    print("\n💾 Model saved. Ready for inference!")
    print("=" * 80)


if __name__ == "__main__":
    main()
