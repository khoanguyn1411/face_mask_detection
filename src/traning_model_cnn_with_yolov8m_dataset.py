import os
# These MUST be set BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_XLA_FLAGS'] = '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found'

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Set seed for reproducibility  
tf.random.set_seed(42)
np.random.seed(42)

# GPU memory growth - CRITICAL for GTX 1650 to avoid OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"Note: {e}")

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "datasets" / "face-mask-detection-processed"
IMAGES_DIR = {"train": DATASET_DIR / "images" / "train", 
              "val": DATASET_DIR / "images" / "val",
              "test": DATASET_DIR / "images" / "test"}
LABELS_DIR = {"train": DATASET_DIR / "labels" / "train",
              "val": DATASET_DIR / "labels" / "val",
              "test": DATASET_DIR / "labels" / "test"}
MODELS_DIR = PROJECT_DIR / "models"

# Training parameters
IMG_SIZE = (128, 128)  # Balance between speed and accuracy
BATCH_SIZE = 16  # Safe with uint8 + ImageDataGenerator
EPOCHS = 100  # Extended for better minority class learning
LEARNING_RATE = 0.001

# Fix the typo in IMAGES_DIR configuration
IMAGES_DIR = {"train": DATASET_DIR / "images" / "train", 
              "val": DATASET_DIR / "images" / "val",
              "test": DATASET_DIR / "images" / "test"}
LABELS_DIR = {"train": DATASET_DIR / "labels" / "train",
              "val": DATASET_DIR / "labels" / "val",
              "test": DATASET_DIR / "labels" / "test"}

# Class mapping
CLASS_MAPPING = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

# Reverse mapping for display
CLASS_NAMES = {v: k.replace("_", " ").title() for k, v in CLASS_MAPPING.items()}


def load_label_from_yolo(label_path):
    """Load class ID from YOLO format label file.
    
    YOLO format: class_id x_center y_center width height
    We only need the class_id (first value)
    """
    try:
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if line:
                class_id = int(line.split()[0])
                return class_id
    except:
        pass
    return None


def load_images_and_labels(split="train"):
    """Load images and labels from pre-processed dataset.
    
    Args:
        split: 'train', 'val', or 'test'
    
    Returns:
        images: list of image arrays (128x128x3)
        labels: list of class IDs (0, 1, 2)
    """
    images_dir = IMAGES_DIR[split]
    labels_dir = LABELS_DIR[split]
    
    images = []
    labels = []
    
    # Get all image files
    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    
    print(f"  Loading {split} images from {images_dir}...")
    
    for img_path in tqdm(image_files, desc=f"Loading {split}"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Resize to standard size
        img_resized = cv2.resize(img, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Load corresponding label
        label_path = labels_dir / (img_path.stem + ".txt")
        class_id = load_label_from_yolo(label_path)
        
        if class_id is not None and class_id in CLASS_MAPPING.values():
            images.append(img_rgb)
            labels.append(class_id)
    
    return np.array(images, dtype=np.uint8), np.array(labels)


def prepare_dataset():
    """Load pre-processed dataset from face-mask-detection-processed."""
    print("=" * 80)
    print("📊 PREPARING DATASET - Loading from Pre-processed Directory")
    print("=" * 80)
    
    print(f"\nDataset Location: {DATASET_DIR}")
    print(f"Structure: train/val/test splits already organized\n")
    
    # Load each split
    X_train, y_train = load_images_and_labels("train")
    X_val, y_val = load_images_and_labels("val")
    X_test, y_test = load_images_and_labels("test")
    
    # Print statistics
    print(f"\n✓ Dataset Loaded Successfully")
    print(f"\n  Train Split: {len(X_train)} images")
    print(f"    - With Mask:              {sum(1 for c in y_train if c == 0)}")
    print(f"    - Without Mask:           {sum(1 for c in y_train if c == 1)}")
    print(f"    - Mask Weared Incorrect:  {sum(1 for c in y_train if c == 2)}")
    
    print(f"\n  Validation Split: {len(X_val)} images")
    print(f"    - With Mask:              {sum(1 for c in y_val if c == 0)}")
    print(f"    - Without Mask:           {sum(1 for c in y_val if c == 1)}")
    print(f"    - Mask Weared Incorrect:  {sum(1 for c in y_val if c == 2)}")
    
    print(f"\n  Test Split: {len(X_test)} images")
    print(f"    - With Mask:              {sum(1 for c in y_test if c == 0)}")
    print(f"    - Without Mask:           {sum(1 for c in y_test if c == 1)}")
    print(f"    - Mask Weared Incorrect:  {sum(1 for c in y_test if c == 2)}")
    
    print(f"\n  Total: {len(X_train) + len(X_val) + len(X_test)} images")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Data Type: uint8 (memory efficient)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(model_type="custom"):
    """Build CNN model for mask classification."""
    print("\n" + "=" * 80)
    print("🔧 BUILDING CNN MODEL")
    print("=" * 80)
    
    if model_type == "custom":
        print("\nUsing Optimized CNN for GTX 1650 (Memory-efficient with improved power)")
        model = keras.Sequential([
            layers.Input(shape=(128, 128, 3)),
            
            # Block 1 - Enhanced with more filters (safe with GlobalAveragePooling)
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2 - More discriminative power
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3 - Final feature extraction
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # GlobalAveragePooling2D drastically reduces memory while preserving features
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(CLASS_MAPPING), activation='softmax')
        ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n📊 Model Summary:")
    print(f"  Total Parameters: {model.count_params():,}")
    
    return model


def train_model(model, X_train, X_val, y_train, y_val):
    """Train the CNN model with class weights for imbalanced data."""
    print("\n" + "=" * 80)
    print("🚀 TRAINING CNN MODEL")
    print("=" * 80)
    
    # Compute class weights to handle imbalanced dataset
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight('balanced', 
                                         classes=np.unique(y_train),
                                         y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print(f"\n📊 Class Weights (for handling imbalance):")
    print(f"  With Mask:               {class_weight_dict.get(0, 1):.2f}x")
    print(f"  Without Mask:            {class_weight_dict.get(1, 1):.2f}x")
    print(f"  Mask Weared Incorrect:   {class_weight_dict.get(2, 1):.2f}x")
    
    # Enhanced callbacks with increased patience for minority class learning
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased from 10 for better minority class convergence
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs:            {EPOCHS}")
    print(f"  Batch Size:        {BATCH_SIZE}")
    print(f"  Learning Rate:     {LEARNING_RATE}")
    print(f"  Early Stop Patience: 15 epochs")
    print(f"  Data Augmentation: rotation, shifts, zoom, flip")
    
    # Enhanced data augmentation to help minority class
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
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=(val_data, y_val),
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight_dict,  # Apply class weights
        steps_per_epoch=len(X_train) // BATCH_SIZE
    )
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    print("\n" + "=" * 80)
    print("📊 EVALUATING MODEL")
    print("=" * 80)
    
    # Convert test data to float32 for evaluation
    X_test_float = X_test.astype(np.float32) / 255.0
    test_loss, test_accuracy = model.evaluate(X_test_float, y_test, verbose=0, batch_size=BATCH_SIZE)
    
    print(f"\n🎯 Test Results:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_pred = model.predict(X_test_float, verbose=0, batch_size=BATCH_SIZE)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=list(CLASS_NAMES.values())))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    print(f"\n🔢 Confusion Matrix:")
    print(cm)
    
    return test_accuracy, y_pred_classes


def visualize_training(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
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
    
    model_path = MODELS_DIR / "face_mask_cnn_with_yolov8m_dataset_best.h5"
    model.save(str(model_path))
    
    print(f"\n✓ Model saved to {model_path}")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 80)
    print("🎯 MediaPipe + CNN Face Mask Detection Training")
    print("=" * 80)
    
    # Step 1: Prepare dataset
    print("\n[1/4] Preparing dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset()
    
    # Step 2: Build model
    print("\n[2/4] Building model...")
    model = build_model(model_type="custom")
    
    # Step 3: Train model
    print("\n[3/4] Training model...")
    history = train_model(model, X_train, X_val, y_train, y_val)
    
    # Step 4: Evaluate model
    print("\n[4/4] Evaluating model...")
    test_accuracy, y_pred = evaluate_model(model, X_test, y_test)
    
    # Visualize training
    print("\n📈 Visualizing training history...")
    visualize_training(history)
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\n📊 Final Results:")
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"  Expected Inference Speed: 8-15ms per image")
    print(f"  Memory Usage: ~200MB")
    print("\n💾 Model saved. Ready for inference with MediaPipe!")
    print("=" * 80)


if __name__ == "__main__":
    main()
