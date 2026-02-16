import os
# These MUST be set BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_XLA_FLAGS'] = '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found'

import cv2
import numpy as np
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
DATASET_DIR = PROJECT_DIR / "datasets" / "face-mask-detection"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
IMAGES_DIR = DATASET_DIR / "images"
PROCESSED_DIR = PROJECT_DIR / "datasets" / "face-mask-detection-mobilenet"
MODELS_DIR = PROJECT_DIR / "models"

# Training parameters
IMG_SIZE = (192, 192)  # Balance between speed and accuracy - smaller is faster
BATCH_SIZE = 32  # Increased for stable gradient updates (was 16)
EPOCHS = 150  # Extended for transfer learning fine-tuning (was 100)
LEARNING_RATE = 0.0005  # Lower LR for fine-tuning pre-trained weights (was 0.0001)
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# Class mapping
CLASS_MAPPING = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

# Reverse mapping for display
CLASS_NAMES = {v: k.replace("_", " ").title() for k, v in CLASS_MAPPING.items()}


def parse_xml_annotation(xml_path):
    """Parse XML annotation to get face labels."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find("filename").text
    objects = []
    
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        class_id = CLASS_MAPPING.get(class_name, 1)
        
        objects.append({
            "class_id": class_id,
            "bbox": (xmin, ymin, xmax, ymax)
        })
    
    return filename, objects


def extract_faces_from_xml(image_path, xml_objects):
    """
    Extract face regions using XML bounding boxes directly.
    No MediaPipe dependency needed.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    
    faces = []
    h, w = image.shape[:2]
    
    # Extract faces using XML bounding boxes
    for obj in xml_objects:
        x_min, y_min, x_max, y_max = obj["bbox"]
        
        # Add padding
        x_min = max(0, x_min - 10)
        y_min = max(0, y_min - 10)
        x_max = min(w, x_max + 10)
        y_max = min(h, y_max + 10)
        
        face_roi = image[y_min:y_max, x_min:x_max]
        
        if face_roi.shape[0] > 20 and face_roi.shape[1] > 20:
            faces.append({
                "image": face_roi,
                "class_id": obj["class_id"],
                "bbox": (x_min, y_min, x_max, y_max)
            })
    
    return faces


def prepare_dataset():
    """Extract and prepare face images from the dataset."""
    print("=" * 80)
    print("📊 PREPARING DATASET - Extracting Face Regions from XML Annotations")
    print("=" * 80)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    faces_data = []
    classes_data = []
    
    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    
    print(f"\nProcessing {len(xml_files)} images...\n")
    
    for xml_file in tqdm(xml_files, desc="Extracting faces"):
        try:
            filename, xml_objects = parse_xml_annotation(xml_file)
            image_path = IMAGES_DIR / filename
            
            if not image_path.exists():
                continue
            
            faces = extract_faces_from_xml(image_path, xml_objects)
            
            for face_data in faces:
                face_image = face_data["image"]
                
                # Resize face to standard size
                face_resized = cv2.resize(face_image, IMG_SIZE)
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                
                faces_data.append(face_rgb)
                classes_data.append(face_data["class_id"])
        
        except Exception as e:
            print(f"Error processing {xml_file.name}: {str(e)[:50]}")
            continue
    
    print(f"\n✓ Extracted {len(faces_data)} face regions")
    print(f"  - With Mask: {sum(1 for c in classes_data if c == 0)}")
    print(f"  - Without Mask: {sum(1 for c in classes_data if c == 1)}")
    print(f"  - Mask Weared Incorrect: {sum(1 for c in classes_data if c == 2)}")
    
    # CRITICAL OPTIMIZATION: Keep as uint8 in RAM to save 4x memory
    # Convert to float32 only during batching (done by ImageDataGenerator)
    X = np.array(faces_data, dtype=np.uint8)  # uint8 uses 4x less memory than float32
    y = np.array(classes_data)
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
        random_state=42,
        stratify=y_temp
    )
    
    print(f"\n📂 Dataset Split:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
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
    """Build Transfer Learning model with MobileNetV2 and Focal Loss."""
    print("\n" + "=" * 80)
    print("🔧 BUILDING TRANSFER LEARNING MODEL (MobileNetV2 + Focal Loss)")
    print("=" * 80)
    
    # Resize input to match IMG_SIZE (currently 192x192)
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    
    # Load pre-trained MobileNetV2 from ImageNet
    print("\n📥 Loading pre-trained MobileNetV2 (ImageNet weights)...")
    base_model = keras.applications.MobileNetV2(
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
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
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
    print(f"  Base Model Parameters: ~2.2M (MobileNetV2)")
    print(f"  Custom Head Parameters: {sum([np.prod(w.shape) for w in model.trainable_weights])}")
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
    print(f"  Batch Size:              {BATCH_SIZE}")
    print(f"  Learning Rate:           {LEARNING_RATE}")
    print(f"  Early Stop Patience:     20 epochs")
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
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=(val_data, y_val),
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight_dict,  # Apply class weights with focal loss
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
    
    model_path = MODELS_DIR / "face_mask_mobilenet_best.h5"
    model.save(str(model_path))
    
    print(f"\n✓ Model saved to {model_path}")


def main():
    """Main training pipeline with Transfer Learning + Focal Loss."""
    print("\n" + "=" * 80)
    print("🎯 Face Mask Detection - Transfer Learning Training")
    print("   Architecture: MobileNetV2 + Focal Loss")
    print("   Goal: Detect minority classes (Without Mask, Incorrect Mask)")
    print("=" * 80)
    
    # Step 1: Prepare dataset
    print("\n[1/4] Preparing dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset()
    
    # Step 2: Build model
    print("\n[2/4] Building model...")
    model = build_model(model_type="mobilenet")
    
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
    print(f"  Architecture: MobileNetV2 (Transfer Learning)")
    print(f"  Loss Function: Focal Loss (focused on minority classes)")
    print(f"  Expected Inference Speed: 5-10ms per image")
    print(f"  Memory Usage: ~100MB")
    print("\n💾 Model saved. Ready for inference!")
    print("=" * 80)


if __name__ == "__main__":
    main()
