# Yolo26 + MobileNetV4 Transfer Learning Training Script

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import os
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import timm
from torchvision import transforms
import torch.nn.functional as F

# Detect platform and configure appropriately
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS_OR_LINUX = platform.system() in ["Windows", "Linux"]

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Flag to ensure device detection prints only once
_DEVICE_SETUP_DONE = False


def setup_device():
    """Initialize and configure device (GPU/CPU) for training."""
    global _DEVICE_SETUP_DONE

    # Compute device info (always)
    if IS_MAC:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_info = {
                "type": "Apple Silicon (Metal)", "is_gpu": True, "device": device}
        else:
            device = torch.device("cpu")
            device_info = {"type": "CPU", "is_gpu": False, "device": device}

    elif IS_WINDOWS_OR_LINUX:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_info = {
                "type": f"NVIDIA CUDA (GPU)",
                "is_gpu": True,
                "device": device
            }
        else:
            device = torch.device("cpu")
            device_info = {"type": "CPU", "is_gpu": False, "device": device}
    else:
        device = torch.device("cpu")
        device_info = {"type": "CPU", "is_gpu": False, "device": device}

    # Print device info only on first call
    if not _DEVICE_SETUP_DONE:
        print("=" * 80)
        print("🖥️  DEVICE DETECTION")
        print("=" * 80)
        print(f"Platform: {platform.system()} {platform.release()}")

        if IS_MAC:
            print("🍎 macOS detected - Using Metal Performance Shaders (MPS)")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("✓ Apple Metal GPU available")
            else:
                print("⚠️  No Metal GPU detected. Using CPU (slower).")

        elif IS_WINDOWS_OR_LINUX:
            print("💻 Windows/Linux detected - Using CUDA (NVIDIA GPU)")
            if torch.cuda.is_available():
                print("✓ NVIDIA GPU detected")
            else:
                print("⚠️  No NVIDIA GPU detected. Using CPU (slower).")
        else:
            print(f"⚠️  Unknown platform: {platform.system()}")

        print(f"Selected Device: {device_info['type']}")
        print("=" * 80 + "\n")

        _DEVICE_SETUP_DONE = True

    return device_info


# Initialize DEVICE_INFO by calling setup_device immediately
DEVICE_INFO = setup_device()


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


# Custom Dataset class for face mask images
class FaceMaskDataset(Dataset):
    """Dataset class for face mask detection with augmentation support."""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


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
    Focal Loss for handling class imbalance (PyTorch implementation).
    Focuses on hard-to-classify examples, especially minority classes.

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples). Default: 2
        alpha: Weighting factor for class balance. Default: 0.25
    """
    def focal_loss_fn(outputs, targets):
        # outputs shape: [batch_size, num_classes] - logits
        # targets shape: [batch_size] - class indices

        # Convert logits to probabilities
        probs = F.softmax(outputs, dim=1)

        # Get the probability of the true class
        targets_one_hot = F.one_hot(targets, num_classes=3).float()
        p_t = (probs * targets_one_hot).sum(dim=1)

        # Calculate focal weight
        focal_weight = (1 - p_t) ** gamma

        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')

        # Apply focal weight
        focal_loss_value = alpha * focal_weight * ce_loss

        return focal_loss_value.mean()

    return focal_loss_fn


class MobileNetV4Classifier(nn.Module):
    """MobileNetV4-based classifier with custom head for face mask detection."""

    def __init__(self, num_classes=3, pretrained=True):
        super(MobileNetV4Classifier, self).__init__()

        # Load MobileNetV4 from timm (hybrid medium variant)
        self.backbone = timm.create_model(
            'mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''   # Remove global pooling
        )

        # Get the number of features from backbone
        # MobileNetV4 hybrid medium outputs 1280 features
        num_features = 1280

        # Custom classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract features
        x = self.backbone(x)

        # Global pooling
        x = self.global_pool(x)

        # Classification
        x = self.classifier(x)

        return x


def build_model(model_type="mobilenetv4"):
    """Build Transfer Learning model with MobileNetV4 and Focal Loss."""
    print("\n" + "=" * 80)
    print("🔧 BUILDING TRANSFER LEARNING MODEL (MobileNetV4 + Focal Loss)")
    print("=" * 80)
    print(f"Compute Device: {DEVICE_INFO['type']}")

    print("\n📥 Loading pre-trained MobileNetV4 (ImageNet-12k + ImageNet-1k weights)...")
    model = MobileNetV4Classifier(num_classes=3, pretrained=True)

    # Move model to device
    model = model.to(DEVICE_INFO['device'])

    # Freeze backbone for initial training (can unfreeze later for fine-tuning)
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"   ✓ Loaded MobileNetV4 Hybrid Medium")
    print(f"\n📊 Model Summary:")
    print(
        f"  Base Model Parameters: ~{(total_params - trainable_params)/1e6:.1f}M (MobileNetV4)")
    print(f"  Custom Head Parameters: ~{trainable_params/1e3:.1f}K")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"\n🎯 Loss Function: Focal Loss (γ=2.0, α=0.25)")
    print(f"   → Focuses on hard-to-classify and minority class samples")
    print(f"   → Excellent for imbalanced datasets")

    return model


def train_model(model, X_train, X_val, y_train, y_val):
    """Train the Transfer Learning model with Focal Loss for imbalanced data."""
    print("\n" + "=" * 80)
    print("🚀 TRAINING TRANSFER LEARNING MODEL (MobileNetV4 + Focal Loss)")
    print("=" * 80)

    # Compute class weights to handle imbalanced dataset
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(y_train),
                                         y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    print(f"\n📊 Class Weights (for reference):")
    print(f"  With Mask:               {class_weight_dict.get(0, 1):.2f}x")
    print(f"  Without Mask:            {class_weight_dict.get(1, 1):.2f}x")
    print(f"  Mask Weared Incorrect:   {class_weight_dict.get(2, 1):.2f}x")

    # Data augmentation transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.15, 0.15)),
        transforms.RandomResizedCrop(IMG_SIZE[0], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = FaceMaskDataset(
        X_train, y_train, transform=train_transform)
    val_dataset = FaceMaskDataset(X_val, y_val, transform=val_transform)

    # Create data loaders - num_workers=0 on macOS to avoid multiprocessing re-imports
    num_workers = 0 if IS_MAC else 2
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7)

    # Loss function
    criterion = focal_loss(gamma=2.0, alpha=0.25)

    print(f"\nTraining Configuration:")
    print(f"  Epochs:                  {EPOCHS}")
    print(
        f"  Batch Size:              {BATCH_SIZE} (optimized for {DEVICE_INFO['type']})")
    print(f"  Learning Rate:           {LEARNING_RATE}")
    print(f"  Early Stop Patience:     20 epochs")
    print(f"  Compute Device:          {DEVICE_INFO['type']}")
    print(f"  Loss Function:           Focal Loss (γ=2.0, α=0.25)")
    print(f"  Data Augmentation:       rotation ±20°, shifts ±15%, zoom ±20%, flip")

    # Training loop
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(
                DEVICE_INFO['device']), targets.to(DEVICE_INFO['device'])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(
                    DEVICE_INFO['device']), targets.to(DEVICE_INFO['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Update history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # Print progress
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Restore best model
    model.load_state_dict(best_model_state)

    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    print("\n" + "=" * 80)
    print("📊 EVALUATING MODEL")
    print("=" * 80)

    # Create test dataset and loader
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    test_dataset = FaceMaskDataset(X_test, y_test, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Evaluation
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []

    criterion = focal_loss(gamma=2.0, alpha=0.25)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(
                DEVICE_INFO['device']), targets.to(DEVICE_INFO['device'])
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = test_correct / test_total

    print(f"\n🎯 Test Results:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred_classes = np.array(all_preds)
    y_test_array = np.array(all_targets)

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test_array, y_pred_classes,
          target_names=list(CLASS_NAMES.values())))

    # Confusion matrix
    cm = confusion_matrix(y_test_array, y_pred_classes)
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

    model_path = MODELS_DIR / "face_mask_mobilenetv4_yolo26m_best.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'MobileNetV4',
    }, str(model_path))

    print(f"\n✓ Model saved to {model_path}")


def main():
    """Main training pipeline with Transfer Learning + Focal Loss."""
    print("\n" + "=" * 80)
    print("🎯 Face Mask Detection - Transfer Learning Training")
    print("   Architecture: MobileNetV4 (timm) + Focal Loss")
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

    # Step 4: Evaluate model
    print("\n[4/4] Evaluating model...")
    test_accuracy, y_pred = evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model)

    # Visualize training
    print("\n📈 Visualizing training history...")
    visualize_training(history)

    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\n📊 Final Results:")
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"  Architecture: MobileNetV4 Hybrid Medium (Transfer Learning)")
    print(f"  Loss Function: Focal Loss (focused on minority classes)")
    print(f"  Expected Inference Speed: 8-15ms per image")
    print(f"  Memory Usage: ~150MB")
    print("\n💾 Model saved. Ready for inference!")
    print("=" * 80)


if __name__ == "__main__":
    main()
