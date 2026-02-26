"""
Visualization script for MobileNet+YOLOv8m model evaluation.

This script evaluates the face_mask_mobilenet_yolov8m_best.h5 model on the test set
and generates comprehensive visualizations including:
- Per-class metrics (precision, recall, F1-score)
- Confusion matrix
- Overall metrics comparison
- Detailed classification report
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def setup_paths():
    """Setup and validate paths."""
    BASE_DIR = Path('/Users/khoanguyen/Workspace/UIT/face_mask_detection')
    MOBILENET_YOLOV8_MODEL_PATH = BASE_DIR / \
        'models' / 'face_mask_mobilenet_yolov8m_best.h5'
    TEST_IMAGES_DIR = BASE_DIR / 'datasets' / \
        'face-mask-detection-processed' / 'images' / 'test'
    TEST_LABELS_DIR = BASE_DIR / 'datasets' / \
        'face-mask-detection-processed' / 'labels' / 'test'

    print("=" * 80)
    print("PATH VALIDATION")
    print("=" * 80)
    print(
        f"MobileNet+YOLOv8m model exists: {MOBILENET_YOLOV8_MODEL_PATH.exists()}")
    print(f"Test images directory exists: {TEST_IMAGES_DIR.exists()}")
    print(f"Test labels directory exists: {TEST_LABELS_DIR.exists()}")
    print("=" * 80 + "\n")

    return MOBILENET_YOLOV8_MODEL_PATH, TEST_IMAGES_DIR, TEST_LABELS_DIR


def load_model(model_path):
    """Load the MobileNet+YOLOv8m model."""
    print("Loading MobileNet+YOLOv8m model...")
    try:
        model = keras.models.load_model(str(model_path), compile=False)
        print("✓ Model loaded successfully!\n")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}\n")
        return None


def get_test_images(test_images_dir):
    """Get list of test images."""
    test_images = sorted(
        list(test_images_dir.glob('*.jpg')) +
        list(test_images_dir.glob('*.png'))
    )
    print(f"Found {len(test_images)} test images\n")
    return test_images


def read_yolo_label(label_path):
    """Read YOLO format label file."""
    labels = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    labels.append((class_id, *bbox))
    return labels


def convert_yolo_to_pixel_coords(bbox_yolo, img_w, img_h):
    """Convert YOLO format bbox to pixel coordinates."""
    center_x, center_y, bbox_w, bbox_h = bbox_yolo

    # Convert from normalized to pixel coordinates
    pixel_center_x = center_x * img_w
    pixel_center_y = center_y * img_h
    pixel_width = bbox_w * img_w
    pixel_height = bbox_h * img_h

    # Calculate top-left and bottom-right corners
    x1 = int(pixel_center_x - pixel_width / 2)
    y1 = int(pixel_center_y - pixel_height / 2)
    x2 = int(pixel_center_x + pixel_width / 2)
    y2 = int(pixel_center_y + pixel_height / 2)

    # Clip to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return x1, y1, x2, y2


def evaluate_model(model, test_images, test_labels_dir, img_size=192):
    """Evaluate model on test set."""
    print("=" * 80)
    print("EVALUATING MOBILENET+YOLOV8M MODEL")
    print("=" * 80)
    print("Running predictions on test set using ground truth bounding boxes...\n")

    y_true = []
    y_pred = []
    confidences = []

    for img_path in tqdm(test_images):
        # Get ground truth labels
        label_path = test_labels_dir / (img_path.stem + '.txt')
        gt_labels = read_yolo_label(label_path)

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Process each ground truth bounding box
        for gt_class, center_x, center_y, bbox_w, bbox_h in gt_labels:
            # Convert YOLO format to pixel coordinates
            x1, y1, x2, y2 = convert_yolo_to_pixel_coords(
                (center_x, center_y, bbox_w, bbox_h), img_w, img_h
            )

            # Crop face region
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Preprocess for MobileNet
            face_resized = cv2.resize(face, (img_size, img_size))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)

            # Predict
            pred = model.predict(face_batch, verbose=0)
            pred_class = np.argmax(pred[0])
            confidence = np.max(pred[0])

            y_true.append(gt_class)
            y_pred.append(pred_class)
            confidences.append(confidence)

    print(f"\n✓ Evaluation complete!")
    print(f"  Total predictions: {len(y_pred)}")
    print(f"  Total ground truth: {len(y_true)}\n")

    return np.array(y_true), np.array(y_pred), np.array(confidences)


def calculate_metrics(y_true, y_pred, class_names):
    """Calculate evaluation metrics."""
    print("=" * 80)
    print("MODEL METRICS")
    print("=" * 80)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(
        y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(
        y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print(f"\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            print(f"  {class_name}:")
            print(f"    Precision: {precision_per_class[i]:.4f}")
            print(f"    Recall:    {recall_per_class[i]:.4f}")
            print(f"    F1-Score:  {f1_per_class[i]:.4f}")

    print("=" * 80 + "\n")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title('MobileNet+YOLOv8m Confusion Matrix',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_per_class_metrics(metrics, class_names):
    """Plot per-class metrics."""
    print("Generating per-class metrics plot...")
    precision_vals = metrics['precision_per_class']
    recall_vals = metrics['recall_per_class']
    f1_vals = metrics['f1_per_class']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Precision
    axes[0].bar(class_names, precision_vals, alpha=0.8, color='steelblue')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision by Class')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(precision_vals):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    # Recall
    axes[1].bar(class_names, recall_vals, alpha=0.8, color='seagreen')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Recall by Class')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(recall_vals):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    # F1-Score
    axes[2].bar(class_names, f1_vals, alpha=0.8, color='coral')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_title('F1-Score by Class')
    axes[2].set_ylim([0, 1.1])
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1_vals):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_overall_metrics(metrics):
    """Plot overall metrics."""
    print("Generating overall metrics plot...")
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1']
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics_names, values, alpha=0.8, color='steelblue')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('MobileNet+YOLOv8m Overall Performance',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report."""
    print("=" * 80)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(y_true, y_pred,
          target_names=class_names, zero_division=0))
    print("=" * 80 + "\n")


def main():
    """Main evaluation pipeline."""
    print("\n" + "=" * 80)
    print("MOBILENET+YOLOV8M MODEL EVALUATION")
    print("=" * 80 + "\n")

    # Class names
    CLASS_NAMES = ['With Mask', 'Without Mask', 'Mask Weared Incorrect']

    # Setup paths
    model_path, test_images_dir, test_labels_dir = setup_paths()

    # Load model
    model = load_model(model_path)
    if model is None:
        return

    # Get test images
    test_images = get_test_images(test_images_dir)
    if len(test_images) == 0:
        print("❌ No test images found!\n")
        return

    # Evaluate model
    y_true, y_pred, confidences = evaluate_model(
        model, test_images, test_labels_dir
    )

    if len(y_pred) == 0:
        print("❌ No predictions generated!\n")
        return

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, CLASS_NAMES)

    # Generate visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")

    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
    plot_per_class_metrics(metrics, CLASS_NAMES)
    plot_overall_metrics(metrics)

    # Print classification report
    print_classification_report(y_true, y_pred, CLASS_NAMES)

    print("=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
