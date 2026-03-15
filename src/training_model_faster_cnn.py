import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path
import os
import sys
from xml.etree import ElementTree as ET


class FaceMaskDataset(Dataset):
    """Dataset class for face mask detection with Faster R-CNN"""

    def __init__(self, img_dir, annotations_dir, transforms=None):
        self.img_dir = Path(img_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transforms = transforms
        # Get all images recursively from subdirectories (train, val, test)
        self.images = sorted(list(self.img_dir.glob(
            '**/*.jpg')) + list(self.img_dir.glob('**/*.png')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.as_tensor(
            image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Load bounding boxes and labels from XML annotation
        annot_path = self.annotations_dir / (img_path.stem + '.xml')
        boxes = []
        labels = []

        if annot_path.exists():
            try:
                tree = ET.parse(str(annot_path))
                root = tree.getroot()

                for obj in root.findall('object'):
                    label_name = obj.find('name').text
                    # Map labels: with_mask=1, without_mask=2
                    label = 1 if 'with_mask' in label_name.lower() else 2

                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    labels.append(label)
                    boxes.append([xmin, ymin, xmax, ymax])
            except Exception as e:
                print(f"Error parsing {annot_path}: {e}")

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


def train():
    # Configuration
    DATASET_PATH = '/Users/khoanguyen/Workspace/UIT/face_mask_detection/datasets/face-mask-detection-processed'
    MODEL_SAVE_PATH = '/Users/khoanguyen/Workspace/UIT/face_mask_detection/models/face_mask_detection_faster_rcnn.pt'
    NUM_EPOCHS = 15
    BATCH_SIZE = 4
    LEARNING_RATE = 0.005

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model setup - Faster R-CNN with ResNet50 backbone
    # Load pretrained model with default weights (91 COCO classes)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Replace the classifier with 3 classes (background + with_mask + without_mask)
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    # Dataset and DataLoader
    if os.path.exists(DATASET_PATH):
        img_dir = os.path.join(DATASET_PATH, 'images')
        annot_dir = os.path.join(DATASET_PATH, 'labels')

        if os.path.exists(img_dir) and os.path.exists(annot_dir):
            print(f"Loading dataset from {DATASET_PATH}")
            train_dataset = FaceMaskDataset(
                img_dir=img_dir,
                annotations_dir=annot_dir
            )

            if len(train_dataset) == 0:
                print("Warning: No images found in dataset!")
                return

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                collate_fn=collate_fn
            )

            print(f"Dataset loaded: {len(train_dataset)} images")
        else:
            print(f"Image or annotation directory not found!")
            return
    else:
        print(f"Dataset path not found: {DATASET_PATH}")
        return

    # Optimizer - only optimize parameters with requires_grad=True
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1)

    # Training loop
    print("Starting training...")
    try:
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            batch_count = 0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += losses.item()
                batch_count += 1

                if (batch_idx + 1) % 5 == 0:
                    print(
                        f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {losses.item():.4f}")

            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            lr_scheduler.step()
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Avg Loss: {avg_loss:.4f}")

        # Save model
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        print(f"Saving model to {MODEL_SAVE_PATH}...")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✓ Model saved successfully!")

        # Verify file was created
        if os.path.exists(MODEL_SAVE_PATH):
            file_size = os.path.getsize(
                MODEL_SAVE_PATH) / (1024**2)  # Size in MB
            print(f"✓ Model file verified: {file_size:.2f} MB")

        print(f"\n✓ Training completed successfully!")
        print(f"Model file location: {MODEL_SAVE_PATH}")
        print(f"Script will now exit.")
        sys.stdout.flush()
        return

    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    print("\n========== FASTER R-CNN TRAINING START ==========")
    print(f"PID: {os.getpid()}")
    print("="*50)
    try:
        train()
    finally:
        print("\n========== SCRIPT TERMINATING ==========")
        import sys
        sys.exit(0)
