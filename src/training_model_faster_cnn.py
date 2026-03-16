import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "datasets" / "face-mask-detection-processed"
TRAIN_IMAGES_DIR = DATASET_DIR / "images" / "train"
TRAIN_LABELS_DIR = DATASET_DIR / "labels" / "train"
VAL_IMAGES_DIR = DATASET_DIR / "images" / "val"
VAL_LABELS_DIR = DATASET_DIR / "labels" / "val"
MODELS_DIR = PROJECT_DIR / "models"
FINAL_MODEL_PATH = MODELS_DIR / "face_mask_detection_faster_rcnn_final.pt"
BEST_MODEL_PATH = MODELS_DIR / "face_mask_detection_faster_rcnn_best.pt"

CLASS_NAMES = {
    0: "With Mask",
    1: "Without Mask",
    2: "Mask Weared Incorrect",
}
NUM_CLASSES = len(CLASS_NAMES) + 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Speed-focused defaults. Set FAST_TRAINING = False for full training.
FAST_TRAINING = True


class FaceMaskDataset(Dataset):
    """Dataset for Faster R-CNN training from YOLO-format labels."""

    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found: {self.label_dir}")

        self.images = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        self.missing_label_files = [
            path.name for path in self.images
            if not (self.label_dir / f"{path.stem}.txt").exists()
        ]

    def __len__(self):
        return len(self.images)

    def _read_targets(self, img_path, img_w, img_h):
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []

        if not label_path.exists():
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label format in {label_path} at line {line_number}: {line}"
                )

            class_id = int(parts[0])
            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class id {class_id} in {label_path} at line {line_number}"
                )

            center_x, center_y, box_w, box_h = map(float, parts[1:])
            x1 = max(0.0, (center_x - box_w / 2.0) * img_w)
            y1 = max(0.0, (center_y - box_h / 2.0) * img_h)
            x2 = min(float(img_w), (center_x + box_w / 2.0) * img_w)
            y2 = min(float(img_h), (center_y + box_h / 2.0) * img_h)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id + 1)

        if not boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        return (
            torch.as_tensor(boxes, dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.int64),
        )

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        boxes, labels = self._read_targets(img_path, img_w, img_h)

        image_tensor = torch.as_tensor(
            image, dtype=torch.float32
        ).permute(2, 0, 1) / 255.0

        if boxes.numel() == 0:
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    """Custom collate function for detection models."""
    return tuple(zip(*batch))


def create_model():
    """Create Faster R-CNN model with a 3-class detection head."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=1,
        min_size=512,
        max_size=768,
    )

    # Freeze most of the backbone to speed up CPU training.
    for name, parameter in model.backbone.named_parameters():
        if "layer4" not in name:
            parameter.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def create_dataloader(dataset, batch_size, shuffle, device):
    """Create a dataloader with conservative worker settings."""
    cpu_count = os.cpu_count() or 0
    num_workers = min(4, cpu_count)
    if device.type == "cpu":
        num_workers = min(2, num_workers)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )


def move_batch_to_device(images, targets, device):
    """Move a detection batch to the selected device."""
    images = [image.to(device) for image in images]
    targets = [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]
    return images, targets


def train_one_epoch(model, dataloader, optimizer, device, epoch_index, num_epochs):
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for batch_index, (images, targets) in enumerate(dataloader, start=1):
        images, targets = move_batch_to_device(images, targets, device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = float(losses.item())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += loss_value

        if batch_index % 10 == 0 or batch_index == len(dataloader):
            print(
                f"  Epoch [{epoch_index}/{num_epochs}] "
                f"Batch [{batch_index}/{len(dataloader)}] "
                f"Loss: {loss_value:.4f}"
            )

    return total_loss / max(len(dataloader), 1)


def calculate_validation_loss(model, dataloader, device):
    """Calculate validation loss.

    Torchvision detection models only return a loss dictionary when targets are
    provided, so validation is intentionally run in train mode under no_grad.
    """
    model.train()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = move_batch_to_device(images, targets, device)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += float(losses.item())

    return total_loss / max(len(dataloader), 1)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_path):
    """Save a training checkpoint with model metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "dataset_dir": str(DATASET_DIR),
    }
    torch.save(checkpoint, save_path)


def print_dataset_summary(train_dataset, val_dataset):
    """Print a brief overview of the train and validation splits."""
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    if train_dataset.missing_label_files:
        print(
            f"Warning: {len(train_dataset.missing_label_files)} train images are missing label files"
        )
    if val_dataset.missing_label_files:
        print(
            f"Warning: {len(val_dataset.missing_label_files)} validation images are missing label files"
        )


def train():
    """Train Faster R-CNN on the processed face mask dataset."""
    if FAST_TRAINING:
        num_epochs = 6
        batch_size = 2
        learning_rate = 0.002
        validate_every = 2
    else:
        num_epochs = 15
        batch_size = 4
        learning_rate = 0.005
        validate_every = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Train images dir: {TRAIN_IMAGES_DIR}")
    print(f"Train labels dir: {TRAIN_LABELS_DIR}")
    print(f"Validation images dir: {VAL_IMAGES_DIR}")
    print(f"Validation labels dir: {VAL_LABELS_DIR}")
    print(f"Fast training mode: {FAST_TRAINING}")

    train_dataset = FaceMaskDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    val_dataset = FaceMaskDataset(VAL_IMAGES_DIR, VAL_LABELS_DIR)
    print_dataset_summary(train_dataset, val_dataset)

    train_loader = create_dataloader(train_dataset, batch_size, True, device)
    val_loader = create_dataloader(val_dataset, batch_size, False, device)

    model = create_model().to(device)

    params = [parameter for parameter in model.parameters()
              if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1,
    )

    print("Starting Faster R-CNN training...")
    best_val_loss = float("inf")

    try:
        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch, num_epochs)

            if epoch % validate_every == 0 or epoch == num_epochs:
                val_loss = calculate_validation_loss(model, val_loader, device)
            else:
                val_loss = float("nan")

            lr_scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            val_loss_is_valid = not torch.isnan(torch.tensor(val_loss))
            if val_loss_is_valid and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    train_loss,
                    val_loss,
                    BEST_MODEL_PATH,
                )
                print(f"  Saved best checkpoint to {BEST_MODEL_PATH}")

        save_checkpoint(
            model,
            optimizer,
            num_epochs,
            train_loss,
            val_loss,
            FINAL_MODEL_PATH,
        )
        print(f"Saved final checkpoint to {FINAL_MODEL_PATH}")

        if BEST_MODEL_PATH.exists():
            best_size_mb = BEST_MODEL_PATH.stat().st_size / (1024 ** 2)
            print(f"Best checkpoint size: {best_size_mb:.2f} MB")
        if FINAL_MODEL_PATH.exists():
            final_size_mb = FINAL_MODEL_PATH.stat().st_size / (1024 ** 2)
            print(f"Final checkpoint size: {final_size_mb:.2f} MB")

        print("Training completed successfully.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        raise SystemExit(1)
    except Exception as exc:
        print(f"\nTraining failed: {exc}")
        raise


if __name__ == "__main__":
    print("\n========== FASTER R-CNN TRAINING START ==========")
    print(f"PID: {os.getpid()}")
    print("=" * 50)
    train()
