from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_MODEL_NAME = "face_mask_detection_yolo26m_v1_best.pt"
DEFAULT_CONFIDENCE = 0.45
DEFAULT_IOU = 0.45

# Define the two models to use
MODEL_FASTER_RCNN = MODELS_DIR / "face_mask_detection_faster_rcnn_final.pt"
MODEL_YOLO26M = MODELS_DIR / "face_mask_detection_yolo26m_v1_best.pt"
IMAGE_SIZE = 640
MAX_DETECTIONS = 300
TABLE_COLUMNS = ["label", "confidence", "x1", "y1", "x2", "y2"]
EXAMPLE_IMAGES = [
    PROJECT_ROOT / "datasets" / "face-mask-detection" / "images" / "maksssksksss0.png",
    PROJECT_ROOT / "datasets" / "face-mask-detection" /
    "images" / "maksssksksss25.png",
    PROJECT_ROOT / "datasets" / "face-mask-detection" /
    "images" / "maksssksksss83.png",
]
CLASS_COLORS = {
    "With Mask": (38, 120, 76),
    "Without Mask": (192, 57, 43),
    "Mask Weared Incorrect": (214, 137, 16),
}
FALLBACK_COLOR = (34, 94, 168)
CSS = """
:root {
    --primary-color: #1e40af;
    --success-color: #16a34a;
    --danger-color: #dc2626;
    --warning-color: #d97706;
    --text-dark: #1f2937;
    --text-light: #ffffff;
    --bg-light: #f9fafb;
    --bg-dark: #111827;
    --border-light: #e5e7eb;
}

body {
    background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
    color: var(--text-dark);
}

.gradio-container {
    max-width: 1280px !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

#hero {
    background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
    border: 2px solid #334155;
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 20px rgba(2, 6, 23, 0.45);
}

#hero h1, #hero > div > p {
    color: var(--text-light) !important;
}

#result-panel {
    background: #0f172a;
    border: 2px solid #334155;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 20px rgba(2, 6, 23, 0.45);
}

#result-panel .gradio-markdown,
#result-panel .gradio-markdown h2,
#result-panel .gradio-markdown h3,
#result-panel .gradio-markdown p,
#result-panel label,
#result-panel span,
#result-panel th,
#result-panel td {
    color: #e5e7eb !important;
}

#result-panel table {
    background: #111827 !important;
    border-color: #374151 !important;
}

.gradio-textbox label,
.gradio-slider label,
.gradio-dropdown label,
.gradio-image label,
.gradio-button label {
    color: var(--text-dark) !important;
    font-weight: 600;
    font-size: 0.95rem;
}

.gradio-button-primary {
    background: var(--primary-color) !important;
    border: none !important;
    color: white !important;
    font-weight: 600;
    padding: 12px 20px;
}

.gradio-button-primary:hover {
    background: #1e3a8a !important;
}

.gradio-markdown h2, .gradio-markdown h3 {
    color: var(--text-dark) !important;
    margin-top: 12px;
}

.gradio-markdown {
    color: var(--text-dark) !important;
}

.gr-box {
    border-color: var(--border-light) !important;
    background: #ffffff !important;
}

.gr-input, .gr-textbox textarea, .gr-textbox input {
    color: var(--text-dark) !important;
    border-color: var(--border-light) !important;
}

.gr-slider-container .gr-slider-label {
    color: var(--text-dark) !important;
}

.gr-title {
    color: var(--text-dark) !important;
}

.gr-prose {
    color: var(--text-dark) !important;
}
"""


def discover_models() -> dict[str, tuple[Path, Literal["yolo", "faster_rcnn"]]]:
    """Load the two primary face mask detection models."""
    models = {}

    if MODEL_FASTER_RCNN.exists():
        models["face_mask_detection_faster_rcnn_final.pt"] = (
            MODEL_FASTER_RCNN, "faster_rcnn")

    if MODEL_YOLO26M.exists():
        models["face_mask_detection_yolo26m_v1_best.pt"] = (
            MODEL_YOLO26M, "yolo")

    return models


AVAILABLE_MODELS = discover_models()
if not AVAILABLE_MODELS:
    raise FileNotFoundError(f"No valid checkpoints found in {MODELS_DIR}")

if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS:
    DEFAULT_MODEL_NAME = next(iter(AVAILABLE_MODELS))


def select_device() -> tuple[str | int, str]:
    if torch.cuda.is_available():
        return 0, f"CUDA ({torch.cuda.get_device_name(0)})"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Metal"

    return "cpu", "CPU"


DEVICE, DEVICE_LABEL = select_device()


@lru_cache(maxsize=None)
def load_model(model_name: str):
    """Load model based on type (YOLO or Faster RCNN)."""
    model_path, model_type = AVAILABLE_MODELS[model_name]

    if model_type == "yolo":
        return YOLO(str(model_path))
    elif model_type == "faster_rcnn":
        # Convert device to proper map_location format
        if isinstance(DEVICE, int):
            map_loc = f"cuda:{DEVICE}"
        else:
            map_loc = DEVICE

        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=4)
        checkpoint = torch.load(str(model_path), map_location=map_loc)
        # Handle both direct state dict and checkpoint dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        model.to(DEVICE)
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def resolve_yolo_model_name(preferred_model_name: str) -> str:
    """Pick a YOLO checkpoint, preferring the requested model name."""
    preferred = AVAILABLE_MODELS.get(preferred_model_name)
    if preferred and preferred[1] == "yolo":
        return preferred_model_name

    for model_name, (_, model_type) in AVAILABLE_MODELS.items():
        if model_type == "yolo":
            return model_name

    raise ValueError("No YOLO checkpoint is available for webcam mode.")


def run_native_webcam_detection(model_name: str, confidence_threshold: float) -> None:
    """Run local realtime webcam detection using OpenCV + YOLO."""
    yolo_model_name = resolve_yolo_model_name(model_name)
    model_path, _ = AVAILABLE_MODELS[yolo_model_name]
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (camera index 0).")

    print("Starting native webcam detection. Press 'q' to quit.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.predict(
                source=frame,
                conf=confidence_threshold,
                iou=DEFAULT_IOU,
                imgsz=IMAGE_SIZE,
                max_det=MAX_DETECTIONS,
                device=DEVICE,
                verbose=False,
            )

            names = results[0].names or {}

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item()
                               ) if box.cls is not None else -1
                confidence = float(
                    box.conf[0].item()) if box.conf is not None else 0.0
                label = names.get(class_id, str(class_id))
                display_text = f"{label} {confidence:.2f}"

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )

                (text_width, text_height), baseline = cv2.getTextSize(
                    display_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1,
                )
                text_x = int(x1)
                text_y = int(y1) - 8
                if text_y - text_height < 0:
                    text_y = int(y1) + text_height + 8

                cv2.rectangle(
                    frame,
                    (text_x, text_y - text_height - baseline - 4),
                    (text_x + text_width + 8, text_y + baseline - 4),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    display_text,
                    (text_x + 4, text_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("YOLO Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=TABLE_COLUMNS)


def initial_summary() -> str:
    return (
        "## Upload an image\n\n"
        f"Default checkpoint: **{DEFAULT_MODEL_NAME}**  \n"
        f"Inference device: **{DEVICE_LABEL}**"
    )


def extract_detections(result) -> list[dict[str, object]]:
    names = result.names or {}
    detections: list[dict[str, object]] = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "label": names.get(class_id, str(class_id)),
                "confidence": round(confidence, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    detections.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return detections


def draw_detections(image: Image.Image, detections: list[dict[str, object]]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_width = max(2, round(min(canvas.size) / 180))

    for detection in detections:
        label = str(detection["label"])
        confidence = float(detection["confidence"])
        x1 = int(detection["x1"])
        y1 = int(detection["y1"])
        x2 = int(detection["x2"])
        y2 = int(detection["y2"])
        color = CLASS_COLORS.get(label, FALLBACK_COLOR)
        box = [x1, y1, x2, y2]

        for offset in range(line_width):
            draw.rectangle(
                [
                    box[0] - offset,
                    box[1] - offset,
                    box[2] + offset,
                    box[3] + offset,
                ],
                outline=color,
            )

        tag = f"{label} {confidence:.2%}"
        text_box = draw.textbbox((0, 0), tag, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        tag_x = max(0, x1)
        tag_y = y1 - text_height - 10

        if tag_y < 0:
            tag_y = min(canvas.height - text_height - 6, y1 + 6)

        background = [
            tag_x,
            tag_y,
            min(canvas.width, tag_x + text_width + 10),
            min(canvas.height, tag_y + text_height + 6),
        ]
        draw.rectangle(background, fill=color)
        draw.text((tag_x + 5, tag_y + 3), tag, fill=(255, 255, 255), font=font)

    return canvas


def build_summary(model_name: str, detections: list[dict[str, object]]) -> str:
    message = [
        "## Detection Summary",
        f"**Model:** {model_name}",
        f"**Device:** {DEVICE_LABEL}",
        f"**Faces detected:** {len(detections)}",
    ]

    if detections:
        counts = Counter(str(item["label"]) for item in detections)
        breakdown = ", ".join(
            f"{label}: {count}" for label, count in sorted(counts.items())
        )
        message.append(f"**Breakdown:** {breakdown}")
    else:
        message.append(
            "No detections passed the selected confidence threshold. Try lowering it."
        )

    return "\n\n".join(message)


def run_detection_yolo(
    image: Image.Image,
    model,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[dict[str, object]]:
    """Run YOLO detection on image."""
    rgb_image = image.convert("RGB")
    result = model.predict(
        source=rgb_image,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=IMAGE_SIZE,
        max_det=MAX_DETECTIONS,
        device=DEVICE,
        verbose=False,
    )[0]
    return extract_detections(result)


def run_detection_faster_rcnn(
    image: Image.Image,
    model,
    confidence_threshold: float,
) -> list[dict[str, object]]:
    """Run Faster RCNN detection on image."""
    rgb_image = image.convert("RGB")
    img_array = np.array(rgb_image).transpose(
        2, 0, 1).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)

    detections = []
    names = {0: "With Mask", 1: "Without Mask", 2: "Mask Weared Incorrect"}

    for box, score, label in zip(outputs[0]["boxes"], outputs[0]["scores"], outputs[0]["labels"]):
        if float(score) >= confidence_threshold:
            # Faster R-CNN was trained with background=0, classes=1..3.
            # Shift to 0..2 so labels match dataset class ids used elsewhere.
            cls_idx = int(label.item()) - 1
            if cls_idx < 0 or cls_idx >= len(names):
                continue
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            detections.append({
                "label": names[cls_idx],
                "confidence": round(float(score.item()), 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

    detections.sort(key=lambda x: float(x["confidence"]), reverse=True)
    return detections


def run_detection(
    image: Image.Image | None,
    model_name: str,
    confidence_threshold: float,
    iou_threshold: float,
):
    if image is None:
        return None, empty_table(), initial_summary()

    model = load_model(model_name)
    model_path, model_type = AVAILABLE_MODELS[model_name]

    if model_type == "yolo":
        detections = run_detection_yolo(
            image, model, confidence_threshold, iou_threshold)
    elif model_type == "faster_rcnn":
        detections = run_detection_faster_rcnn(
            image, model, confidence_threshold)
    else:
        detections = []

    rgb_image = image.convert("RGB")
    annotated_image = draw_detections(rgb_image, detections)
    detections_table = pd.DataFrame(detections, columns=TABLE_COLUMNS)

    return annotated_image, detections_table, build_summary(model_name, detections)


def run_realtime_detection(
    image: Image.Image | None,
    model_name: str,
    confidence_threshold: float,
    iou_threshold: float,
):
    """Run detection for webcam stream frames and return light-weight outputs."""
    if image is None:
        return None, initial_summary()

    annotated_image, _, summary = run_detection(
        image,
        model_name,
        confidence_threshold,
        iou_threshold,
    )
    return annotated_image, summary


def build_demo() -> gr.Blocks:
    example_inputs = [str(path) for path in EXAMPLE_IMAGES if path.exists()]

    with gr.Blocks(title="Face Mask Detection Demo", css=CSS) as demo:
        gr.Markdown(
            "# Face Mask Detection Demo\n\n"
            "Upload an image to detect faces and classify each face as With Mask, "
            "Without Mask, or Mask Weared Incorrect.",
            elem_id="hero",
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=DEFAULT_MODEL_NAME,
                label="Checkpoint",
            )
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                step=0.05,
                value=DEFAULT_CONFIDENCE,
                label="Confidence threshold",
            )
            iou_slider = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                step=0.05,
                value=DEFAULT_IOU,
                label="IoU threshold",
            )

        gr.Markdown(
            "Use **Image mode** for detailed analysis with a detections table, "
            "or **Realtime mode** for webcam streaming."
        )

        with gr.Tabs():
            with gr.TabItem("Image mode"):
                with gr.Row():
                    with gr.Column(scale=5):
                        input_image = gr.Image(
                            type="pil",
                            image_mode="RGB",
                            sources=["upload", "clipboard"],
                            label="Input image",
                        )
                        detect_button = gr.Button(
                            "Run detection", variant="primary")

                        if example_inputs:
                            gr.Examples(
                                examples=example_inputs,
                                inputs=input_image,
                                label="Quick examples",
                            )

                    with gr.Column(scale=7, elem_id="result-panel"):
                        output_image = gr.Image(
                            type="pil", label="Annotated result")
                        summary = gr.Markdown(value=initial_summary())
                        detections_table = gr.Dataframe(
                            value=empty_table(),
                            headers=TABLE_COLUMNS,
                            datatype=[
                                "str",
                                "number",
                                "number",
                                "number",
                                "number",
                                "number",
                            ],
                            interactive=False,
                            label="Detected faces",
                        )

                detect_button.click(
                    fn=run_detection,
                    inputs=[
                        input_image,
                        model_dropdown,
                        confidence_slider,
                        iou_slider,
                    ],
                    outputs=[output_image, detections_table, summary],
                )

            with gr.TabItem("Realtime mode"):
                gr.Markdown(
                    "Start webcam and stream predictions frame-by-frame. "
                    "For smoother realtime performance, use the YOLO checkpoint."
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        webcam_input = gr.Image(
                            type="pil",
                            image_mode="RGB",
                            sources=["webcam"],
                            streaming=True,
                            label="Webcam stream",
                        )
                    with gr.Column(scale=7, elem_id="result-panel"):
                        realtime_output_image = gr.Image(
                            type="pil", label="Realtime annotated stream"
                        )
                        realtime_summary = gr.Markdown(value=initial_summary())

                webcam_input.stream(
                    fn=run_realtime_detection,
                    inputs=[
                        webcam_input,
                        model_dropdown,
                        confidence_slider,
                        iou_slider,
                    ],
                    outputs=[realtime_output_image, realtime_summary],
                    show_progress="hidden",
                )

    return demo


def main() -> None:
    native_webcam = os.getenv("WEB_DEMO_NATIVE_WEBCAM",
                              "false").lower() == "true"
    native_model_name = os.getenv("WEB_DEMO_NATIVE_MODEL", DEFAULT_MODEL_NAME)
    native_confidence = float(
        os.getenv("WEB_DEMO_NATIVE_CONFIDENCE", str(DEFAULT_CONFIDENCE))
    )

    if native_webcam:
        run_native_webcam_detection(native_model_name, native_confidence)
        return

    load_model(DEFAULT_MODEL_NAME)
    demo = build_demo()
    host = os.getenv("WEB_DEMO_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_DEMO_PORT", "7860"))
    open_browser = os.getenv("WEB_DEMO_OPEN_BROWSER",
                             "false").lower() == "true"
    share = os.getenv("WEB_DEMO_SHARE", "false").lower() == "true"

    launch_kwargs = {
        "server_name": host,
        "server_port": port,
        "inbrowser": open_browser,
        # Work around Gradio schema generation failures in mixed dependency sets.
        "show_api": False,
        "share": share,
    }

    try:
        demo.launch(**launch_kwargs)
    except ValueError as exc:
        if "localhost is not accessible" not in str(exc):
            raise
        # Auto-fallback for remote/devcontainer sessions where localhost checks fail.
        launch_kwargs["share"] = True
        demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
