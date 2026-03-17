# Face Mask Detection Web Demo

This demo launches a Gradio web app for the trained face mask detector. It loads the YOLO checkpoints in `models/` whose filenames contain `yolo`, and it defaults to `face_mask_detection_yolov8m_v2_best.pt`.

## Run

From the project root:

```bash
/home/khoanguyen/workspace/UIT/face_mask_detection/.venv/bin/python web_demo/app.py
```

Or use the wrapper script:

```bash
/home/khoanguyen/workspace/UIT/face_mask_detection/.venv/bin/python web_demo/web_demo.py
```

The app starts on `http://127.0.0.1:7860` by default.

## What it does

- Upload an image or paste one from the clipboard.
- Run face mask detection on the selected checkpoint.
- Return the image with bounding boxes and predicted labels.
- Show a detections table with confidence scores and coordinates.

## Optional environment variables

- `WEB_DEMO_HOST`: bind host, default `127.0.0.1`
- `WEB_DEMO_PORT`: bind port, default `7860`
- `WEB_DEMO_OPEN_BROWSER`: set to `true` to open a browser automatically