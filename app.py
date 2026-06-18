"""Hugging Face Spaces Gradio app for driver cellphone-use YOLO inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = Path("models/best.pt")
DEVICE = "cpu"
DETECTION_COLUMNS = ["class_name", "confidence", "x1", "y1", "x2", "y2"]
ALERT_CLASS_NAMES = {"Cellphone-in-drivers", "Cellphone-in-driver"}

MODEL = None
MODEL_ERROR: str | None = None


def load_model() -> Any:
    """Load YOLO model once for inference."""
    global MODEL_ERROR
    if not DEFAULT_WEIGHTS.exists():
        MODEL_ERROR = (
            "models/best.pt was not found. Upload trained weights from Kaggle to "
            "models/best.pt before running this demo."
        )
        return None

    try:
        from ultralytics import YOLO

        return YOLO(str(DEFAULT_WEIGHTS))
    except Exception as exc:
        MODEL_ERROR = f"Could not load model: {exc}"
        return None


MODEL = load_model()


def empty_table() -> pd.DataFrame:
    """Return an empty detection table with stable columns."""
    return pd.DataFrame(columns=DETECTION_COLUMNS)


def model_status() -> str:
    """Return model status markdown."""
    if MODEL is None:
        return f"**Model status:** {MODEL_ERROR}"
    return f"**Model status:** loaded `{DEFAULT_WEIGHTS}` on CPU."


def alert_status(rows: list[dict[str, Any]]) -> str:
    """Return alert markdown for driver cellphone-use detections."""
    has_alert = any(str(row.get("class_name")) in ALERT_CLASS_NAMES for row in rows)
    if has_alert:
        return (
            "### Alert: Cellphone-in-drivers detected\n"
            "The model found a driver cellphone-use detection in this frame."
        )
    return "### Alert status\nNo driver cellphone-use detection found."


def detect(
    image: Any,
    confidence: float,
    image_size: int,
    iou_threshold: float = 0.7,
) -> tuple[Any, pd.DataFrame, str, str]:
    """Run image inference and return annotated image, detections, alert, and performance text."""
    if image is None:
        return None, empty_table(), alert_status([]), "Provide an image to run inference."
    if MODEL is None:
        return image, empty_table(), alert_status([]), MODEL_ERROR or "Model is not available."

    start = time.perf_counter()
    image_array = to_rgb_array(image)
    results = MODEL.predict(
        source=image_array,
        conf=float(confidence),
        imgsz=int(image_size),
        iou=float(iou_threshold),
        device=DEVICE,
        verbose=False,
    )
    result = results[0]

    rows = []
    names = getattr(result, "names", None) or getattr(MODEL, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            rows.append(
                {
                    "class_name": str(names.get(class_id, class_id)),
                    "confidence": round(float(box.conf.item()), 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            )

    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    response_time = time.perf_counter() - start
    speed = getattr(result, "speed", {}) if result is not None else {}

    performance = [
        "Device: CPU",
        f"Image size: {image_size}",
        f"Confidence threshold: {confidence:.2f}",
        f"IoU threshold: {iou_threshold:.2f}",
        f"Detections: {len(rows)}",
        f"End-to-end response time: {response_time:.2f}s",
    ]
    if isinstance(speed, dict):
        if speed.get("preprocess") is not None:
            performance.append(f"Preprocess time: {speed['preprocess']:.2f}ms")
        if speed.get("inference") is not None:
            performance.append(f"Model inference time: {speed['inference']:.2f}ms")
        if speed.get("postprocess") is not None:
            performance.append(f"Postprocess time: {speed['postprocess']:.2f}ms")
    performance.append(
        "Note: latency depends on Hugging Face Space hardware and current server load."
    )

    table = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
    return annotated_rgb, table, alert_status(rows), "\n".join(performance)


def to_rgb_array(image: Any) -> np.ndarray:
    """Convert a Gradio PIL or numpy image input to RGB numpy array."""
    if hasattr(image, "convert"):
        return np.array(image.convert("RGB"))
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return np.stack([image_array] * 3, axis=-1)
    if image_array.shape[-1] == 4:
        return image_array[:, :, :3]
    return image_array


with gr.Blocks(title="Driver Cellphone Use YOLO Detector") as demo:
    gr.Markdown("# Driver Cellphone Use YOLO Detector")
    gr.Markdown(
        "Fine-tuned YOLO model for detecting driver cellphone-use scenes, drivers, phones, and steering wheels."
    )
    gr.Markdown(
        "The detection table shows all four classes; the alert is raised only for `Cellphone-in-drivers`."
    )
    gr.Markdown(model_status())

    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        sources=["upload", "webcam", "clipboard"],
                        type="pil",
                        label="Image",
                    )
                    confidence_input = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="Confidence threshold",
                    )
                    image_size_input = gr.Dropdown(
                        choices=[416, 512, 640, 768],
                        value=640,
                        label="Image size",
                    )
                    iou_input = gr.Slider(
                        minimum=0.3,
                        maximum=0.9,
                        value=0.7,
                        step=0.05,
                        label="IoU threshold",
                    )
                    run_button = gr.Button("Run detection")
                with gr.Column():
                    image_output = gr.Image(type="numpy", label="Annotated image")
                    detection_output = gr.Dataframe(
                        headers=DETECTION_COLUMNS,
                        datatype=["str", "number", "number", "number", "number", "number"],
                        label="Detections",
                    )
                    alert_output = gr.Markdown()
                    performance_output = gr.Textbox(label="Performance summary", lines=9)

            run_button.click(
                detect,
                inputs=[image_input, confidence_input, image_size_input, iou_input],
                outputs=[image_output, detection_output, alert_output, performance_output],
            )

        with gr.Tab("Webcam live"):
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="pil",
                        label="Webcam",
                    )
                    webcam_confidence_input = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="Confidence threshold",
                    )
                    webcam_image_size_input = gr.Dropdown(
                        choices=[416, 512, 640, 768],
                        value=640,
                        label="Image size",
                    )
                    webcam_iou_input = gr.Slider(
                        minimum=0.3,
                        maximum=0.9,
                        value=0.7,
                        step=0.05,
                        label="IoU threshold",
                    )
                with gr.Column():
                    webcam_output = gr.Image(type="numpy", label="Live detections")
                    webcam_detection_output = gr.Dataframe(
                        headers=DETECTION_COLUMNS,
                        datatype=["str", "number", "number", "number", "number", "number"],
                        label="Detections",
                    )
                    webcam_alert_output = gr.Markdown()
                    webcam_performance_output = gr.Textbox(
                        label="Performance summary",
                        lines=9,
                    )

            webcam_input.stream(
                detect,
                inputs=[
                    webcam_input,
                    webcam_confidence_input,
                    webcam_image_size_input,
                    webcam_iou_input,
                ],
                outputs=[
                    webcam_output,
                    webcam_detection_output,
                    webcam_alert_output,
                    webcam_performance_output,
                ],
                stream_every=0.75,
                concurrency_limit=1,
                queue=False,
                show_progress="hidden",
            )


if __name__ == "__main__":
    demo.launch()
