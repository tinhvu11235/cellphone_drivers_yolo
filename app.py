"""Hugging Face Spaces Gradio app for driver cellphone-use YOLO inference."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

from src.driver_phone_heuristic import infer_driver_phone_usage


DEFAULT_WEIGHTS = Path("models/best.pt")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("outputs/ultralytics_config")))
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/matplotlib_config")))

DEVICE = "cpu"
DETECTION_COLUMNS = ["class_name", "confidence", "x1", "y1", "x2", "y2"]
ALERT_CLASS_NAMES = {"Cellphone-in-drivers", "Cellphone-in-driver"}
DRIVER_CLASS_NAMES = {"driver"}
PHONE_CLASS_NAMES = {"phone"}
WHEEL_CLASS_NAMES = {"wheel"}
DRIVER_PAD_X = 0.2
DRIVER_PAD_Y = 0.12
WHEEL_PAD = 0.3

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


def class_name(row: dict[str, Any]) -> str:
    """Return a normalized detection class name."""
    return str(row.get("class_name", "")).strip()


def has_class(row: dict[str, Any], names: set[str]) -> bool:
    """Return True when a detection row belongs to one of the class names."""
    return class_name(row).lower() in {name.lower() for name in names}


def box_area(row: dict[str, Any]) -> float:
    """Return bbox area in pixels."""
    return max(0.0, float(row["x2"]) - float(row["x1"])) * max(0.0, float(row["y2"]) - float(row["y1"]))


def box_center(row: dict[str, Any]) -> tuple[float, float]:
    """Return bbox center in pixels."""
    return ((float(row["x1"]) + float(row["x2"])) / 2, (float(row["y1"]) + float(row["y2"])) / 2)


def expand_box(row: dict[str, Any], pad_x: float, pad_y: float) -> dict[str, float]:
    """Expand a bbox by a ratio of its width and height."""
    width = max(0.0, float(row["x2"]) - float(row["x1"]))
    height = max(0.0, float(row["y2"]) - float(row["y1"]))
    return {
        "x1": float(row["x1"]) - width * pad_x,
        "y1": float(row["y1"]) - height * pad_y,
        "x2": float(row["x2"]) + width * pad_x,
        "y2": float(row["y2"]) + height * pad_y,
    }


def contains_point(box: dict[str, float] | dict[str, Any], point: tuple[float, float]) -> bool:
    """Return True when a point is inside a bbox."""
    x, y = point
    return float(box["x1"]) <= x <= float(box["x2"]) and float(box["y1"]) <= y <= float(box["y2"])


def intersection_area(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return bbox intersection area."""
    x1 = max(float(first["x1"]), float(second["x1"]))
    y1 = max(float(first["y1"]), float(second["y1"]))
    x2 = min(float(first["x2"]), float(second["x2"]))
    y2 = min(float(first["y2"]), float(second["y2"]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def phone_overlap_ratio(phone: dict[str, Any], driver: dict[str, Any]) -> float:
    """Return how much of the phone box overlaps the driver box."""
    area = box_area(phone)
    if area <= 0:
        return 0.0
    return intersection_area(phone, driver) / area


def relative_driver_y(phone: dict[str, Any], driver: dict[str, Any]) -> float:
    """Return phone center Y position relative to driver height."""
    _, phone_y = box_center(phone)
    driver_height = max(1.0, float(driver["y2"]) - float(driver["y1"]))
    return (phone_y - float(driver["y1"])) / driver_height


def near_any_wheel(phone: dict[str, Any], wheels: list[dict[str, Any]]) -> bool:
    """Return True when a phone is close to a steering wheel detection."""
    phone_center = box_center(phone)
    for wheel in wheels:
        expanded_wheel = expand_box(wheel, WHEEL_PAD, WHEEL_PAD)
        if contains_point(expanded_wheel, phone_center) or intersection_area(phone, expanded_wheel) > 0:
            return True
    return False


def driver_phone_risk(
    driver: dict[str, Any],
    phone: dict[str, Any],
    wheels: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    """Score the risk that a phone belongs to the driver."""
    score = 0
    reasons = []
    phone_center = box_center(phone)
    expanded_driver = expand_box(driver, DRIVER_PAD_X, DRIVER_PAD_Y)
    overlap = phone_overlap_ratio(phone, driver)
    rel_y = relative_driver_y(phone, driver)

    if contains_point(driver, phone_center):
        score += 4
        reasons.append("phone center is inside the driver box")
    elif overlap >= 0.2:
        score += 4
        reasons.append("phone overlaps the driver box")
    elif overlap >= 0.05:
        score += 3
        reasons.append("phone partially overlaps the driver box")
    elif contains_point(expanded_driver, phone_center):
        score += 2
        reasons.append("phone is close to the driver box")

    if contains_point(expanded_driver, phone_center) and rel_y <= 0.45:
        score += 2
        reasons.append("phone is near the upper driver area")
    if contains_point(expanded_driver, phone_center) and 0.35 <= rel_y <= 1.1:
        score += 2
        reasons.append("phone is in the driver's looking-down area")
    if near_any_wheel(phone, wheels) and contains_point(expanded_driver, phone_center):
        score += 1
        reasons.append("phone is near the steering wheel")

    return score, reasons


def best_driver_phone_alert(rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Return the strongest driver-phone alert level and supporting reasons."""
    if any(has_class(row, ALERT_CLASS_NAMES) for row in rows):
        return "high", ["model predicted the legacy alert class"]

    drivers = [row for row in rows if has_class(row, DRIVER_CLASS_NAMES)]
    phones = [row for row in rows if has_class(row, PHONE_CLASS_NAMES)]
    wheels = [row for row in rows if has_class(row, WHEEL_CLASS_NAMES)]
    if not drivers or not phones:
        return "none", []

    best_score = 0
    best_reasons: list[str] = []
    for driver in drivers:
        for phone in phones:
            score, reasons = driver_phone_risk(driver, phone, wheels)
            if score > best_score:
                best_score = score
                best_reasons = reasons

    if best_score >= 6:
        return "high", best_reasons
    if best_score >= 3:
        return "medium", best_reasons
    return "none", []


def rows_to_heuristic_detections(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert app table rows to the shared heuristic input format."""
    return [
        {
            "class_name": row["class_name"],
            "confidence": row["confidence"],
            "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
        }
        for row in rows
    ]


def alert_status(rows: list[dict[str, Any]], image_shape: tuple[int, ...] | None = None) -> str:
    """Return alert markdown for driver cellphone-use detections."""
    if image_shape is None:
        return "### Alert status\nNo driver cellphone-use detection found."

    prediction = infer_driver_phone_usage(rows_to_heuristic_detections(rows), image_shape)
    risk_score = float(prediction["risk_score"])
    reason = prediction["reason"]
    if prediction["driver_using_phone"]:
        return (
            "### Alert: potential driver cellphone use\n"
            f"Rule reason: `{reason}`. Risk score: {risk_score:.2f}."
        )
    return (
        "### Alert status\n"
        f"No driver cellphone-use detection found. Rule reason: `{reason}`. Risk score: {risk_score:.2f}."
    )


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
        source=to_model_source(image, image_array),
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
    return annotated_rgb, table, alert_status(rows, image_array.shape), "\n".join(performance)


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


def to_model_source(image: Any, image_array: np.ndarray) -> Any:
    """Return an Ultralytics input with color channels matching its loader."""
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return image_array[:, :, ::-1]


with gr.Blocks(title="Driver Cellphone Use YOLO Detector") as demo:
    gr.Markdown("# Driver Cellphone Use YOLO Detector")
    gr.Markdown(
        "Fine-tuned YOLO model for detecting driver cellphone-use scenes, drivers, phones, and steering wheels."
    )
    gr.Markdown(
        "The detection table shows all classes; the alert is raised when a phone is spatially associated with the driver."
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
