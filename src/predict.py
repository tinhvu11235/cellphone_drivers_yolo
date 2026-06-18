"""Run YOLO inference on an image or folder of images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .utils import ensure_dir, list_image_files
except ImportError:
    from utils import ensure_dir, list_image_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run YOLO prediction on images.")
    parser.add_argument("--weights", required=True, help="Path to trained weights.")
    parser.add_argument("--source", required=True, help="Image file or image folder.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Device, for example cpu or 0.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def is_builtin_yolo_weight(weights: str) -> bool:
    """Return True for Ultralytics model names that may be downloaded automatically."""
    name = Path(weights).name.lower()
    return name.startswith("yolo") and name.endswith(".pt")


def print_detections(image_path: Path, result: Any, names: dict[int, str]) -> None:
    """Print object detections for one image."""
    print(f"\n{image_path}")
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        print("- no detections")
        return

    for box in boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        class_name = names.get(class_id, str(class_id))
        print(
            f"- {class_name}: confidence={confidence:.3f}, "
            f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )


def main() -> None:
    """Run prediction."""
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists() and not is_builtin_yolo_weight(args.weights):
        raise SystemExit(f"Weights file not found: {weights_path}")

    images = list_image_files(args.source)
    if not images:
        raise SystemExit(f"No image files found in source: {args.source}")

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: install ultralytics and opencv-python-headless.") from exc

    output_dir = ensure_dir("outputs/predictions")
    model = YOLO(str(args.weights))
    names = {int(key): str(value) for key, value in getattr(model, "names", {}).items()}

    for image_path in images:
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        print_detections(image_path, result, names)

        if args.save:
            annotated = result.plot()
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), annotated)
            print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
