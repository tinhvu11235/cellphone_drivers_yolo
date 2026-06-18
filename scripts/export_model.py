"""Export a YOLO model to deployment formats such as ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export a trained YOLO model.")
    parser.add_argument("--weights", default="models/best.pt", help="Path to trained .pt weights.")
    parser.add_argument("--format", default="onnx", help="Export format, for example onnx.")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    """Export model using Ultralytics."""
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: install ultralytics before exporting.") from exc

    model = YOLO(str(weights_path))
    exported_path = model.export(format=args.format, imgsz=args.imgsz)
    print(f"Exported model: {exported_path}")
    print("For Hugging Face Spaces in this project, keep PyTorch weights at models/best.pt by default.")


if __name__ == "__main__":
    main()
