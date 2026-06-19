"""Evaluate rule-based driver cellphone-use inference on manual image labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

try:
    from .driver_phone_heuristic import infer_driver_phone_usage
    from .utils import ensure_dir, save_json
except ImportError:
    from driver_phone_heuristic import infer_driver_phone_usage
    from utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the rule-based driver phone-use heuristic on a manual CSV."
    )
    parser.add_argument("--weights", default="models/best.pt", help="Path to trained YOLO weights.")
    parser.add_argument(
        "--labels-csv",
        default="configs/driver_phone_usage_labels_template.csv",
        help="CSV with columns: image,label. Label 1 means driver using phone, 0 means not using phone.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Optional image root for relative image paths in the CSV. Defaults to the CSV parent directory.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Device, for example cpu or 0.")
    parser.add_argument(
        "--prediction-conf",
        type=float,
        default=0.25,
        help="YOLO prediction confidence before heuristic per-class filtering.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--output-json", default="outputs/heuristic_eval_summary.json")
    parser.add_argument("--output-csv", default="outputs/heuristic_eval_predictions.csv")
    return parser.parse_args()


def load_manual_labels(labels_csv: str | Path, images_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load image-level manual labels from CSV."""
    csv_path = Path(labels_csv)
    if not csv_path.exists():
        raise SystemExit(f"Labels CSV not found: {csv_path}")

    image_root = Path(images_dir) if images_dir is not None else csv_path.parent
    records: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"image", "label"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise SystemExit("Labels CSV must contain columns: image,label")

        for row_number, row in enumerate(reader, start=2):
            image_value = (row.get("image") or "").strip()
            label_value = (row.get("label") or "").strip()
            if not image_value:
                continue
            if label_value not in {"0", "1"}:
                raise SystemExit(f"Invalid label at row {row_number}: expected 0 or 1, got {label_value!r}")

            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = image_root / image_path
            if not image_path.exists():
                raise SystemExit(f"Image not found at row {row_number}: {image_path}")

            records.append(
                {
                    "image": image_value,
                    "image_path": image_path,
                    "label": int(label_value),
                }
            )

    if not records:
        raise SystemExit(f"No labeled images found in {csv_path}")
    return records


def detections_from_result(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    """Convert an Ultralytics result to the heuristic detection format."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    detections = []
    for box in boxes:
        class_id = int(box.cls.item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "class_name": str(names.get(class_id, class_id)),
                "confidence": float(box.conf.item()),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return detections


def image_shape_from_result(result: Any) -> tuple[int, int]:
    """Return image shape as (height, width) from an Ultralytics result."""
    orig_shape = getattr(result, "orig_shape", None)
    if orig_shape is None or len(orig_shape) < 2:
        raise RuntimeError("Ultralytics result does not expose orig_shape.")
    return int(orig_shape[0]), int(orig_shape[1])


def safe_div(numerator: float, denominator: float) -> float | None:
    """Return numerator/denominator, or None when denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute binary classification metrics for image-level behavior labels."""
    tp = fp = fn = tn = 0
    for row in rows:
        label = int(row["label"])
        prediction = int(row["prediction"])
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 1 and prediction == 0:
            fn += 1
        else:
            tn += 1

    accuracy = safe_div(tp + tn, tp + fp + fn + tn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Run YOLO, apply heuristic, and compute metrics."""
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: install ultralytics before evaluation.") from exc

    records = load_manual_labels(args.labels_csv, args.images_dir)
    model = YOLO(str(weights_path))
    names = {int(key): str(value) for key, value in getattr(model, "names", {}).items()}

    rows: list[dict[str, Any]] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        image_path = record["image_path"]
        if index == 1 or index % 25 == 0 or index == total:
            print(f"Evaluating {index}/{total}: {image_path.name}")

        results = model.predict(
            source=str(image_path),
            conf=args.prediction_conf,
            imgsz=args.imgsz,
            iou=args.nms_iou,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        detections = detections_from_result(result, names)
        prediction = infer_driver_phone_usage(detections, image_shape_from_result(result))
        pred_label = int(bool(prediction["driver_using_phone"]))

        rows.append(
            {
                "image": record["image"],
                "image_path": str(image_path),
                "label": record["label"],
                "prediction": pred_label,
                "risk_score": prediction["risk_score"],
                "reason": prediction["reason"],
                "driver_bbox": prediction["driver_bbox"],
                "phone_bbox": prediction["phone_bbox"],
                "wheel_bbox": prediction["wheel_bbox"],
                "outcome": _outcome(record["label"], pred_label),
            }
        )

    metrics = compute_metrics(rows)
    predictions_csv = write_predictions_csv(rows, args.output_csv)
    summary = {
        "task": "rule-based driver cellphone-use behavior inference",
        "weights": str(weights_path),
        "labels_csv": str(Path(args.labels_csv)),
        "image_count": len(rows),
        "positive_labels": sum(1 for row in rows if row["label"] == 1),
        "negative_labels": sum(1 for row in rows if row["label"] == 0),
        "prediction_conf": args.prediction_conf,
        "imgsz": args.imgsz,
        "device": str(args.device),
        "metrics": metrics,
        "outputs": {
            "json": str(Path(args.output_json)),
            "csv": str(predictions_csv),
        },
    }
    save_json(summary, args.output_json)
    return summary


def write_predictions_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> Path:
    """Write per-image heuristic predictions for manual inspection."""
    output_path = Path(output_csv)
    ensure_dir(output_path.parent)
    fieldnames = [
        "image",
        "image_path",
        "label",
        "prediction",
        "risk_score",
        "reason",
        "driver_bbox",
        "phone_bbox",
        "wheel_bbox",
        "outcome",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def print_summary(summary: dict[str, Any]) -> None:
    """Print the most useful evaluation numbers."""
    metrics = summary["metrics"]
    matrix = metrics["confusion_matrix"]
    print("\nRule-based driver phone-use evaluation")
    print("=" * 44)
    print(f"images: {summary['image_count']}")
    print(f"positive labels: {summary['positive_labels']}")
    print(f"negative labels: {summary['negative_labels']}")
    print(f"accuracy: {metrics['accuracy']}")
    print(f"precision: {metrics['precision']}")
    print(f"recall: {metrics['recall']}")
    print(f"f1: {metrics['f1']}")
    print(f"confusion matrix [[TN, FP], [FN, TP]]: {matrix['matrix']}")
    print(f"saved JSON: {summary['outputs']['json']}")
    print(f"saved CSV: {summary['outputs']['csv']}")


def _outcome(label: int, prediction: int) -> str:
    if label == 1 and prediction == 1:
        return "TP"
    if label == 0 and prediction == 1:
        return "FP"
    if label == 1 and prediction == 0:
        return "FN"
    return "TN"


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    summary = run_evaluation(args)
    print_summary(summary)


if __name__ == "__main__":
    main()
