"""Evaluate YOLO validation metrics and save a JSON summary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .utils import as_list, ensure_dir, safe_float, save_json
except ImportError:
    from utils import as_list, ensure_dir, safe_float, save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on a validation split.")
    parser.add_argument("--weights", required=True, help="Path to best.pt.")
    parser.add_argument("--data", required=True, help="Path to dataset data.yaml.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Device, for example 0 or cpu.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--output", default="outputs/eval_summary.json")
    return parser.parse_args()


def get_attr(obj: Any, name: str) -> Any:
    """Get an attribute from an object, returning None when unavailable."""
    return getattr(obj, name, None) if obj is not None else None


def to_float_list(value: Any) -> list[float | None]:
    """Convert a metric array-like value to a list of floats or None values."""
    return [safe_float(item) for item in as_list(value)]


def class_name_map(metrics: Any, model: Any) -> dict[int, str]:
    """Read class names from metrics or model metadata."""
    names = get_attr(metrics, "names") or getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def extract_metrics(metrics: Any, model: Any) -> dict[str, Any]:
    """Extract object detection metrics from Ultralytics results."""
    box = get_attr(metrics, "box")
    precision = safe_float(get_attr(box, "mp"))
    recall = safe_float(get_attr(box, "mr"))
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    names = class_name_map(metrics, model)
    per_class_precision = to_float_list(get_attr(box, "p"))
    per_class_recall = to_float_list(get_attr(box, "r"))
    per_class_map50 = to_float_list(get_attr(box, "ap50"))
    per_class_map5095 = to_float_list(get_attr(box, "maps"))
    metric_class_ids = [int(class_id) for class_id in as_list(get_attr(box, "ap_class_index"))]
    if not metric_class_ids:
        metric_class_ids = sorted(names)
    metric_positions = {class_id: position for position, class_id in enumerate(metric_class_ids)}

    per_class: list[dict[str, Any]] = []
    for class_id in sorted(names):
        position = metric_positions.get(class_id)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "precision": _metric_get(per_class_precision, position),
                "recall": _metric_get(per_class_recall, position),
                "mAP50": _metric_get(per_class_map50, position),
                "mAP50_95": _map_get(per_class_map5095, class_id, position, len(names)),
            }
        )

    save_dir = get_attr(metrics, "save_dir")
    plots = {}
    if save_dir is not None:
        run_dir = Path(save_dir)
        for plot in [
            "confusion_matrix.png",
            "PR_curve.png",
            "F1_curve.png",
            "P_curve.png",
            "R_curve.png",
            "results.png",
        ]:
            plots[plot] = str(run_dir / plot)

    return {
        "mAP50": safe_float(get_attr(box, "map50")),
        "mAP50_95": safe_float(get_attr(box, "map")),
        "mean_precision": precision,
        "mean_recall": recall,
        "F1_score": f1,
        "per_class": per_class,
        "plots": plots,
    }


def _list_get(values: list[Any], index: int) -> Any:
    """Return list value or None when unavailable."""
    return values[index] if index < len(values) else None


def _metric_get(values: list[Any], position: int | None) -> Any:
    """Return a metric by compact class position, or None for absent classes."""
    if position is None:
        return None
    return _list_get(values, position)


def _map_get(values: list[Any], class_id: int, position: int | None, class_count: int) -> Any:
    """Return class mAP, avoiding misleading values for absent classes."""
    if position is None:
        return None
    if len(values) == class_count:
        return _list_get(values, class_id)
    return _list_get(values, position)


def warn_unavailable(summary: dict[str, Any]) -> None:
    """Print warnings for unavailable metrics."""
    for key in ["mAP50", "mAP50_95", "mean_precision", "mean_recall", "F1_score"]:
        if summary.get(key) is None:
            print(f"Warning: metric unavailable from Ultralytics result object: {key}")


def print_summary(summary: dict[str, Any], output_path: str | Path) -> None:
    """Print a concise validation metrics table."""
    print("\nValidation Metrics")
    print("=" * 22)
    print(f"{'Metric':<18}Value")
    print("-" * 34)
    for key in ["mAP50", "mAP50_95", "mean_precision", "mean_recall", "F1_score"]:
        value = summary.get(key)
        print(f"{key:<18}{value if value is not None else 'N/A'}")

    if summary.get("plots"):
        print("\nValidation plots are expected at:")
        for name, path in summary["plots"].items():
            print(f"- {name}: {path}")

    print(f"\nSaved evaluation summary to: {output_path}")


def main() -> None:
    """Run validation evaluation."""
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")

    ensure_dir(Path(args.output).parent)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: install ultralytics before evaluation.") from exc

    model = YOLO(str(weights_path))
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device, split=args.split, plots=True)
    summary = extract_metrics(metrics, model)
    summary.update(
        {
            "weights": str(weights_path),
            "data": str(args.data),
            "imgsz": args.imgsz,
            "device": str(args.device),
            "split": args.split,
        }
    )
    save_json(summary, args.output)
    warn_unavailable(summary)
    print_summary(summary, args.output)


if __name__ == "__main__":
    main()
