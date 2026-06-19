"""Evaluate driver cellphone-use alert performance on a YOLO split.

This script evaluates the deployed alert behavior, not only generic object
detection quality. The app raises an alert when the model predicts the
`Cellphone-in-drivers` class, so this report treats each image as an alert/no
alert decision and also reports object-level matching for that class.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

try:
    from .dataset_check import (
        collect_images,
        dataset_root_from_yaml,
        infer_label_path_for_image,
        normalize_names,
        resolve_split_paths,
    )
    from .utils import ensure_dir, load_yaml, save_json
except ImportError:
    from dataset_check import (
        collect_images,
        dataset_root_from_yaml,
        infer_label_path_for_image,
        normalize_names,
        resolve_split_paths,
    )
    from utils import ensure_dir, load_yaml, save_json


DEFAULT_ALERT_ALIASES = ["Cellphone-in-drivers", "Cellphone-in-driver"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate image-level alert performance for driver cellphone-use detection."
    )
    parser.add_argument("--weights", default="models/best.pt", help="Path to trained YOLO weights.")
    parser.add_argument("--data", required=True, help="Path to dataset data.yaml.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Device, for example cpu or 0.")
    parser.add_argument(
        "--alert-class",
        default=",".join(DEFAULT_ALERT_ALIASES),
        help="Comma-separated alert class name aliases. First match in data.yaml is used.",
    )
    parser.add_argument("--alert-class-id", type=int, default=None)
    parser.add_argument(
        "--default-conf",
        type=float,
        default=0.25,
        help="Confidence threshold used for the main reported confusion matrix.",
    )
    parser.add_argument(
        "--prediction-conf",
        type=float,
        default=0.001,
        help="Low confidence threshold used while collecting predictions for threshold sweep.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold passed to YOLO prediction.",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold for object-level matching of the alert class.",
    )
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--max-conf", type=float, default=0.95)
    parser.add_argument("--conf-step", type=float, default=0.05)
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.90,
        help="Recall target for a safety-oriented recommended threshold.",
    )
    parser.add_argument(
        "--precision-floor",
        type=float,
        default=0.80,
        help="Precision floor for a low-false-alert recommended threshold.",
    )
    parser.add_argument("--output-json", default="outputs/alert_eval_test.json")
    parser.add_argument("--output-images-csv", default="outputs/alert_eval_images_test.csv")
    parser.add_argument("--output-thresholds-csv", default="outputs/alert_eval_thresholds_test.csv")
    parser.add_argument("--plots-dir", default="outputs/alert_eval_test_plots")
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float | None:
    """Divide and return None when denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def harmonic_f1(precision: float | None, recall: float | None) -> float | None:
    """Compute F1 from precision and recall."""
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def metric_score(value: float | None) -> float:
    """Return a sortable score for optional metrics."""
    return -1.0 if value is None else value


def thresholds(min_conf: float, max_conf: float, step: float, default_conf: float) -> list[float]:
    """Build a sorted threshold grid that always includes default_conf."""
    if step <= 0:
        raise SystemExit("--conf-step must be greater than 0.")
    if min_conf > max_conf:
        raise SystemExit("--min-conf must be <= --max-conf.")

    values: set[float] = {round(default_conf, 6)}
    current = min_conf
    while current <= max_conf + 1e-9:
        values.add(round(current, 6))
        current += step
    return sorted(values)


def resolve_alert_class_id(
    names: dict[int, str],
    alert_aliases: list[str],
    explicit_class_id: int | None,
) -> tuple[int, str]:
    """Find the alert class id from data.yaml class names."""
    if explicit_class_id is not None:
        if explicit_class_id not in names:
            raise SystemExit(f"--alert-class-id {explicit_class_id} is not present in data.yaml names.")
        return explicit_class_id, names[explicit_class_id]

    lookup = {name.lower(): class_id for class_id, name in names.items()}
    for alias in alert_aliases:
        alias = alias.strip()
        if alias.lower() in lookup:
            class_id = lookup[alias.lower()]
            return class_id, names[class_id]

    aliases = ", ".join(alert_aliases)
    available = ", ".join(f"{class_id}:{name}" for class_id, name in sorted(names.items()))
    raise SystemExit(f"Alert class not found. Tried [{aliases}]. Available classes: {available}")


def load_split_images(data_path: str | Path, split: str) -> tuple[dict[str, Any], Path, list[Path]]:
    """Load data.yaml and collect image paths for a split."""
    yaml_path = Path(data_path).expanduser()
    if not yaml_path.exists():
        raise SystemExit(f"data.yaml not found: {yaml_path}")
    config = load_yaml(yaml_path)
    dataset_root = dataset_root_from_yaml(config, yaml_path)
    split_paths = resolve_split_paths(config.get(split), dataset_root)
    images, issues = collect_images(split_paths)
    if issues:
        for issue in issues:
            print(f"Warning: {issue}")
    if not images:
        raise SystemExit(f"No images found for split '{split}' in {yaml_path}")
    return config, dataset_root, images


def read_target_labels(label_path: Path, target_class_id: int) -> list[tuple[float, float, float, float]]:
    """Read normalized YOLO boxes for the target class from one label file."""
    if not label_path.exists():
        return []

    boxes = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if class_id == target_class_id:
            boxes.append((x_center, y_center, width, height))
    return boxes


def normalized_to_xyxy(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO xywh to absolute xyxy."""
    x_center, y_center, width, height = box
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    return (x1, y1, x2, y2)


def get_image_size(image_path: Path, result: Any) -> tuple[int, int]:
    """Return image width and height from Ultralytics result or the image file."""
    orig_shape = getattr(result, "orig_shape", None)
    if orig_shape is not None and len(orig_shape) >= 2:
        return int(orig_shape[1]), int(orig_shape[0])

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except Exception as exc:
        raise RuntimeError(f"Could not read image size for {image_path}: {exc}") from exc


def collect_target_predictions(result: Any, target_class_id: int) -> list[dict[str, Any]]:
    """Collect alert-class predictions from an Ultralytics result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    predictions: list[dict[str, Any]] = []
    for box in boxes:
        class_id = int(box.cls.item())
        if class_id != target_class_id:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        predictions.append(
            {
                "confidence": float(box.conf.item()),
                "xyxy": (x1, y1, x2, y2),
            }
        )
    return predictions


def box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    """Compute image-level binary classification metrics."""
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)
    f1 = harmonic_f1(precision, recall)
    false_alarm_rate = safe_div(fp, fp + tn)
    miss_rate = safe_div(fn, fn + tp)
    npv = safe_div(tn, tn + fn)
    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = (recall + specificity) / 2

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
    }


def object_metrics(records: list[dict[str, Any]], threshold: float, match_iou: float) -> dict[str, Any]:
    """Compute object-level precision/recall/F1 for the alert class."""
    tp = 0
    fp = 0
    fn = 0
    total_gt = 0
    total_pred = 0

    for record in records:
        gt_boxes = record["gt_boxes_xyxy"]
        predictions = [
            prediction
            for prediction in record["target_predictions"]
            if prediction["confidence"] >= threshold
        ]
        predictions = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
        matched_gt: set[int] = set()
        total_gt += len(gt_boxes)
        total_pred += len(predictions)

        for prediction in predictions:
            best_index = None
            best_iou = 0.0
            for index, gt_box in enumerate(gt_boxes):
                if index in matched_gt:
                    continue
                iou = box_iou(prediction["xyxy"], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index is not None and best_iou >= match_iou:
                matched_gt.add(best_index)
                tp += 1
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "threshold": threshold,
        "match_iou": match_iou,
        "ground_truth_instances": total_gt,
        "predicted_instances": total_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": harmonic_f1(precision, recall),
    }


def alert_metrics(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Compute image-level alert/no-alert metrics at a confidence threshold."""
    tp = fp = fn = tn = 0
    for record in records:
        gt_alert = bool(record["gt_alert"])
        pred_alert = any(
            prediction["confidence"] >= threshold
            for prediction in record["target_predictions"]
        )
        if gt_alert and pred_alert:
            tp += 1
        elif not gt_alert and pred_alert:
            fp += 1
        elif gt_alert and not pred_alert:
            fn += 1
        else:
            tn += 1

    metrics = binary_metrics(tp, fp, fn, tn)
    metrics["threshold"] = threshold
    metrics["alert_rate"] = safe_div(tp + fp, len(records))
    return metrics


def threshold_sweep(
    records: list[dict[str, Any]],
    threshold_values: list[float],
    match_iou: float,
) -> list[dict[str, Any]]:
    """Compute alert and object metrics across confidence thresholds."""
    sweep = []
    for threshold in threshold_values:
        alert = alert_metrics(records, threshold)
        objects = object_metrics(records, threshold, match_iou)
        sweep.append(
            {
                "threshold": threshold,
                "alert_precision": alert["precision"],
                "alert_recall": alert["recall"],
                "alert_f1": alert["f1"],
                "alert_accuracy": alert["accuracy"],
                "alert_balanced_accuracy": alert["balanced_accuracy"],
                "alert_specificity": alert["specificity"],
                "alert_false_alarm_rate": alert["false_alarm_rate"],
                "alert_miss_rate": alert["miss_rate"],
                "alert_tp": alert["tp"],
                "alert_fp": alert["fp"],
                "alert_fn": alert["fn"],
                "alert_tn": alert["tn"],
                "object_precision": objects["precision"],
                "object_recall": objects["recall"],
                "object_f1": objects["f1"],
                "object_tp": objects["tp"],
                "object_fp": objects["fp"],
                "object_fn": objects["fn"],
            }
        )
    return sweep


def choose_recommendations(
    sweep: list[dict[str, Any]],
    default_conf: float,
    target_recall: float,
    precision_floor: float,
) -> dict[str, Any]:
    """Choose useful operating thresholds for reporting."""
    f1_candidates = [row for row in sweep if row["alert_f1"] is not None]
    balanced_candidates = [row for row in sweep if row["alert_balanced_accuracy"] is not None]
    best_f1 = (
        max(f1_candidates, key=lambda row: (metric_score(row["alert_f1"]), row["threshold"]))
        if f1_candidates
        else None
    )
    best_balanced = (
        max(
            balanced_candidates,
            key=lambda row: (metric_score(row["alert_balanced_accuracy"]), row["threshold"]),
        )
        if balanced_candidates
        else None
    )

    recall_candidates = [
        row for row in sweep if row["alert_recall"] is not None and row["alert_recall"] >= target_recall
    ]
    if recall_candidates:
        recall_priority = max(
            recall_candidates,
            key=lambda row: (metric_score(row["alert_precision"]), metric_score(row["alert_f1"]), row["threshold"]),
        )
        recall_note = f"highest precision while recall >= {target_recall:.2f}"
    elif not any(row["alert_recall"] is not None for row in sweep):
        recall_priority = None
        recall_note = "recall is undefined because the split has no positive alert labels"
    else:
        recall_priority = max(
            sweep,
            key=lambda row: (metric_score(row["alert_recall"]), metric_score(row["alert_precision"]), row["threshold"]),
        )
        recall_note = f"no threshold reached recall >= {target_recall:.2f}; selected highest recall"

    precision_candidates = [
        row for row in sweep if row["alert_precision"] is not None and row["alert_precision"] >= precision_floor
    ]
    if precision_candidates:
        precision_guardrail = max(
            precision_candidates,
            key=lambda row: (metric_score(row["alert_recall"]), metric_score(row["alert_f1"]), row["threshold"]),
        )
        precision_note = f"highest recall while precision >= {precision_floor:.2f}"
    elif not any(row["alert_precision"] is not None for row in sweep):
        precision_guardrail = None
        precision_note = "precision is undefined because no alert predictions were made or no positive labels exist"
    else:
        precision_guardrail = max(
            sweep,
            key=lambda row: (metric_score(row["alert_precision"]), metric_score(row["alert_recall"]), row["threshold"]),
        )
        precision_note = f"no threshold reached precision >= {precision_floor:.2f}; selected highest precision"

    app_default = min(sweep, key=lambda row: abs(row["threshold"] - default_conf))
    return {
        "app_default": app_default,
        "best_alert_f1": best_f1,
        "best_balanced_accuracy": best_balanced,
        "recall_priority": {"policy": recall_note, "metrics": recall_priority},
        "precision_guardrail": {"policy": precision_note, "metrics": precision_guardrail},
    }


def write_images_csv(records: list[dict[str, Any]], threshold: float, output_path: str | Path) -> Path:
    """Write per-image alert decisions at the default threshold."""
    output = Path(output_path)
    ensure_dir(output.parent)
    fieldnames = [
        "image_path",
        "label_path",
        "gt_alert",
        "gt_alert_instances",
        "pred_alert",
        "pred_alert_instances",
        "max_alert_confidence",
        "outcome",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            predictions = [
                prediction
                for prediction in record["target_predictions"]
                if prediction["confidence"] >= threshold
            ]
            gt_alert = bool(record["gt_alert"])
            pred_alert = bool(predictions)
            if gt_alert and pred_alert:
                outcome = "TP"
            elif not gt_alert and pred_alert:
                outcome = "FP"
            elif gt_alert and not pred_alert:
                outcome = "FN"
            else:
                outcome = "TN"

            max_confidence = max(
                [prediction["confidence"] for prediction in record["target_predictions"]],
                default=None,
            )
            writer.writerow(
                {
                    "image_path": record["image_path"],
                    "label_path": record["label_path"],
                    "gt_alert": gt_alert,
                    "gt_alert_instances": record["gt_alert_instances"],
                    "pred_alert": pred_alert,
                    "pred_alert_instances": len(predictions),
                    "max_alert_confidence": max_confidence,
                    "outcome": outcome,
                }
            )
    return output


def write_thresholds_csv(sweep: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write threshold sweep metrics."""
    output = Path(output_path)
    ensure_dir(output.parent)
    fieldnames = list(sweep[0].keys()) if sweep else []
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep)
    return output


def save_plots(
    sweep: list[dict[str, Any]],
    default_alert: dict[str, Any],
    plots_dir: str | Path,
) -> dict[str, str]:
    """Save simple PNG plots for report and README usage."""
    output_dir = ensure_dir(plots_dir)
    plot_paths: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib is unavailable; skipping alert plots: {exc}")
        return plot_paths

    x_values = [row["threshold"] for row in sweep]

    plt.figure(figsize=(8, 5))
    for key, label in [
        ("alert_precision", "Precision"),
        ("alert_recall", "Recall"),
        ("alert_f1", "F1"),
        ("alert_balanced_accuracy", "Balanced accuracy"),
    ]:
        plt.plot(x_values, [row[key] if row[key] is not None else float("nan") for row in sweep], label=label)
    plt.xlabel("Confidence threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Alert-level metrics across confidence thresholds")
    curves_path = output_dir / "alert_threshold_curves.png"
    plt.tight_layout()
    plt.savefig(curves_path, dpi=160)
    plt.close()
    plot_paths["alert_threshold_curves"] = str(curves_path)

    plt.figure(figsize=(8, 5))
    for key, label in [
        ("alert_miss_rate", "Miss rate"),
        ("alert_false_alarm_rate", "False alarm rate"),
    ]:
        plt.plot(x_values, [row[key] if row[key] is not None else float("nan") for row in sweep], label=label)
    plt.xlabel("Confidence threshold")
    plt.ylabel("Rate")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Alert error rates across confidence thresholds")
    errors_path = output_dir / "alert_error_rates.png"
    plt.tight_layout()
    plt.savefig(errors_path, dpi=160)
    plt.close()
    plot_paths["alert_error_rates"] = str(errors_path)

    matrix = [
        [default_alert["tn"], default_alert["fp"]],
        [default_alert["fn"], default_alert["tp"]],
    ]
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["Pred no alert", "Pred alert"])
    plt.yticks([0, 1], ["GT no alert", "GT alert"])
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            plt.text(column_index, row_index, str(value), ha="center", va="center", fontsize=13)
    plt.title(f"Alert confusion matrix at conf={default_alert['threshold']:.2f}")
    plt.colorbar(fraction=0.046, pad=0.04)
    matrix_path = output_dir / "alert_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=160)
    plt.close()
    plot_paths["alert_confusion_matrix"] = str(matrix_path)

    return plot_paths


def collect_records(
    model: Any,
    images: list[Path],
    alert_class_id: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Run prediction and collect ground-truth/prediction records."""
    records = []
    total = len(images)
    for index, image_path in enumerate(images, start=1):
        if index == 1 or index % 25 == 0 or index == total:
            print(f"Evaluating image {index}/{total}: {image_path.name}")

        results = model.predict(
            source=str(image_path),
            conf=args.prediction_conf,
            imgsz=args.imgsz,
            iou=args.nms_iou,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        image_width, image_height = get_image_size(image_path, result)
        label_path = infer_label_path_for_image(image_path)
        gt_normalized = read_target_labels(label_path, alert_class_id)
        gt_boxes_xyxy = [
            normalized_to_xyxy(box, image_width, image_height)
            for box in gt_normalized
        ]
        target_predictions = collect_target_predictions(result, alert_class_id)
        records.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "width": image_width,
                "height": image_height,
                "gt_alert": bool(gt_boxes_xyxy),
                "gt_alert_instances": len(gt_boxes_xyxy),
                "gt_boxes_xyxy": gt_boxes_xyxy,
                "target_predictions": target_predictions,
            }
        )
    return records


def print_summary(summary: dict[str, Any]) -> None:
    """Print the most important results."""
    alert = summary["alert_level_at_default_conf"]
    objects = summary["object_level_at_default_conf"]
    recommended = summary["recommended_thresholds"]["best_alert_f1"]

    print("\nDriver cellphone-use alert evaluation")
    print("=" * 44)
    print(f"split: {summary['split']}")
    print(f"images: {summary['image_count']}")
    print(f"positive alert images: {summary['positive_images']}")
    print(f"negative alert images: {summary['negative_images']}")
    print(f"alert class: {summary['alert_class']['id']} ({summary['alert_class']['name']})")
    print(f"\nAt app/default confidence = {alert['threshold']:.2f}")
    print(f"TP={alert['tp']} FP={alert['fp']} FN={alert['fn']} TN={alert['tn']}")
    print(f"precision={alert['precision']} recall={alert['recall']} f1={alert['f1']}")
    print(f"miss_rate={alert['miss_rate']} false_alarm_rate={alert['false_alarm_rate']}")
    print(f"\nObject-level alert class at IoU >= {objects['match_iou']:.2f}")
    print(f"TP={objects['tp']} FP={objects['fp']} FN={objects['fn']}")
    print(f"precision={objects['precision']} recall={objects['recall']} f1={objects['f1']}")
    if recommended is None:
        print("\nBest alert F1 threshold: unavailable")
    else:
        print(f"\nBest alert F1 threshold: {recommended['threshold']:.2f}")
        print(
            f"precision={recommended['alert_precision']} "
            f"recall={recommended['alert_recall']} "
            f"f1={recommended['alert_f1']}"
        )
    if summary.get("warnings"):
        print("\nWarnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")
    print(f"\nSaved JSON: {summary['outputs']['json']}")
    print(f"Saved image CSV: {summary['outputs']['images_csv']}")
    print(f"Saved threshold CSV: {summary['outputs']['thresholds_csv']}")
    if summary["outputs"].get("plots"):
        print("Saved plots:")
        for name, path in summary["outputs"]["plots"].items():
            print(f"- {name}: {path}")


def main() -> None:
    """Run alert evaluation."""
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")

    config, dataset_root, images = load_split_images(args.data, args.split)
    names = normalize_names(config.get("names"))
    if not names:
        raise SystemExit("Class names are missing or invalid in data.yaml.")

    alert_aliases = [item.strip() for item in args.alert_class.split(",") if item.strip()]
    alert_class_id, alert_class_name = resolve_alert_class_id(names, alert_aliases, args.alert_class_id)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: install ultralytics before alert evaluation.") from exc

    model = YOLO(str(weights_path))
    records = collect_records(model, images, alert_class_id, args)

    threshold_values = thresholds(args.min_conf, args.max_conf, args.conf_step, args.default_conf)
    sweep = threshold_sweep(records, threshold_values, args.match_iou)
    default_alert = alert_metrics(records, args.default_conf)
    default_objects = object_metrics(records, args.default_conf, args.match_iou)
    positive_images = sum(1 for record in records if record["gt_alert"])
    negative_images = sum(1 for record in records if not record["gt_alert"])
    warnings = []
    if positive_images == 0:
        warnings.append(
            "The selected split has no positive images for the alert class; alert recall, F1, and threshold selection are undefined."
        )
    if negative_images == 0:
        warnings.append(
            "The selected split has no negative images for the alert class; false-alarm rate and specificity are undefined."
        )
    recommendations = choose_recommendations(
        sweep,
        args.default_conf,
        args.target_recall,
        args.precision_floor,
    )

    images_csv = write_images_csv(records, args.default_conf, args.output_images_csv)
    thresholds_csv = write_thresholds_csv(sweep, args.output_thresholds_csv)
    plot_paths = save_plots(sweep, default_alert, args.plots_dir)

    summary = {
        "task": "image-level driver cellphone-use alert plus object-level alert-class matching",
        "weights": str(weights_path),
        "data": str(Path(args.data).expanduser()),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "imgsz": args.imgsz,
        "device": str(args.device),
        "nms_iou": args.nms_iou,
        "prediction_conf": args.prediction_conf,
        "default_conf": args.default_conf,
        "match_iou": args.match_iou,
        "alert_class": {"id": alert_class_id, "name": alert_class_name},
        "image_count": len(records),
        "positive_images": positive_images,
        "negative_images": negative_images,
        "ground_truth_alert_instances": sum(record["gt_alert_instances"] for record in records),
        "alert_level_at_default_conf": default_alert,
        "object_level_at_default_conf": default_objects,
        "threshold_sweep": sweep,
        "recommended_thresholds": recommendations,
        "warnings": warnings,
        "outputs": {
            "json": str(Path(args.output_json)),
            "images_csv": str(images_csv),
            "thresholds_csv": str(thresholds_csv),
            "plots": plot_paths,
        },
        "notes": [
            "Alert-level metrics match the app behavior: an image is positive when the model predicts the alert class.",
            "Object-level metrics only match boxes of the alert class using the configured IoU threshold.",
            "The default threshold is useful for current app behavior; the sweep shows operating points for product trade-offs.",
        ],
    }
    save_json(summary, args.output_json)
    print_summary(summary)


if __name__ == "__main__":
    main()
