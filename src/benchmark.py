"""Benchmark YOLO inference latency and FPS."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Any

try:
    from .utils import ensure_dir, get_device_name, list_image_files, safe_float, save_json
except ImportError:
    from utils import ensure_dir, get_device_name, list_image_files, safe_float, save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark YOLO inference latency.")
    parser.add_argument("--weights", required=True, help="Path to best.pt.")
    parser.add_argument("--source", required=True, help="Image file or image folder.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Device, for example cpu or 0.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--save-json", default="outputs/benchmark_summary.json")
    parser.add_argument("--save-csv", default="outputs/benchmark_results.csv")
    return parser.parse_args()


def percentile(values: list[float], percent: float) -> float:
    """Compute percentile with linear interpolation."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * (percent / 100)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def speed_value(result: Any, key: str) -> float | None:
    """Read Ultralytics speed values in milliseconds when available."""
    speed = getattr(result, "speed", None)
    if not isinstance(speed, dict):
        return None
    return safe_float(speed.get(key))


def write_csv(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write benchmark per-image records to CSV."""
    csv_path = Path(path)
    ensure_dir(csv_path.parent)
    fieldnames = [
        "run_index",
        "image_path",
        "latency_ms",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def print_summary(summary: dict[str, Any]) -> None:
    """Print a concise benchmark summary."""
    print("\nInference Benchmark Summary")
    print("=" * 32)
    print(f"model: {summary['model_path']}")
    print(f"device: {summary['device']} ({summary['device_name']})")
    print(f"image size: {summary['imgsz']}")
    print(f"confidence threshold: {summary['conf']}")
    print(f"warmup runs: {summary['warmup_runs']}")
    print(f"benchmark runs: {summary['benchmark_runs']}")
    print(f"total source images: {summary['total_source_images']}")
    print(f"average latency: {summary['average_latency_ms']:.2f} ms/image")
    print(f"p50 latency: {summary['p50_latency_ms']:.2f} ms/image")
    print(f"p95 latency: {summary['p95_latency_ms']:.2f} ms/image")
    print(f"min latency: {summary['min_latency_ms']:.2f} ms/image")
    print(f"max latency: {summary['max_latency_ms']:.2f} ms/image")
    print(f"FPS: {summary['fps']:.2f}")
    print("Note: latency and FPS depend on hardware, image size, model size, and server load.")


def main() -> None:
    """Run benchmark."""
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")

    images = list_image_files(args.source)
    if not images:
        raise SystemExit(f"No image files found in source: {args.source}")
    if args.runs <= 0:
        raise SystemExit("--runs must be greater than 0.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: install ultralytics before benchmarking.") from exc

    model = YOLO(str(weights_path))
    image_cycle = cycle(images)

    print(f"Running {args.warmup} warmup iteration(s)...")
    for image_path in islice(image_cycle, max(args.warmup, 0)):
        model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

    print(f"Running {args.runs} benchmark iteration(s) with batch size 1...")
    records: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for run_index, image_path in enumerate(islice(image_cycle, args.runs), start=1):
        start = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        result = results[0] if results else None
        latencies_ms.append(latency_ms)
        records.append(
            {
                "run_index": run_index,
                "image_path": str(image_path),
                "latency_ms": latency_ms,
                "preprocess_ms": speed_value(result, "preprocess"),
                "inference_ms": speed_value(result, "inference"),
                "postprocess_ms": speed_value(result, "postprocess"),
            }
        )

    avg_latency = statistics.mean(latencies_ms)
    summary = {
        "model_path": str(weights_path),
        "device": str(args.device),
        "device_name": get_device_name(args.device),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "warmup_runs": args.warmup,
        "benchmark_runs": args.runs,
        "total_source_images": len(images),
        "average_latency_ms": avg_latency,
        "p50_latency_ms": percentile(latencies_ms, 50),
        "median_latency_ms": statistics.median(latencies_ms),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "min_latency_ms": min(latencies_ms),
        "max_latency_ms": max(latencies_ms),
        "fps": 1000 / avg_latency if avg_latency > 0 else None,
        "batch_size": 1,
    }
    save_json(summary, args.save_json)
    write_csv(records, args.save_csv)
    print_summary(summary)
    print(f"Summary JSON: {args.save_json}")
    print(f"Per-image CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
