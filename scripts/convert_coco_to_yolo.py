"""Convert a Roboflow-style COCO detection dataset to YOLO format."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert COCO annotations to Ultralytics YOLO format.")
    parser.add_argument("--source", required=True, help="COCO dataset root with train/valid/test folders.")
    parser.add_argument("--output", default="datasets/cellphone_drivers_yolo", help="Output YOLO dataset directory.")
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy images into the YOLO output directory.",
    )
    parser.add_argument("--report", default=None, help="Optional JSON conversion report path.")
    return parser.parse_args()


def load_coco(annotation_path: Path) -> dict[str, Any]:
    """Load one COCO annotation JSON file."""
    with annotation_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_categories(coco_by_split: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect a stable category list across all splits."""
    categories: dict[int, str] = {}
    for coco in coco_by_split.values():
        for category in coco.get("categories", []):
            category_id = int(category["id"])
            categories.setdefault(category_id, str(category["name"]))
            if categories[category_id] != str(category["name"]):
                raise ValueError(f"Category id {category_id} has multiple names across splits.")
    return [{"id": category_id, "name": categories[category_id]} for category_id in sorted(categories)]


def clip_box(x: float, y: float, width: float, height: float, image_width: float, image_height: float) -> tuple[float, float, float, float] | None:
    """Clip a COCO xywh box to image bounds and return xyxy."""
    x1 = max(0.0, min(float(x), image_width))
    y1 = max(0.0, min(float(y), image_height))
    x2 = max(0.0, min(float(x) + float(width), image_width))
    y2 = max(0.0, min(float(y) + float(height), image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def coco_box_to_yolo(annotation: dict[str, Any], image: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Convert one COCO bbox to normalized YOLO xywh."""
    image_width = float(image["width"])
    image_height = float(image["height"])
    if image_width <= 0 or image_height <= 0:
        return None

    clipped = clip_box(*annotation["bbox"], image_width=image_width, image_height=image_height)
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    box_width = x2 - x1
    box_height = y2 - y1
    x_center = x1 + box_width / 2.0
    y_center = y1 + box_height / 2.0
    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def write_data_yaml(output_root: Path, categories: list[dict[str, Any]]) -> Path:
    """Write a portable Ultralytics data.yaml."""
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
    ]
    if (output_root / "images" / "test").exists():
        lines.append("test: images/test")
    lines.append("names:")
    for yolo_id, category in enumerate(categories):
        lines.append(f"  {yolo_id}: {category['name']}")

    yaml_path = output_root / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def convert_split(
    source_root: Path,
    output_root: Path,
    source_split: str,
    yolo_split: str,
    coco: dict[str, Any],
    category_to_yolo: dict[int, int],
    copy_images: bool,
) -> dict[str, Any]:
    """Convert one dataset split."""
    image_dir = source_root / source_split
    output_image_dir = output_root / "images" / yolo_split
    output_label_dir = output_root / "labels" / yolo_split
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    skipped_boxes = 0
    missing_images = 0
    written_boxes = 0
    used_names: set[str] = set()

    for image_id, image in sorted(images_by_id.items()):
        file_name = str(image["file_name"])
        source_image = image_dir / file_name
        if not source_image.exists():
            source_image = image_dir / Path(file_name).name
        if not source_image.exists():
            missing_images += 1
            continue

        output_name = Path(file_name).name
        if output_name in used_names:
            output_name = f"{Path(output_name).stem}_{image_id}{Path(output_name).suffix}"
        used_names.add(output_name)

        if copy_images:
            shutil.copy2(source_image, output_image_dir / output_name)

        label_lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            category_id = int(annotation["category_id"])
            if category_id not in category_to_yolo:
                skipped_boxes += 1
                continue

            yolo_box = coco_box_to_yolo(annotation, image)
            if yolo_box is None:
                skipped_boxes += 1
                continue

            values = " ".join(f"{value:.6f}" for value in yolo_box)
            label_lines.append(f"{category_to_yolo[category_id]} {values}")
            written_boxes += 1

        label_path = output_label_dir / f"{Path(output_name).stem}.txt"
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    return {
        "source_split": source_split,
        "yolo_split": yolo_split,
        "images": len(images_by_id),
        "annotations": len(coco.get("annotations", [])),
        "written_boxes": written_boxes,
        "skipped_boxes": skipped_boxes,
        "missing_images": missing_images,
    }


def discover_splits(source_root: Path) -> dict[str, Path]:
    """Find available COCO annotation files under the dataset root."""
    discovered = {}
    for source_split, yolo_split in SPLIT_MAP.items():
        annotation_path = source_root / source_split / "_annotations.coco.json"
        if annotation_path.exists():
            discovered[source_split] = annotation_path
    return discovered


def main() -> None:
    """Run conversion."""
    args = parse_args()
    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not source_root.exists():
        raise SystemExit(f"Source dataset root not found: {source_root}")

    annotation_paths = discover_splits(source_root)
    if not annotation_paths:
        raise SystemExit(f"No _annotations.coco.json files found under: {source_root}")
    if "train" not in annotation_paths or "valid" not in annotation_paths and "val" not in annotation_paths:
        raise SystemExit("Expected at least train and valid/val COCO splits.")

    coco_by_split = {split: load_coco(path) for split, path in annotation_paths.items()}
    categories = collect_categories(coco_by_split)
    category_to_yolo = {int(category["id"]): index for index, category in enumerate(categories)}

    output_root.mkdir(parents=True, exist_ok=True)
    split_reports = []
    for source_split, coco in coco_by_split.items():
        split_reports.append(
            convert_split(
                source_root=source_root,
                output_root=output_root,
                source_split=source_split,
                yolo_split=SPLIT_MAP[source_split],
                coco=coco,
                category_to_yolo=category_to_yolo,
                copy_images=args.copy_images,
            )
        )

    yaml_path = write_data_yaml(output_root, categories)
    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "data_yaml": str(yaml_path),
        "classes": {str(index): category["name"] for index, category in enumerate(categories)},
        "splits": split_reports,
    }

    report_path = Path(args.report).expanduser() if args.report else output_root / "conversion_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("COCO to YOLO conversion completed.")
    print(f"data.yaml: {yaml_path}")
    print(f"report: {report_path}")
    for split_report in split_reports:
        print(
            f"- {split_report['source_split']} -> {split_report['yolo_split']}: "
            f"{split_report['images']} images, {split_report['written_boxes']} boxes, "
            f"{split_report['skipped_boxes']} skipped boxes"
        )


if __name__ == "__main__":
    main()
