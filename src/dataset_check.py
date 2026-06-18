"""Validate a YOLO-format detection dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .utils import IMAGE_EXTENSIONS, ensure_dir, load_yaml, save_json
except ImportError:
    from utils import IMAGE_EXTENSIONS, ensure_dir, load_yaml, save_json


MAX_INVALID_DETAILS = 100


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset.")
    parser.add_argument("--data", required=True, help="Path to data.yaml.")
    parser.add_argument(
        "--output",
        default="outputs/dataset_report.json",
        help="Path to save the JSON validation report.",
    )
    return parser.parse_args()


def normalize_names(names: Any) -> dict[int, str]:
    """Normalize YOLO class names from list or dict into {id: name}."""
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        normalized: dict[int, str] = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return dict(sorted(normalized.items()))
    return {}


def dataset_root_from_yaml(config: dict[str, Any], yaml_path: Path) -> Path:
    """Resolve the dataset root according to Ultralytics data.yaml conventions."""
    root = config.get("path")
    if root in (None, ""):
        return yaml_path.parent.resolve()
    root_path = Path(str(root)).expanduser()
    if root_path.is_absolute():
        return root_path
    candidate = (yaml_path.parent / root_path).resolve()
    if candidate.exists():
        return candidate

    # Some dataset copies keep the Ultralytics YAML inside the dataset root while
    # the YAML still contains a download-time parent path from the original export.
    # In that case, prefer the YAML parent when the declared split paths exist.
    for split_name in ("train", "val"):
        split_value = config.get(split_name)
        split_items = split_value if isinstance(split_value, list) else [split_value]
        for item in split_items:
            if item in (None, ""):
                continue
            split_path = Path(str(item)).expanduser()
            if not split_path.is_absolute() and (yaml_path.parent / split_path).exists():
                return yaml_path.parent.resolve()

    return candidate


def resolve_split_paths(value: Any, dataset_root: Path) -> list[Path]:
    """Resolve a train/val split that may be a string or list of strings."""
    if value in (None, ""):
        return []
    raw_values = value if isinstance(value, list) else [value]
    resolved = []
    for item in raw_values:
        path = Path(str(item)).expanduser()
        resolved.append(path if path.is_absolute() else (dataset_root / path).resolve())
    return resolved


def read_image_list(list_file: Path) -> list[Path]:
    """Read image paths from a YOLO split text file."""
    images: list[Path] = []
    try:
        lines = list_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return images

    for line in lines:
        line = line.strip()
        if not line:
            continue
        image_path = Path(line).expanduser()
        if not image_path.is_absolute():
            image_path = (list_file.parent / image_path).resolve()
        images.append(image_path)
    return images


def collect_images(split_paths: list[Path]) -> tuple[list[Path], list[str]]:
    """Collect image files from split directories, image files, or text lists."""
    images: list[Path] = []
    issues: list[str] = []
    for split_path in split_paths:
        if split_path.is_dir():
            images.extend(
                path
                for path in split_path.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif split_path.is_file() and split_path.suffix.lower() == ".txt":
            images.extend(read_image_list(split_path))
        elif split_path.is_file() and split_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(split_path)
        else:
            issues.append(f"Missing or unsupported image split path: {split_path}")
    return sorted(set(images)), issues


def infer_label_path_for_image(image_path: Path) -> Path:
    """Infer a label file path from an image path by replacing images with labels."""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def infer_label_dirs(split_paths: list[Path]) -> list[Path]:
    """Infer label directories from image split directories when possible."""
    label_dirs: list[Path] = []
    for split_path in split_paths:
        if not split_path.is_dir():
            continue
        parts = list(split_path.parts)
        replaced = False
        for index in range(len(parts) - 1, -1, -1):
            if parts[index].lower() == "images":
                parts[index] = "labels"
                label_dirs.append(Path(*parts))
                replaced = True
                break
        if not replaced:
            label_dirs.append(split_path.parent / "labels" / split_path.name)
    return sorted(set(label_dirs))


def collect_label_files(label_dirs: list[Path]) -> tuple[list[Path], list[str]]:
    """Collect label text files from inferred label directories."""
    labels: list[Path] = []
    issues: list[str] = []
    for label_dir in label_dirs:
        if label_dir.exists() and label_dir.is_dir():
            labels.extend(path for path in label_dir.rglob("*.txt") if path.is_file())
        else:
            issues.append(f"Missing inferred label directory: {label_dir}")
    return sorted(set(labels)), issues


def validate_label_file(
    label_path: Path,
    class_ids: set[int],
    distribution: Counter[int],
    invalid_lines: list[dict[str, Any]],
) -> tuple[int, int]:
    """Validate one YOLO label file and update class distribution."""
    try:
        raw_text = label_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = label_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        _record_invalid(invalid_lines, label_path, 0, f"Could not read label file: {exc}", "")
        return 0, 1

    stripped = raw_text.strip()
    if not stripped:
        return 1, 0

    invalid_count = 0
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            invalid_count += 1
            _record_invalid(invalid_lines, label_path, line_number, "Empty label line", line)
            continue

        parts = line.split()
        if len(parts) != 5:
            invalid_count += 1
            _record_invalid(
                invalid_lines,
                label_path,
                line_number,
                "Expected 5 values: class_id x_center y_center width height",
                line,
            )
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            invalid_count += 1
            _record_invalid(invalid_lines, label_path, line_number, "class_id is not an integer", line)
            continue

        if class_id not in class_ids:
            invalid_count += 1
            _record_invalid(invalid_lines, label_path, line_number, f"Invalid class_id {class_id}", line)
            continue

        try:
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            invalid_count += 1
            _record_invalid(invalid_lines, label_path, line_number, "Coordinates are not numeric", line)
            continue

        if any(value < 0.0 or value > 1.0 for value in coords):
            invalid_count += 1
            _record_invalid(invalid_lines, label_path, line_number, "Coordinates must be normalized to [0, 1]", line)
            continue

        if coords[2] <= 0.0 or coords[3] <= 0.0:
            invalid_count += 1
            _record_invalid(invalid_lines, label_path, line_number, "Width and height must be > 0", line)
            continue

        distribution[class_id] += 1

    return 0, invalid_count


def _record_invalid(
    invalid_lines: list[dict[str, Any]],
    label_path: Path,
    line_number: int,
    reason: str,
    line: str,
) -> None:
    """Record invalid label details up to a cap to keep reports readable."""
    if len(invalid_lines) >= MAX_INVALID_DETAILS:
        return
    invalid_lines.append(
        {
            "file": str(label_path),
            "line_number": line_number,
            "reason": reason,
            "line": line,
        }
    )


def validate_dataset(data_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Validate dataset structure and labels, returning a JSON-ready report."""
    yaml_path = Path(data_path).expanduser()
    issues: list[str] = []
    warnings: list[str] = []

    if not yaml_path.exists():
        issues.append(f"data.yaml not found: {yaml_path}")
        report = _empty_report(yaml_path, issues, warnings)
        save_json(report, output_path)
        return report

    try:
        config = load_yaml(yaml_path)
    except Exception as exc:
        issues.append(str(exc))
        report = _empty_report(yaml_path, issues, warnings)
        save_json(report, output_path)
        return report
    required_keys = ["train", "val", "names"]
    for key in required_keys:
        if key not in config:
            issues.append(f"Missing required key in data.yaml: {key}")

    names = normalize_names(config.get("names"))
    if not names:
        issues.append("Class names must be a list or a dict in data.yaml.")
    class_ids = set(names.keys())
    dataset_root = dataset_root_from_yaml(config, yaml_path)

    split_reports: dict[str, Any] = {}
    invalid_lines: list[dict[str, Any]] = []
    distribution: Counter[int] = Counter()
    total_invalid_lines = 0
    total_empty_label_files = 0

    split_names = ["train", "val"]
    if config.get("test") not in (None, ""):
        split_names.append("test")

    for split_name in split_names:
        split_paths = resolve_split_paths(config.get(split_name), dataset_root)
        images, image_issues = collect_images(split_paths)
        issues.extend(image_issues)

        label_dirs = infer_label_dirs(split_paths)
        labels, label_issues = collect_label_files(label_dirs)
        warnings.extend(label_issues)

        empty_label_files = 0
        invalid_count = 0
        for label_path in labels:
            empty_count, file_invalid_count = validate_label_file(
                label_path,
                class_ids,
                distribution,
                invalid_lines,
            )
            empty_label_files += empty_count
            invalid_count += file_invalid_count

        total_empty_label_files += empty_label_files
        total_invalid_lines += invalid_count

        missing_label_files = [
            str(infer_label_path_for_image(image_path))
            for image_path in images
            if not infer_label_path_for_image(image_path).exists()
        ]
        if missing_label_files:
            warnings.append(
                f"{split_name}: {len(missing_label_files)} image(s) do not have matching label files."
            )

        split_reports[split_name] = {
            "image_paths": [str(path) for path in split_paths],
            "label_dirs": [str(path) for path in label_dirs],
            "image_count": len(images),
            "label_count": len(labels),
            "empty_label_files": empty_label_files,
            "invalid_label_lines": invalid_count,
            "missing_label_files_count": len(missing_label_files),
            "missing_label_files_sample": missing_label_files[:25],
        }

    if not split_reports.get("train", {}).get("image_count") or not split_reports.get("val", {}).get("image_count"):
        warnings.append(
            "Full dataset images were not found locally. You can still train on Kaggle by pointing --data to the Kaggle data.yaml."
        )

    class_distribution = {
        str(class_id): {"name": names.get(class_id), "instances": distribution.get(class_id, 0)}
        for class_id in sorted(names)
    }

    report = {
        "data_yaml": str(yaml_path),
        "dataset_root": str(dataset_root),
        "required_keys_present": {key: key in config for key in required_keys},
        "class_count": len(names),
        "class_names": {str(key): value for key, value in names.items()},
        "splits": split_reports,
        "empty_label_files_total": total_empty_label_files,
        "invalid_label_lines_total": total_invalid_lines,
        "invalid_label_lines_sample": invalid_lines,
        "invalid_label_lines_sample_truncated": total_invalid_lines > len(invalid_lines),
        "class_distribution": class_distribution,
        "issues": issues,
        "warnings": warnings,
    }
    save_json(report, output_path)
    return report


def _empty_report(yaml_path: Path, issues: list[str], warnings: list[str]) -> dict[str, Any]:
    """Build a minimal report when data.yaml cannot be loaded."""
    return {
        "data_yaml": str(yaml_path),
        "dataset_root": None,
        "required_keys_present": {},
        "class_count": 0,
        "class_names": {},
        "splits": {},
        "empty_label_files_total": 0,
        "invalid_label_lines_total": 0,
        "invalid_label_lines_sample": [],
        "class_distribution": {},
        "issues": issues,
        "warnings": warnings,
    }


def print_summary(report: dict[str, Any], output_path: str | Path) -> None:
    """Print a concise terminal summary."""
    print("\nDataset Validation Summary")
    print("=" * 34)
    print(f"data.yaml: {report['data_yaml']}")
    print(f"dataset root: {report.get('dataset_root')}")
    print(f"classes: {report.get('class_count', 0)}")

    for split_name, split_report in report.get("splits", {}).items():
        print(
            f"{split_name}: {split_report['image_count']} images, "
            f"{split_report['label_count']} labels, "
            f"{split_report['empty_label_files']} empty label files, "
            f"{split_report['invalid_label_lines']} invalid label lines"
        )

    print(f"total invalid label lines: {report.get('invalid_label_lines_total', 0)}")
    print(f"report saved to: {Path(output_path)}")

    if report.get("issues"):
        print("\nIssues:")
        for issue in report["issues"]:
            print(f"- {issue}")
    if report.get("warnings"):
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    if report.get("invalid_label_lines_sample"):
        print("\nInvalid label line samples:")
        for item in report["invalid_label_lines_sample"][:10]:
            print(f"- {item['file']}:{item['line_number']} - {item['reason']}")


def main() -> None:
    """Run dataset validation."""
    args = parse_args()
    ensure_dir(Path(args.output).parent)
    report = validate_dataset(args.data, args.output)
    print_summary(report, args.output)


if __name__ == "__main__":
    main()
