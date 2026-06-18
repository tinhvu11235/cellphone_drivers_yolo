"""Shared utilities for the driver cellphone-use YOLO project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only before dependencies are installed.
    yaml = None


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for empty files."""
    yaml_path = Path(path)
    text = yaml_path.read_text(encoding="utf-8")
    if yaml is None:
        return _load_simple_yaml(text)
    data = yaml.safe_load(text)
    return data or {}


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by this project when PyYAML is unavailable."""
    data: dict[str, Any] = {}
    current_map: dict[Any, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith((" ", "\t")):
            current_map = None
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                current_map = {}
                data[key] = current_map
            else:
                data[key] = _parse_simple_yaml_scalar(value)
            continue

        if current_map is None or ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        current_map[_parse_simple_yaml_scalar(key.strip())] = _parse_simple_yaml_scalar(value.strip())

    return data


def _parse_simple_yaml_scalar(value: str) -> Any:
    """Parse a scalar from the project's simple YAML files."""
    if value in ("", "null", "Null", "NULL", "~"):
        return None
    if value in ("true", "True", "TRUE"):
        return True
    if value in ("false", "False", "FALSE"):
        return False
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def save_json(data: Any, path: str | Path) -> Path:
    """Save JSON data with stable formatting."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    return output_path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_image_files(source: str | Path) -> list[Path]:
    """Return image files from a single image path or a directory."""
    source_path = Path(source)
    if source_path.is_file():
        return [source_path] if source_path.suffix.lower() in IMAGE_EXTENSIONS else []
    if not source_path.exists() or not source_path.is_dir():
        return []
    return sorted(
        path
        for path in source_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_path(base_dir: str | Path, path: str | Path | None) -> Path | None:
    """Resolve a path relative to a base directory when it is not absolute."""
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path(base_dir) / candidate).resolve()


def format_ms(seconds: float | int | None) -> str:
    """Format seconds as milliseconds for human-readable logs."""
    value = safe_float(seconds)
    if value is None:
        return "N/A"
    return f"{value * 1000:.2f} ms"


def get_device_name(device: str | int | None) -> str:
    """Return a device label that includes GPU name when available."""
    if device is None:
        return "auto"
    device_text = str(device)
    if device_text.lower() == "cpu":
        return "CPU"

    try:
        import torch

        if torch.cuda.is_available():
            if device_text.startswith("cuda:"):
                index = int(device_text.split(":", 1)[1])
            else:
                index = int(device_text.split(",", 1)[0])
            return f"CUDA:{index} ({torch.cuda.get_device_name(index)})"
    except Exception:
        pass

    return device_text


def safe_float(value: Any) -> float | None:
    """Convert common numeric values to float, returning None when unavailable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value.item())
        except Exception:
            return None


def as_list(value: Any) -> list[Any]:
    """Convert scalars, tuples, arrays, and tensors to a plain Python list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def first_existing(paths: Iterable[str | Path]) -> Path | None:
    """Return the first path that exists."""
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None
