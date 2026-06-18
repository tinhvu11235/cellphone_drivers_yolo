"""Train or fine-tune a YOLO detector with Ultralytics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .utils import load_yaml
except ImportError:
    from utils import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on a custom detection dataset.")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to training config YAML.")
    parser.add_argument("--model", default=None, help="Pretrained YOLO weights, for example yolo11n.pt.")
    parser.add_argument("--data", default=None, help="Path to dataset data.yaml.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None, help="Device, for example 0 or cpu.")
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    return parser.parse_args()


def load_config(config_path: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    """Load defaults and apply CLI overrides."""
    config = load_yaml(config_path)
    override_keys = [
        "model",
        "data",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "project",
        "name",
        "patience",
        "seed",
        "fraction",
    ]
    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def print_artifact_locations(run_dir: Path | None) -> None:
    """Print common Ultralytics training artifacts."""
    if run_dir is None:
        print("Training completed, but the run directory could not be inferred.")
        return

    weights_dir = run_dir / "weights"
    print("\nTraining artifacts")
    print("=" * 24)
    print(f"run directory: {run_dir}")
    print(f"best.pt: {weights_dir / 'best.pt'}")
    print(f"last.pt: {weights_dir / 'last.pt'}")
    for artifact in [
        "results.png",
        "confusion_matrix.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "args.yaml",
    ]:
        print(f"{artifact}: {run_dir / artifact}")


def main() -> None:
    """Run YOLO fine-tuning."""
    args = parse_args()
    config = load_config(args.config, args)

    data_path = Path(str(config["data"]))
    if not data_path.exists():
        print(
            f"Note: dataset YAML was not found locally: {data_path}\n"
            "This is expected when preparing locally. On Kaggle, pass the Kaggle dataset data.yaml path."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: install ultralytics before training.") from exc

    print("Loading pretrained model for transfer learning/fine-tuning:")
    print(f"- model: {config['model']}")
    print(f"- data: {config['data']}")

    model = YOLO(str(config["model"]))
    results = model.train(
        data=str(config["data"]),
        epochs=int(config["epochs"]),
        imgsz=int(config["imgsz"]),
        batch=int(config["batch"]),
        device=config["device"],
        project=str(config["project"]),
        name=str(config["name"]),
        patience=int(config["patience"]),
        seed=int(config["seed"]),
        fraction=float(config.get("fraction", 1.0)),
        val=True,
        plots=True,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir is None and getattr(model, "trainer", None) is not None:
        save_dir = getattr(model.trainer, "save_dir", None)
    print_artifact_locations(Path(save_dir) if save_dir else None)


if __name__ == "__main__":
    main()
