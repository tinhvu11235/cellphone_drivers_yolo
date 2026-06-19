"""Small GUI for manually labeling driver cellphone-use images.

The tool shows one image at a time and writes a CSV with:

image,label
some_image.jpg,1
other_image.jpg,0

Label 1 means the driver is using/listening to a phone. Label 0 means not.
Blank labels are kept for images that have not been reviewed yet.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Any

from PIL import Image, ImageOps, ImageTk


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_IMAGES_DIR = Path(r"D:\cellphone_drivers_yolo\images\test")
DEFAULT_OUTPUT_CSV = Path("configs/driver_phone_usage_labels_template.csv")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="GUI labeler for driver cellphone-use images.")
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_IMAGES_DIR),
        help="Directory containing images to label.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="CSV path to write labels to. Existing 0/1 labels are preserved.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional 1-based image index to start from.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Maximum displayed image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=560,
        help="Maximum displayed image height.",
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    """Return sorted image files from a directory."""
    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"No image files found in: {images_dir}")
    return images


def load_labels(csv_path: Path) -> dict[str, str]:
    """Load existing 0/1 labels from a CSV, ignoring blanks."""
    if not csv_path.exists():
        return {}

    labels: dict[str, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"image", "label"}.issubset(reader.fieldnames):
            raise SystemExit(f"CSV must contain image,label columns: {csv_path}")
        for row in reader:
            image_name = (row.get("image") or "").strip()
            label = (row.get("label") or "").strip()
            if image_name and label in {"0", "1"}:
                labels[image_name] = label
    return labels


def save_labels(csv_path: Path, images: list[Path], labels: dict[str, str]) -> Path:
    """Write labels for every image, keeping blanks for unlabeled images."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    _write_labels_csv(temp_path, images, labels)

    try:
        temp_path.replace(csv_path)
        return csv_path
    except PermissionError:
        # OneDrive, Excel, or file ownership can block atomic replace on Windows.
        # A direct overwrite usually still works; otherwise fall back to outputs/.
        try:
            _write_labels_csv(csv_path, images, labels)
            temp_path.unlink(missing_ok=True)
            return csv_path
        except PermissionError:
            fallback_path = Path("outputs") / csv_path.name
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            _write_labels_csv(fallback_path, images, labels)
            temp_path.unlink(missing_ok=True)
            return fallback_path


def _write_labels_csv(csv_path: Path, images: list[Path], labels: dict[str, str]) -> None:
    """Write the full label CSV to one path."""
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "label"])
        writer.writeheader()
        for image_path in images:
            writer.writerow(
                {
                    "image": image_path.name,
                    "label": labels.get(image_path.name, ""),
                }
            )


class LabelingApp:
    """Tkinter app for fast image-level behavior labeling."""

    def __init__(
        self,
        root: tk.Tk,
        images: list[Path],
        labels: dict[str, str],
        output_csv: Path,
        max_size: tuple[int, int],
        start_index: int | None,
    ) -> None:
        self.root = root
        self.images = images
        self.labels = labels
        self.output_csv = output_csv
        self.max_size = max_size
        self.current_photo: ImageTk.PhotoImage | None = None
        self.index = self._initial_index(start_index)

        self.root.title("Driver Phone Usage Labeler")
        self.root.geometry("1080x760")
        self.root.minsize(820, 620)

        self.status_var = tk.StringVar()
        self.filename_var = tk.StringVar()
        self.current_label_var = tk.StringVar()

        self._build_ui()
        self._bind_keys()
        self.show_current_image()

    def _initial_index(self, start_index: int | None) -> int:
        if start_index is not None:
            return max(0, min(start_index - 1, len(self.images) - 1))
        for index, image_path in enumerate(self.images):
            if image_path.name not in self.labels:
                return index
        return 0

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root, padx=12, pady=10)
        outer.pack(fill=tk.BOTH, expand=True)

        tk.Label(outer, textvariable=self.status_var, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(outer, textvariable=self.filename_var, font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 4))
        tk.Label(outer, textvariable=self.current_label_var, font=("Segoe UI", 10)).pack(anchor="w")

        buttons = tk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(10, 8))

        tk.Button(
            buttons,
            text="0 - Not using phone",
            command=lambda: self.set_label("0"),
            width=24,
            height=2,
            font=("Segoe UI", 11, "bold"),
            bg="#e6f1ff",
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            buttons,
            text="1 - Using phone",
            command=lambda: self.set_label("1"),
            width=24,
            height=2,
            font=("Segoe UI", 11, "bold"),
            bg="#ffe5e5",
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(buttons, text="Back", command=self.previous_image, width=10).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        tk.Button(buttons, text="Next", command=self.next_image, width=10).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        tk.Button(buttons, text="Clear", command=self.clear_label, width=10).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        tk.Button(buttons, text="Save + Quit (Q)", command=self.quit_app, width=14).pack(side=tk.RIGHT)

        image_frame = tk.Frame(outer, bg="#1f1f1f", bd=1, relief=tk.SUNKEN)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.image_label = tk.Label(image_frame, bg="#1f1f1f")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        help_text = (
            "Shortcuts: 0 = not using, 1 = using, Left/B = back, Right/N = next, "
            "C = clear, Q/Esc = save and quit"
        )
        tk.Label(outer, text=help_text, font=("Segoe UI", 9), fg="#555555").pack(anchor="w", pady=(8, 0))

    def _bind_keys(self) -> None:
        self.root.bind("0", lambda _event: self.set_label("0"))
        self.root.bind("1", lambda _event: self.set_label("1"))
        self.root.bind("<Left>", lambda _event: self.previous_image())
        self.root.bind("<Right>", lambda _event: self.next_image())
        self.root.bind("b", lambda _event: self.previous_image())
        self.root.bind("B", lambda _event: self.previous_image())
        self.root.bind("n", lambda _event: self.next_image())
        self.root.bind("N", lambda _event: self.next_image())
        self.root.bind("c", lambda _event: self.clear_label())
        self.root.bind("C", lambda _event: self.clear_label())
        self.root.bind("q", lambda _event: self.quit_app())
        self.root.bind("Q", lambda _event: self.quit_app())
        self.root.bind("<Escape>", lambda _event: self.quit_app())
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def show_current_image(self) -> None:
        image_path = self.images[self.index]
        try:
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image)
            image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
        except Exception as exc:
            messagebox.showerror("Image load error", f"Could not open image:\n{image_path}\n\n{exc}")
            self.next_image()
            return

        self.current_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.current_photo)
        self._update_text()

    def set_label(self, label: str) -> None:
        image_name = self.images[self.index].name
        self.labels[image_name] = label
        self.save()
        self.next_image()

    def clear_label(self) -> None:
        image_name = self.images[self.index].name
        self.labels.pop(image_name, None)
        self.save()
        self._update_text()

    def previous_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.show_current_image()

    def next_image(self) -> None:
        if self.index < len(self.images) - 1:
            self.index += 1
            self.show_current_image()
            return
        self._update_text()
        if self.unlabeled_count() == 0:
            messagebox.showinfo("Done", f"All images are labeled.\nSaved to:\n{self.output_csv}")
        else:
            messagebox.showinfo("End reached", "This is the last image. Use Back to review earlier images.")

    def save(self) -> None:
        self.output_csv = save_labels(self.output_csv, self.images, self.labels)

    def quit_app(self) -> None:
        self.save()
        self.root.destroy()

    def labeled_count(self) -> int:
        image_names = {path.name for path in self.images}
        return sum(1 for image_name in image_names if image_name in self.labels)

    def unlabeled_count(self) -> int:
        return len(self.images) - self.labeled_count()

    def _update_text(self) -> None:
        image_path = self.images[self.index]
        label = self.labels.get(image_path.name)
        label_text = {"0": "0 - not using phone", "1": "1 - using phone"}.get(label, "unlabeled")
        self.status_var.set(
            f"Image {self.index + 1}/{len(self.images)} | "
            f"Labeled {self.labeled_count()}/{len(self.images)} | "
            f"Remaining {self.unlabeled_count()}"
        )
        self.filename_var.set(str(image_path))
        self.current_label_var.set(f"Current label: {label_text}")


def main() -> None:
    """Run the labeler."""
    args = parse_args()
    images_dir = Path(args.images_dir)
    output_csv = Path(args.output_csv)
    images = list_images(images_dir)
    labels = load_labels(output_csv)

    root = tk.Tk()
    app = LabelingApp(
        root=root,
        images=images,
        labels=labels,
        output_csv=output_csv,
        max_size=(args.width, args.height),
        start_index=args.start_index,
    )
    app.save()
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
