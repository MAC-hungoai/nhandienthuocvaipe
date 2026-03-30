from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import cv2


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".PNG")
DEFAULT_REAL_DATA_ROOT = Path("data") / "user_real_photos" / "pill"


def resolve_image_path(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def safe_box(annotation: dict, width: int, height: int) -> Tuple[int, int, int, int] | None:
    x = int(annotation.get("x", 0))
    y = int(annotation.get("y", 0))
    w = int(annotation.get("w", 0))
    h = int(annotation.get("h", 0))
    if w < 2 or h < 2:
        return None

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a user-provided pill dataset before fine-tuning.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_REAL_DATA_ROOT, help="Path to custom pill dataset root")
    parser.add_argument("--show-errors", type=int, default=20, help="Maximum number of errors to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.data_root / "image"
    label_dir = args.data_root / "label"

    print("=" * 80)
    print("VALIDATE REAL PHOTO DATASET")
    print("=" * 80)
    print(f"Data root: {args.data_root}")

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Expected both '{image_dir}' and '{label_dir}'. Create those folders and place your images/labels there."
        )

    label_files = sorted(label_dir.glob("*.json"))
    if not label_files:
        raise RuntimeError("No JSON labels found. Add label files first, then run validation again.")

    errors: List[str] = []
    warnings: List[str] = []
    class_counts: Counter[int] = Counter()
    total_boxes = 0
    matched_images = 0

    for label_path in label_files:
        image_path = resolve_image_path(image_dir, label_path.stem)
        if image_path is None:
            errors.append(f"{label_path.name}: missing matching image file in {image_dir}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            errors.append(f"{label_path.name}: image could not be read -> {image_path.name}")
            continue
        matched_images += 1
        height, width = image.shape[:2]

        try:
            with open(label_path, "r", encoding="utf-8") as handle:
                annotations = json.load(handle)
        except Exception as exc:
            errors.append(f"{label_path.name}: invalid JSON ({exc})")
            continue

        if not isinstance(annotations, list):
            errors.append(f"{label_path.name}: root JSON value must be a list of annotations")
            continue
        if not annotations:
            warnings.append(f"{label_path.name}: label file is empty")
            continue

        for ann_index, ann in enumerate(annotations):
            if not isinstance(ann, dict):
                errors.append(f"{label_path.name}[{ann_index}]: annotation must be an object")
                continue

            missing_keys = [key for key in ("x", "y", "w", "h", "label") if key not in ann]
            if missing_keys:
                errors.append(f"{label_path.name}[{ann_index}]: missing keys {missing_keys}")
                continue

            try:
                label_id = int(ann["label"])
            except Exception:
                errors.append(f"{label_path.name}[{ann_index}]: label must be an integer")
                continue

            if safe_box(ann, width=width, height=height) is None:
                errors.append(f"{label_path.name}[{ann_index}]: invalid or out-of-bounds box")
                continue

            total_boxes += 1
            class_counts[label_id] += 1

    print(f"Images with labels: {matched_images}")
    print(f"Label files: {len(label_files)}")
    print(f"Valid boxes: {total_boxes}")
    print(f"Classes seen: {len(class_counts)}")

    if class_counts:
        print("Top labels:")
        for label_id, count in class_counts.most_common(10):
            print(f"  - {label_id}: {count}")

    if warnings:
        print(f"Warnings: {len(warnings)}")
        for message in warnings[: min(10, len(warnings))]:
            print(f"  - {message}")

    if errors:
        print(f"Errors: {len(errors)}")
        for message in errors[: max(1, args.show_errors)]:
            print(f"  - {message}")
        raise SystemExit(1)

    print("Dataset looks valid. You can start fine-tuning now.")


if __name__ == "__main__":
    main()
