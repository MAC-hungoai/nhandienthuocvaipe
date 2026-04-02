from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".PNG")
DEFAULT_REAL_DATA_ROOT = Path("data") / "user_real_photos" / "pill"
DEFAULT_OUTPUT_ROOT = Path("data") / "user_real_photo_splits" / "stratified_holdout_v1"


def resolve_image_path(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic adapt-train and holdout splits for labeled real pill photos.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_REAL_DATA_ROOT, help="Path to labeled real pill photos")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output directory for generated splits")
    parser.add_argument("--holdout-ratio", type=float, default=0.2, help="Target holdout ratio per label signature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def read_label_signature(label_path: Path) -> Tuple[int, ...]:
    annotations = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, list):
        raise ValueError(f"Label file must contain a list: {label_path}")
    return tuple(sorted(int(annotation["label"]) for annotation in annotations))


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_pair(image_path: Path, label_path: Path, output_root: Path, split_name: str) -> None:
    image_dir = output_root / split_name / "pill" / "image"
    label_dir = output_root / split_name / "pill" / "label"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_dir / image_path.name)
    shutil.copy2(label_path, label_dir / label_path.name)


def main() -> None:
    args = parse_args()
    image_dir = args.data_root / "image"
    label_dir = args.data_root / "label"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected both '{image_dir}' and '{label_dir}'.")

    grouped: Dict[Tuple[int, ...], List[Tuple[Path, Path]]] = defaultdict(list)
    for label_path in sorted(label_dir.glob("*.json")):
        image_path = resolve_image_path(image_dir, label_path.stem)
        if image_path is None:
            continue
        signature = read_label_signature(label_path)
        grouped[signature].append((image_path, label_path))

    if not grouped:
        raise RuntimeError("No labeled real-photo pairs found.")

    rng = random.Random(args.seed)
    ensure_clean_dir(args.output_root)

    split_summary = {
        "data_root": str(args.data_root),
        "holdout_ratio": float(args.holdout_ratio),
        "seed": int(args.seed),
        "adapt_train": {"images": 0, "label_signatures": {}, "labels": {}},
        "holdout": {"images": 0, "label_signatures": {}, "labels": {}},
    }
    label_counters = {
        "adapt_train": Counter(),
        "holdout": Counter(),
    }

    for signature, pairs in sorted(grouped.items()):
        samples = list(pairs)
        rng.shuffle(samples)
        count = len(samples)
        holdout_count = 0
        if count >= 2:
            holdout_count = max(1, int(round(count * args.holdout_ratio)))
            holdout_count = min(holdout_count, count - 1)

        holdout_pairs = samples[:holdout_count]
        adapt_pairs = samples[holdout_count:]

        for split_name, split_pairs in (("holdout", holdout_pairs), ("adapt_train", adapt_pairs)):
            split_summary[split_name]["label_signatures"][str(signature)] = len(split_pairs)
            split_summary[split_name]["images"] += len(split_pairs)
            for image_path, label_path in split_pairs:
                copy_pair(image_path, label_path, args.output_root, split_name)
                for label_id in signature:
                    label_counters[split_name][int(label_id)] += 1

    split_summary["adapt_train"]["labels"] = {str(label_id): count for label_id, count in sorted(label_counters["adapt_train"].items())}
    split_summary["holdout"]["labels"] = {str(label_id): count for label_id, count in sorted(label_counters["holdout"].items())}

    summary_path = args.output_root / "split_summary.json"
    summary_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print("REAL PHOTO HOLDOUT PREP COMPLETE")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Adapt-train images: {split_summary['adapt_train']['images']}")
    print(f"Holdout images: {split_summary['holdout']['images']}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
