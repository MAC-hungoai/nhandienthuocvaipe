from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from test import load_checkpoint
from train import (
    DEFAULT_COLOR_BINS,
    VAIPECropDataset,
    create_loader,
    evaluate_model_detailed,
    make_generator,
    plot_confusion_matrix,
    prepare_crop_cache,
    save_json,
)


DEFAULT_REAL_DATA_ROOT = Path("data") / "user_real_photos" / "pill"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a classifier checkpoint on labeled real pill photos.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to classifier checkpoint")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_REAL_DATA_ROOT, help="Path to labeled real pill photos")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save metrics and confusion matrix. Defaults to <checkpoint_parent>/real_photo_benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on Windows for maximum stability.",
    )
    return parser.parse_args()


def build_real_photo_summary(records: list[Dict[str, object]]) -> Dict[str, object]:
    class_counts = Counter(int(record["label_id"]) for record in records)
    source_images = {
        str(record.get("source_image_path") or record.get("image_path") or "")
        for record in records
        if str(record.get("source_image_path") or record.get("image_path") or "")
    }
    return {
        "total_crops": len(records),
        "total_images": len(source_images),
        "num_classes": len(class_counts),
        "class_distribution": {str(label_id): count for label_id, count in sorted(class_counts.items())},
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    output_dir = args.output_dir or (args.checkpoint.parent / "real_photo_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, class_to_idx, idx_to_class = load_checkpoint(args.checkpoint, device)
    image_size = int(checkpoint.get("image_size", 160))
    color_bins = int(checkpoint.get("color_bins", DEFAULT_COLOR_BINS))

    cache_dir = output_dir / f"crop_cache_{image_size}"
    records = prepare_crop_cache(args.data_root, cache_dir=cache_dir, image_size=image_size)
    dataset = VAIPECropDataset(
        records=records,
        class_to_idx=class_to_idx,
        image_size=image_size,
        is_train=False,
        color_bins=color_bins,
    )
    loader = create_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=make_generator(int(checkpoint.get("seed", 42)) + 909),
    )

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model_detailed(model, loader, criterion, device, idx_to_class)
    summary = build_real_photo_summary(records)
    payload = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "image_size": image_size,
        "color_bins": color_bins,
        "summary": summary,
        "metrics": metrics,
    }

    save_json(output_dir / "real_photo_metrics.json", payload)
    plot_confusion_matrix(metrics, output_dir / "real_photo_confusion_matrix.png", normalize=True)

    print("=" * 80)
    print("REAL PHOTO BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Images: {summary['total_images']}")
    print(f"Crops: {summary['total_crops']}")
    print(f"Classes: {summary['num_classes']}")
    print(f"Top-1 accuracy: {metrics['accuracy']:.2%}")
    print(f"Top-3 accuracy: {metrics['top3_accuracy']:.2%}")
    print(f"Macro-F1: {metrics['macro_f1']:.2%}")
    print(f"Saved: {output_dir / 'real_photo_metrics.json'}")
    print(f"Saved: {output_dir / 'real_photo_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
