from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from detection_utils import (
    DEFAULT_DETECTION_BATCH_SIZE,
    DEFAULT_DETECTION_DATA_ROOT,
    DEFAULT_DETECTION_EPOCHS,
    DEFAULT_DETECTION_IOU_THRESHOLD,
    DEFAULT_DETECTION_LR,
    DEFAULT_DETECTION_MAX_SIZE,
    DEFAULT_DETECTION_MIN_SIZE,
    DEFAULT_DETECTION_MODEL,
    DEFAULT_DETECTION_OUTPUT_DIR,
    DEFAULT_DETECTION_PATIENCE,
    DEFAULT_DETECTION_HARD_MINING_BOOST,
    DEFAULT_DETECTION_HARD_MINING_TOPK,
    DEFAULT_DETECTION_HARD_MINING_WARMUP,
    DEFAULT_DETECTION_RECORDS_CACHE,
    DEFAULT_DETECTION_RESIZE_LONG_SIDE,
    DEFAULT_DETECTION_SAMPLER_POWER,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_DETECTION_SEED,
    DEFAULT_DETECTION_TEST_RATIO,
    DEFAULT_DETECTION_VAL_RATIO,
    DEFAULT_DETECTION_WEIGHT_DECAY,
    VAIPEMultiPillDetectionDataset,
    build_detection_records,
    build_detector,
    compute_detection_sample_weights,
    create_detection_loader,
    create_label_mappings,
    evaluate_detection_model,
    make_generator,
    normalize_data_roots,
    plot_detection_training_curves,
    save_detection_split_manifest,
    save_json,
    select_hard_example_multipliers,
    set_seed,
    split_detection_records,
    train_detection_epoch,
)


def checkpoint_payload(
    model: torch.nn.Module,
    label_to_index: Dict[int, int],
    best_epoch: int,
    best_val_f1: float,
    args: argparse.Namespace,
    split_manifest_name: str,
) -> Dict[str, object]:
    index_to_label = {index: label_id for label_id, index in label_to_index.items()}
    return {
        "state_dict": model.state_dict(),
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "num_classes": len(label_to_index) + 1,
        "model_name": args.model_name,
        "resize_long_side": args.resize_long_side,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "split_manifest": split_manifest_name,
        "deterministic": bool(args.deterministic),
        "pretrained_backbone": bool(args.pretrained_backbone),
        "sampler_power": float(args.sampler_power),
        "hard_mining_topk": float(args.hard_mining_topk),
        "hard_mining_boost": float(args.hard_mining_boost),
        "hard_mining_warmup": int(args.hard_mining_warmup),
        "data_roots": [str(args.data_root), *[str(root) for root in args.extra_data_root]],
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-pill detector on full VAIPE images.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DETECTION_DATA_ROOT, help="Path to public_train/pill")
    parser.add_argument(
        "--extra-data-root",
        type=Path,
        action="append",
        default=[],
        help="Additional pill dataset roots with matching image/label folders. Repeat this flag to add more sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DETECTION_OUTPUT_DIR, help="Directory for detector outputs")
    parser.add_argument(
        "--records-cache",
        type=Path,
        default=DEFAULT_DETECTION_RECORDS_CACHE,
        help="Path to cached detection metadata built from image-level labels.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_DETECTION_MODEL,
        choices=["fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_resnet50_fpn_v2"],
        help="Torchvision detector backbone.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional detection checkpoint to load before fine-tuning.",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained torchvision weights when available.",
    )
    parser.add_argument("--resize-long-side", type=int, default=DEFAULT_DETECTION_RESIZE_LONG_SIDE, help="Downscale longer side before feeding detector")
    parser.add_argument("--min-size", type=int, default=DEFAULT_DETECTION_MIN_SIZE, help="Detector min_size")
    parser.add_argument("--max-size", type=int, default=DEFAULT_DETECTION_MAX_SIZE, help="Detector max_size")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_DETECTION_BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_DETECTION_EPOCHS, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_DETECTION_PATIENCE, help="Early stopping patience on val F1")
    parser.add_argument("--lr", type=float, default=DEFAULT_DETECTION_LR, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_DETECTION_WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=DEFAULT_DETECTION_SEED, help="Random seed")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_DETECTION_VAL_RATIO, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_DETECTION_TEST_RATIO, help="Test split ratio")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_DETECTION_SCORE_THRESHOLD, help="Score threshold for val/test metrics")
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_DETECTION_IOU_THRESHOLD, help="IoU threshold for TP matching")
    parser.add_argument(
        "--sampler-power",
        type=float,
        default=DEFAULT_DETECTION_SAMPLER_POWER,
        help="Image-level class-balanced sampling strength. Set 0 to disable.",
    )
    parser.add_argument(
        "--hard-mining-topk",
        type=float,
        default=DEFAULT_DETECTION_HARD_MINING_TOPK,
        help="Top ratio of high-loss train images to replay next epoch. Set 0 to disable.",
    )
    parser.add_argument(
        "--hard-mining-boost",
        type=float,
        default=DEFAULT_DETECTION_HARD_MINING_BOOST,
        help="Sampling multiplier applied to selected hard images.",
    )
    parser.add_argument(
        "--hard-mining-warmup",
        type=int,
        default=DEFAULT_DETECTION_HARD_MINING_WARMUP,
        help="Number of initial epochs before hard-example replay starts.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic behavior where possible.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use AMP on CUDA.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--limit-train-images", type=int, default=0, help="Optional train subset for smoke tests")
    parser.add_argument("--limit-val-images", type=int, default=0, help="Optional val subset for smoke tests")
    parser.add_argument("--limit-test-images", type=int, default=0, help="Optional test subset for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("=" * 80, flush=True)
    print("VAIPE MULTI-PILL DETECTION", flush=True)
    print("=" * 80, flush=True)
    data_roots = normalize_data_roots(args.data_root, args.extra_data_root)
    print(f"Device: {device}", flush=True)
    print(f"Data root: {args.data_root}", flush=True)
    if len(data_roots) > 1:
        print("Extra data roots:", flush=True)
        for root in data_roots[1:]:
            print(f"  - {root}", flush=True)
    print(f"Model: {args.model_name}", flush=True)
    print(f"Pretrained backbone: {args.pretrained_backbone}", flush=True)
    print(f"Resize long side: {args.resize_long_side}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Sampler power: {args.sampler_power}", flush=True)
    print(f"Hard mining: topk={args.hard_mining_topk}, boost={args.hard_mining_boost}, warmup={args.hard_mining_warmup}", flush=True)
    print(f"Deterministic: {args.deterministic}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(flush=True)

    records = build_detection_records(data_roots, cache_path=args.records_cache)
    train_records, val_records, test_records = split_detection_records(
        records=records,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    if args.limit_train_images > 0:
        train_records = train_records[: args.limit_train_images]
    if args.limit_val_images > 0:
        val_records = val_records[: args.limit_val_images]
    if args.limit_test_images > 0:
        test_records = test_records[: args.limit_test_images]

    split_manifest_path = save_detection_split_manifest(
        output_dir=args.output_dir,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    label_to_index, index_to_label = create_label_mappings(records)
    dataset_summary = {
        "total_images": len(records),
        "train_images": len(train_records),
        "val_images": len(val_records),
        "test_images": len(test_records),
        "total_boxes": int(sum(int(record["num_boxes"]) for record in records)),
        "train_boxes": int(sum(int(record["num_boxes"]) for record in train_records)),
        "val_boxes": int(sum(int(record["num_boxes"]) for record in val_records)),
        "test_boxes": int(sum(int(record["num_boxes"]) for record in test_records)),
        "num_classes": len(label_to_index),
        "class_distribution": {
            str(label_id): count
            for label_id, count in sorted(Counter(int(label) for record in records for label in record["labels"]).items())
        },
        "split_manifest": str(split_manifest_path),
        "resize_long_side": args.resize_long_side,
        "model_name": args.model_name,
        "data_roots": [str(root) for root in data_roots],
        "sampler_power": float(args.sampler_power),
        "hard_mining_topk": float(args.hard_mining_topk),
        "hard_mining_boost": float(args.hard_mining_boost),
        "hard_mining_warmup": int(args.hard_mining_warmup),
    }
    save_json(args.output_dir / "dataset_summary.json", dataset_summary)

    print(f"Total images: {len(records):,}", flush=True)
    print(f"Total boxes: {dataset_summary['total_boxes']:,}", flush=True)
    print(f"Classes: {len(label_to_index)}", flush=True)
    print(
        f"Split -> train: {len(train_records):,}, val: {len(val_records):,}, test: {len(test_records):,}",
        flush=True,
    )
    print(f"Split manifest: {split_manifest_path}", flush=True)
    print(flush=True)

    train_dataset = VAIPEMultiPillDetectionDataset(
        records=train_records,
        label_to_index=label_to_index,
        resize_long_side=args.resize_long_side,
        is_train=True,
    )
    val_dataset = VAIPEMultiPillDetectionDataset(
        records=val_records,
        label_to_index=label_to_index,
        resize_long_side=args.resize_long_side,
        is_train=False,
    )
    test_dataset = VAIPEMultiPillDetectionDataset(
        records=test_records,
        label_to_index=label_to_index,
        resize_long_side=args.resize_long_side,
        is_train=False,
    )

    val_loader = create_detection_loader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        generator=make_generator(args.seed + 202),
    )
    test_loader = create_detection_loader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        generator=make_generator(args.seed + 303),
    )

    model = build_detector(
        model_name=args.model_name,
        num_classes=len(label_to_index) + 1,
        min_size=args.min_size,
        max_size=args.max_size,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    if args.init_checkpoint is not None:
        init_payload = torch.load(args.init_checkpoint, map_location=device)
        init_state_dict = init_payload.get("state_dict", init_payload)
        model.load_state_dict(init_state_dict)
        print(f"Initialized model from checkpoint: {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_mean_iou": [],
        "lr": [],
        "epoch_time_sec": [],
        "sampler_weight_min": [],
        "sampler_weight_max": [],
        "hard_examples_next_epoch": [],
        "hard_example_loss_threshold": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    train_start = time.perf_counter()
    hard_example_multipliers: Dict[int, float] = {}
    hard_example_summary: Dict[str, float] = {"selected_count": 0.0, "loss_threshold": 0.0, "max_loss": 0.0}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        sample_weights = compute_detection_sample_weights(
            train_records,
            sampler_power=args.sampler_power,
            hard_example_multipliers=hard_example_multipliers,
        )
        train_loader = create_detection_loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=sample_weights is None,
            generator=make_generator(args.seed + 101 + epoch),
            sample_weights=sample_weights,
            num_samples=len(train_dataset),
        )
        train_metrics = train_detection_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler if use_amp else None,
        )
        if epoch >= args.hard_mining_warmup:
            hard_example_multipliers, hard_example_summary = select_hard_example_multipliers(
                train_metrics.get("image_losses", {}),
                topk_ratio=args.hard_mining_topk,
                boost_factor=args.hard_mining_boost,
            )
        else:
            hard_example_multipliers = {}
            hard_example_summary = {"selected_count": 0.0, "loss_threshold": 0.0, "max_loss": 0.0}
        val_metrics = evaluate_detection_model(
            model=model,
            loader=val_loader,
            device=device,
            index_to_label=index_to_label,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
        )
        scheduler.step(val_metrics["f1"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start

        history["train_loss"].append(float(train_metrics["loss"]))
        history["val_precision"].append(float(val_metrics["precision"]))
        history["val_recall"].append(float(val_metrics["recall"]))
        history["val_f1"].append(float(val_metrics["f1"]))
        history["val_mean_iou"].append(float(val_metrics["mean_matched_iou"]))
        history["lr"].append(float(current_lr))
        history["epoch_time_sec"].append(float(epoch_time))
        if sample_weights is None:
            history["sampler_weight_min"].append(1.0)
            history["sampler_weight_max"].append(1.0)
        else:
            history["sampler_weight_min"].append(float(np.min(sample_weights)))
            history["sampler_weight_max"].append(float(np.max(sample_weights)))
        history["hard_examples_next_epoch"].append(float(hard_example_summary["selected_count"]))
        history["hard_example_loss_threshold"].append(float(hard_example_summary["loss_threshold"]))

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_precision={val_metrics['precision']:.2%} | "
            f"val_recall={val_metrics['recall']:.2%} | "
            f"val_f1={val_metrics['f1']:.2%} | "
            f"val_iou={val_metrics['mean_matched_iou']:.2%} | "
            f"sampler=[{history['sampler_weight_min'][-1]:.2f},{history['sampler_weight_max'][-1]:.2f}] | "
            f"next_hard={int(round(hard_example_summary['selected_count']))} | "
            f"lr={current_lr:.2e} | time={epoch_time:.1f}s",
            end="",
            flush=True,
        )

        if float(val_metrics["f1"]) > best_val_f1 + 1e-4:
            best_val_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                checkpoint_payload(
                    model=model,
                    label_to_index=label_to_index,
                    best_epoch=best_epoch,
                    best_val_f1=best_val_f1,
                    args=args,
                    split_manifest_name=split_manifest_path.name,
                ),
                args.output_dir / "best_model.pth",
            )
            print(" | saved best_model.pth", flush=True)
        else:
            patience_counter += 1
            print(f" | patience={patience_counter}/{args.patience}", flush=True)
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.", flush=True)
                break

    total_train_time = time.perf_counter() - train_start

    torch.save(
        checkpoint_payload(
            model=model,
            label_to_index=label_to_index,
            best_epoch=best_epoch,
            best_val_f1=best_val_f1,
            args=args,
            split_manifest_name=split_manifest_path.name,
        ),
        args.output_dir / "final_model.pth",
    )

    best_checkpoint = torch.load(args.output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["state_dict"])
    test_metrics = evaluate_detection_model(
        model=model,
        loader=test_loader,
        device=device,
        index_to_label=index_to_label,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
    )

    history_summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "epochs_ran": len(history["train_loss"]),
        "total_train_time_sec": total_train_time,
        "avg_epoch_time_sec": float(np.mean(history["epoch_time_sec"])) if history["epoch_time_sec"] else 0.0,
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_mean_matched_iou": float(test_metrics["mean_matched_iou"]),
    }

    save_json(args.output_dir / "history.json", {**history, "summary": history_summary})
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    save_json(args.output_dir / "per_class_metrics.json", {"per_class_metrics": test_metrics["per_class_metrics"]})
    plot_detection_training_curves(history, args.output_dir / "training_curves.png", best_epoch=best_epoch)

    print(flush=True)
    print("=" * 80, flush=True)
    print("DETECTION TRAINING COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best val F1: {best_val_f1:.2%}", flush=True)
    print(f"Test precision: {float(test_metrics['precision']):.2%}", flush=True)
    print(f"Test recall: {float(test_metrics['recall']):.2%}", flush=True)
    print(f"Test F1: {float(test_metrics['f1']):.2%}", flush=True)
    print(f"Test mean matched IoU: {float(test_metrics['mean_matched_iou']):.2%}", flush=True)
    print(f"Training time: {total_train_time / 60:.1f} minutes", flush=True)
    print(flush=True)
    print("Saved files:", flush=True)
    print(f"  - {args.output_dir / 'best_model.pth'}", flush=True)
    print(f"  - {args.output_dir / 'final_model.pth'}", flush=True)
    print(f"  - {args.output_dir / 'history.json'}", flush=True)
    print(f"  - {args.output_dir / 'test_metrics.json'}", flush=True)
    print(f"  - {args.output_dir / 'per_class_metrics.json'}", flush=True)
    print(f"  - {args.output_dir / 'training_curves.png'}", flush=True)
    print(f"  - {split_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
