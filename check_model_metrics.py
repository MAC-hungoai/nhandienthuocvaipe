"""
Inspect saved evaluation metrics and warn when headline accuracy may be misleading.

Usage:
  python check_model_metrics.py
  python check_model_metrics.py --metrics checkpoints/test_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def load_json(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def build_accuracy_diagnostics(
    metrics: Dict[str, object],
    support_floor: int,
) -> Dict[str, object]:
    accuracy = float(metrics.get("accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    per_class_metrics = list(metrics.get("per_class_metrics", []))
    per_class_accuracy_map = dict(metrics.get("per_class_accuracy", {}))

    if not per_class_metrics and not per_class_accuracy_map:
        return {
            "balanced_accuracy": None,
            "median_recall": None,
            "dominant_prediction_ratio": None,
            "zero_recall_classes": [],
            "low_recall_classes": [],
            "accuracy_gap_vs_macro_f1": accuracy - macro_f1,
            "reasons": [
                "No per-class detail found, so confusion-matrix-style diagnosis is not possible from this file."
            ],
            "verdict": "NEEDS_MORE_DETAIL",
        }

    support_known = bool(per_class_metrics)
    if per_class_metrics:
        normalized_per_class = [
            {
                "label_id": int(item["label_id"]),
                "support": int(item.get("support", 0)),
                "recall": float(item.get("recall", item.get("accuracy", 0.0))),
                "top_confusions": list(item.get("top_confusions", [])),
            }
            for item in per_class_metrics
        ]
    else:
        normalized_per_class = [
            {
                "label_id": int(label_id),
                "support": None,
                "recall": float(recall),
                "top_confusions": [],
            }
            for label_id, recall in per_class_accuracy_map.items()
        ]

    supports = np.asarray(
        [
            int(item["support"])
            for item in normalized_per_class
            if item["support"] is not None
        ],
        dtype=np.int64,
    )
    recalls = np.asarray([float(item["recall"]) for item in normalized_per_class], dtype=np.float32)

    balanced_accuracy = float(recalls.mean()) if len(recalls) > 0 else 0.0
    median_recall = float(np.median(recalls)) if len(recalls) > 0 else 0.0
    gap_vs_macro_f1 = accuracy - macro_f1
    gap_vs_balanced = accuracy - balanced_accuracy

    zero_recall_classes = [
        {
            "label_id": int(item["label_id"]),
            "support": None if item["support"] is None else int(item["support"]),
        }
        for item in normalized_per_class
        if float(item["recall"]) == 0.0 and (item["support"] is None or int(item["support"]) > 0)
    ]

    low_recall_classes = [
        {
            "label_id": int(item["label_id"]),
            "support": None if item["support"] is None else int(item["support"]),
            "recall": float(item["recall"]),
            "top_confusions": list(item["top_confusions"])[:3],
        }
        for item in normalized_per_class
        if float(item["recall"]) < 0.50
        and (item["support"] is None or int(item["support"]) >= support_floor)
    ]
    low_recall_classes.sort(
        key=lambda item: (
            item["recall"],
            -1 if item["support"] is None else item["support"],
            item["label_id"],
        )
    )

    dominant_prediction_ratio = None
    confusion_matrix_payload = metrics.get("confusion_matrix")
    if confusion_matrix_payload:
        matrix = np.asarray(confusion_matrix_payload, dtype=np.int64)
        total = int(matrix.sum())
        if total > 0:
            dominant_prediction_ratio = float(matrix.sum(axis=0).max() / total)

    reasons: List[str] = []
    if gap_vs_balanced >= 0.10:
        reasons.append(
            f"Accuracy ({accuracy:.2%}) is {gap_vs_balanced:.2%} higher than balanced accuracy ({balanced_accuracy:.2%})."
        )
    if gap_vs_macro_f1 >= 0.10:
        reasons.append(
            f"Accuracy ({accuracy:.2%}) is {gap_vs_macro_f1:.2%} higher than macro-F1 ({macro_f1:.2%})."
        )
    if zero_recall_classes:
        if support_known:
            reasons.append(f"{len(zero_recall_classes)} class(es) have zero recall despite non-zero support.")
        else:
            reasons.append(f"{len(zero_recall_classes)} class(es) have zero recall in per_class_accuracy.")
    if low_recall_classes:
        if support_known:
            reasons.append(
                f"{len(low_recall_classes)} class(es) with support >= {support_floor} still have recall below 50%."
            )
        else:
            reasons.append(f"{len(low_recall_classes)} class(es) have recall below 50%.")
    if dominant_prediction_ratio is not None and dominant_prediction_ratio >= 0.35:
        reasons.append(
            f"One predicted class absorbs {dominant_prediction_ratio:.2%} of all predictions."
        )
    if median_recall < 0.75 and accuracy >= 0.90:
        reasons.append(
            f"Median class recall is only {median_recall:.2%} while headline accuracy is {accuracy:.2%}."
        )

    if gap_vs_balanced >= 0.20 or len(zero_recall_classes) >= 3:
        verdict = "LIKELY_MISLEADING"
    elif reasons:
        verdict = "NEEDS_CONTEXT"
    else:
        verdict = "LOOKS_CONSISTENT"

    return {
        "balanced_accuracy": balanced_accuracy,
        "median_recall": median_recall,
        "dominant_prediction_ratio": dominant_prediction_ratio,
        "zero_recall_classes": zero_recall_classes,
        "low_recall_classes": low_recall_classes[:10],
        "support_known": support_known,
        "accuracy_gap_vs_balanced_accuracy": gap_vs_balanced,
        "accuracy_gap_vs_macro_f1": gap_vs_macro_f1,
        "reasons": reasons,
        "verdict": verdict,
    }


def print_checkpoint_summary(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        print(f"Unsupported checkpoint format: {checkpoint_path}")
        return

    print("Checkpoint metadata:")
    for key in ["model_name", "model_variant", "best_epoch", "best_val_loss", "metrics"]:
        if key in checkpoint:
            print(f"  {key}: {checkpoint[key]}")


def print_history_summary(history_path: Path) -> None:
    history = load_json(history_path)
    if history is None:
        print(f"History not found: {history_path}")
        return

    print("\nTraining history:")
    if "val_acc" in history and history["val_acc"]:
        print(f"  best_val_acc: {max(history['val_acc']):.4f}")
    if "train_acc" in history and history["train_acc"]:
        print(f"  final_train_acc: {history['train_acc'][-1]:.4f}")


def print_dataset_summary(dataset_summary_path: Path) -> None:
    dataset_info = load_json(dataset_summary_path)
    if dataset_info is None:
        print(f"Dataset summary not found: {dataset_summary_path}")
        return

    print("\nDataset info:")
    for key in ["num_classes", "train_samples", "val_samples", "test_samples"]:
        if key in dataset_info:
            print(f"  {key}: {dataset_info[key]}")


def print_metrics_summary(metrics_path: Path, support_floor: int) -> None:
    metrics = load_json(metrics_path)
    if metrics is None:
        print(f"Metrics not found: {metrics_path}")
        return

    diagnostics = build_accuracy_diagnostics(metrics, support_floor=support_floor)

    print("\nEvaluation metrics:")
    for key in ["accuracy", "top3_accuracy", "macro_f1", "loss", "samples"]:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float) and key != "loss":
                print(f"  {key}: {value:.4f} ({value:.2%})")
            else:
                print(f"  {key}: {value}")

    balanced_accuracy = diagnostics.get("balanced_accuracy")
    if balanced_accuracy is not None:
        print(f"  balanced_accuracy: {balanced_accuracy:.4f} ({balanced_accuracy:.2%})")
    median_recall = diagnostics.get("median_recall")
    if median_recall is not None:
        print(f"  median_class_recall: {median_recall:.4f} ({median_recall:.2%})")
    dominant_prediction_ratio = diagnostics.get("dominant_prediction_ratio")
    if dominant_prediction_ratio is not None:
        print(
            f"  dominant_prediction_ratio: {dominant_prediction_ratio:.4f} ({dominant_prediction_ratio:.2%})"
        )

    print("\nAccuracy verdict:")
    print(f"  {diagnostics['verdict']}")
    reasons = diagnostics.get("reasons", [])
    if reasons:
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("  - Accuracy, balanced accuracy, and per-class behavior look aligned.")

    zero_recall_classes = diagnostics.get("zero_recall_classes", [])
    if zero_recall_classes:
        preview = ", ".join(
            (
                f"label {item['label_id']} (n={item['support']})"
                if item["support"] is not None
                else f"label {item['label_id']}"
            )
            for item in zero_recall_classes[:10]
        )
        print(f"\nZero-recall classes: {preview}")

    low_recall_classes = diagnostics.get("low_recall_classes", [])
    if low_recall_classes:
        if diagnostics.get("support_known"):
            print(f"\nWorst supported classes (support >= {support_floor}):")
        else:
            print("\nWorst classes by recall:")
        for item in low_recall_classes:
            confusion_text = ", ".join(
                f"{conf['predicted_label_id']} ({conf['count']})"
                for conf in item["top_confusions"]
            ) or "no confusion detail"
            support_text = (
                f"support={item['support']}, " if item["support"] is not None else ""
            )
            print(
                f"  label {item['label_id']}: {support_text}"
                f"recall={item['recall']:.2%}, top_confusions={confusion_text}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect whether model accuracy is trustworthy.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints") / "best_model.pth",
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("checkpoints") / "test_metrics.json",
        help="Path to test_metrics.json or evaluation_metrics.json.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("checkpoints") / "history.json",
        help="Path to history.json.",
    )
    parser.add_argument(
        "--dataset-summary",
        type=Path,
        default=Path("checkpoints") / "dataset_summary.json",
        help="Path to dataset_summary.json.",
    )
    parser.add_argument(
        "--support-floor",
        type=int,
        default=5,
        help="Only flag low-recall classes with support >= this threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_checkpoint_summary(args.checkpoint)
    print_metrics_summary(args.metrics, support_floor=args.support_floor)
    print_history_summary(args.history)
    print_dataset_summary(args.dataset_summary)


if __name__ == "__main__":
    main()
