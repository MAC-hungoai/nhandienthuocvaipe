"""
VAIPE pill evaluation and single-image inference.

Usage:
  python test.py
  python test.py --image path/to/pill_crop.jpg
  python test.py --image "archive (1)/public_train/pill/image/VAIPE_P_0_0.jpg" --label-json "archive (1)/public_train/pill/label/VAIPE_P_0_0.json" --box-index 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from check_model_metrics import build_accuracy_diagnostics
from train import (
    DEFAULT_COLOR_BINS,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_MANIFEST,
    VAIPECropDataset,
    build_model,
    build_transforms,
    create_loader,
    crop_with_padding,
    evaluate_model_detailed,
    extract_color_histogram,
    forward_model,
    infer_model_variant_from_state_dict,
    maybe_load_split_manifest,
    plot_confusion_matrix,
    prepare_crop_cache,
    safe_box,
    split_records,
)


class GradCAMHooks:
    def __init__(self, target_module: nn.Module) -> None:
        self.target_module = target_module
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = None
        self.backward_handle = None

    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor | None, ...],
        grad_output: Tuple[torch.Tensor | None, ...],
    ) -> None:
        self.gradients = grad_output[0]

    def __enter__(self) -> "GradCAMHooks":
        self.forward_handle = self.target_module.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_module.register_full_backward_hook(self._backward_hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.forward_handle is not None:
            self.forward_handle.remove()
        if self.backward_handle is not None:
            self.backward_handle.remove()


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    if bool(getattr(model, "uses_color_stream", False)):
        return model.image_backbone.layer4[-1].conv2
    if hasattr(model, "layer4"):
        return model.layer4[-1].conv2
    raise ValueError("Grad-CAM is only supported for the ResNet18-based classifier variants.")


def build_gradcam_map(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    target_size: Tuple[int, int],
) -> np.ndarray:
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy().astype(np.float32)


def overlay_gradcam(image_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap = np.uint8(np.clip(cam, 0.0, 1.0) * 255.0)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.58, heatmap, 0.42, 0.0)


def save_gradcam_preview(
    pil_image: Image.Image,
    gradcam_items: List[Dict[str, object]],
    output_path: Path,
) -> None:
    if not gradcam_items:
        return

    image_rgb = np.asarray(pil_image).astype(np.uint8)
    figure, axes = plt.subplots(1, len(gradcam_items) + 1, figsize=(4.2 * (len(gradcam_items) + 1), 4.8))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    axes[0].imshow(image_rgb)
    axes[0].axis("off")
    axes[0].set_title("Model input", fontsize=12, fontweight="bold")

    for axis, item in zip(axes[1:], gradcam_items):
        overlay = overlay_gradcam(image_rgb, np.asarray(item["cam"], dtype=np.float32))
        axis.imshow(overlay)
        axis.axis("off")
        axis.set_title(
            f"Grad-CAM top {item['rank']}\nlabel {item['label_id']} | {item['probability']:.2%}",
            fontsize=11,
            fontweight="bold",
        )

    figure.suptitle("Grad-CAM explanation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def predict_with_optional_gradcam(
    model: nn.Module,
    tensor: torch.Tensor,
    color_features: torch.Tensor,
    idx_to_class: Dict[int, int],
    device: torch.device,
    top_k: int,
    enable_gradcam: bool,
    gradcam_topk: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    image_batch = tensor.unsqueeze(0).to(device)
    color_batch = color_features.unsqueeze(0).to(device)

    if not enable_gradcam:
        with torch.no_grad():
            logits = forward_model(model, image_batch, color_batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_count = min(top_k, probabilities.shape[0])
            top_probs, top_indices = probabilities.topk(top_count)
        predictions = [
            {
                "rank": rank,
                "label_id": idx_to_class[int(class_index)],
                "probability": float(probability),
                "class_index": int(class_index),
            }
            for rank, (probability, class_index) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1)
        ]
        return predictions, []

    target_layer = get_gradcam_target_layer(model)
    gradcam_items: List[Dict[str, object]] = []
    with GradCAMHooks(target_layer) as hooks:
        logits = forward_model(model, image_batch, color_batch)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_count = min(top_k, probabilities.shape[0])
        top_probs, top_indices = probabilities.topk(top_count)
        predictions = [
            {
                "rank": rank,
                "label_id": idx_to_class[int(class_index)],
                "probability": float(probability),
                "class_index": int(class_index),
            }
            for rank, (probability, class_index) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1)
        ]

        gradcam_count = min(max(0, gradcam_topk), len(predictions))
        for item in predictions[:gradcam_count]:
            model.zero_grad(set_to_none=True)
            score = logits[0, int(item["class_index"])]
            score.backward(retain_graph=True)
            if hooks.activations is None or hooks.gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")
            cam = build_gradcam_map(
                activations=hooks.activations,
                gradients=hooks.gradients,
                target_size=(tensor.shape[-2], tensor.shape[-1]),
            )
            gradcam_items.append(
                {
                    "rank": int(item["rank"]),
                    "label_id": int(item["label_id"]),
                    "probability": float(item["probability"]),
                    "cam": cam,
                }
            )

    for item in predictions:
        item.pop("class_index", None)
    return predictions, gradcam_items


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object], Dict[int, int], Dict[int, int]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "class_to_idx" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    class_to_idx = {int(label_id): int(class_index) for label_id, class_index in checkpoint["class_to_idx"].items()}
    idx_to_class_payload = checkpoint.get("idx_to_class")
    if idx_to_class_payload is None:
        idx_to_class = {class_index: label_id for label_id, class_index in class_to_idx.items()}
    else:
        idx_to_class = {int(class_index): int(label_id) for class_index, label_id in idx_to_class_payload.items()}

    state_dict = checkpoint["state_dict"]
    model_variant = str(checkpoint.get("model_variant") or infer_model_variant_from_state_dict(state_dict))
    color_bins = int(checkpoint.get("color_bins", DEFAULT_COLOR_BINS))
    color_feature_dim = int(checkpoint.get("color_feature_dim", color_bins * 3))

    model = build_model(
        num_classes=len(class_to_idx),
        model_variant=model_variant,
        color_feature_dim=color_feature_dim,
        pretrained=False,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    checkpoint["model_variant"] = model_variant
    checkpoint["color_bins"] = color_bins
    checkpoint["color_feature_dim"] = color_feature_dim
    return model, checkpoint, class_to_idx, idx_to_class


def repair_manifest_records(records: List[Dict[str, object]], cache_dir: Path) -> List[Dict[str, object]]:
    repaired: List[Dict[str, object]] = []
    cache_image_dir = cache_dir / "images"
    for item in records:
        record = dict(item)
        crop_path = record.get("crop_path")
        if crop_path is not None and not Path(str(crop_path)).exists():
            candidate = cache_image_dir / Path(str(crop_path)).name
            if candidate.exists():
                record["crop_path"] = str(candidate)
        repaired.append(record)
    return repaired


def resolve_test_records(
    checkpoint: Dict[str, object],
    data_root: Path,
    cache_dir: Path,
) -> List[Dict[str, object]]:
    split_manifest_name = str(checkpoint.get("split_manifest", DEFAULT_SPLIT_MANIFEST))
    split_manifest_path = cache_dir.parent / split_manifest_name
    manifest_payload = maybe_load_split_manifest(split_manifest_path)
    if manifest_payload is not None:
        manifest_records = manifest_payload.get("test_records", [])
        if manifest_records:
            repaired_records = repair_manifest_records(list(manifest_records), cache_dir)
            if all(Path(str(item.get("crop_path"))).exists() for item in repaired_records if item.get("crop_path") is not None):
                print(f"Loaded exact test split from {split_manifest_path}")
                return repaired_records

    records = prepare_crop_cache(data_root, cache_dir=cache_dir, image_size=int(checkpoint["image_size"]))

    _, _, test_records = split_records(
        records=records,
        seed=int(checkpoint["seed"]),
        val_ratio=float(checkpoint.get("val_ratio", 0.10)),
        test_ratio=float(checkpoint.get("test_ratio", 0.10)),
    )
    return test_records


def evaluate_saved_model(
    model: nn.Module,
    checkpoint: Dict[str, object],
    class_to_idx: Dict[int, int],
    idx_to_class: Dict[int, int],
    device: torch.device,
    data_root: Path,
    output_dir: Path,
    cache_root: Path,
    batch_size: int,
    num_workers: int,
    support_threshold: int,
) -> None:
    cache_dir = cache_root / f"crop_cache_{int(checkpoint['image_size'])}"
    test_records = resolve_test_records(
        checkpoint=checkpoint,
        data_root=data_root,
        cache_dir=cache_dir,
    )
    image_size = int(checkpoint["image_size"])
    color_bins = int(checkpoint.get("color_bins", DEFAULT_COLOR_BINS))
    dataset = VAIPECropDataset(
        test_records,
        class_to_idx,
        image_size=image_size,
        is_train=False,
        color_bins=color_bins,
    )
    loader = create_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model_detailed(model, loader, criterion, device, idx_to_class)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    per_class_path = output_dir / "per_class_metrics.json"
    with open(per_class_path, "w", encoding="utf-8") as handle:
        json.dump(metrics["per_class_metrics"], handle, indent=2)

    low_support_metrics = summarize_low_support_classes(metrics["per_class_metrics"], support_threshold=support_threshold)
    low_support_path = output_dir / "low_support_analysis.json"
    with open(low_support_path, "w", encoding="utf-8") as handle:
        json.dump(low_support_metrics, handle, indent=2)

    accuracy_diagnostics = build_accuracy_diagnostics(metrics, support_floor=support_threshold)
    accuracy_diagnostics_path = output_dir / "accuracy_diagnostics.json"
    with open(accuracy_diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump(accuracy_diagnostics, handle, indent=2)

    plot_confusion_matrix(metrics, output_dir / "confusion_matrix.png", normalize=True)
    plot_low_support_classes(low_support_metrics, output_dir / "low_support_accuracy.png")

    print("=" * 80)
    print("HELD-OUT TEST SPLIT EVALUATION")
    print("=" * 80)
    print(f"Model variant: {checkpoint.get('model_variant', 'unknown')}")
    print(f"Samples: {metrics['samples']}")
    print(f"Accuracy: {float(metrics['accuracy']):.2%}")
    print(f"Top-3 accuracy: {float(metrics['top3_accuracy']):.2%}")
    print(f"Macro-F1: {float(metrics['macro_f1']):.2%}")
    if accuracy_diagnostics.get("balanced_accuracy") is not None:
        print(f"Balanced accuracy: {float(accuracy_diagnostics['balanced_accuracy']):.2%}")
    if accuracy_diagnostics.get("median_recall") is not None:
        print(f"Median class recall: {float(accuracy_diagnostics['median_recall']):.2%}")
    print(f"Loss: {float(metrics['loss']):.4f}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {per_class_path}")
    print(f"Saved: {low_support_path}")
    print(f"Saved: {accuracy_diagnostics_path}")
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")
    print(f"Saved: {output_dir / 'low_support_accuracy.png'}")
    print()
    print(f"Accuracy verdict: {accuracy_diagnostics['verdict']}")
    reasons = list(accuracy_diagnostics.get("reasons", []))
    if reasons:
        for reason in reasons:
            print(f"  - {reason}")

    zero_recall_classes = list(accuracy_diagnostics.get("zero_recall_classes", []))
    if zero_recall_classes:
        print("Zero-recall classes:")
        for item in zero_recall_classes[:10]:
            support_text = (
                f"support={item['support']}"
                if item.get("support") is not None
                else "support=unknown"
            )
            print(f"  label {item['label_id']}: {support_text}")

    print()
    print("Worst low-support classes:")
    for item in low_support_metrics[:10]:
        confusion_text = ", ".join(
            f"{conf['predicted_label_id']} ({conf['count']})"
            for conf in item["top_confusions"][:3]
        ) or "no confusion"
        print(
            f"  label {item['label_id']}: support={item['support']}, "
            f"acc={item['accuracy']:.2%}, top_confusions={confusion_text}"
        )


def summarize_low_support_classes(
    per_class_metrics: List[Dict[str, object]],
    support_threshold: int = 20,
) -> List[Dict[str, object]]:
    selected = [
        item
        for item in per_class_metrics
        if 0 < int(item["support"]) <= support_threshold
    ]
    selected.sort(key=lambda item: (float(item["accuracy"]), int(item["support"]), int(item["label_id"])))
    return selected


def plot_low_support_classes(low_support_metrics: List[Dict[str, object]], output_path: Path, limit: int = 15) -> None:
    if not low_support_metrics:
        return

    selected = low_support_metrics[:limit]
    labels = [str(item["label_id"]) for item in selected]
    accuracies = [float(item["accuracy"]) for item in selected]
    supports = [int(item["support"]) for item in selected]

    plt.figure(figsize=(12, 7))
    bars = plt.barh(labels, accuracies, color="#DC2626", alpha=0.85)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1.0)
    plt.xlabel("Accuracy")
    plt.ylabel("Low-support label ID")
    plt.title("Lowest-performing low-support classes")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))

    for bar, support in zip(bars, supports):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"n={support}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def read_image_for_inference(
    image_path: Path,
    label_json: Path | None = None,
    box_index: int = 0,
) -> Tuple[Image.Image, int | None]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    true_label = None

    if label_json is not None:
        with open(label_json, "r", encoding="utf-8") as handle:
            annotations: List[Dict[str, int]] = json.load(handle)
        if box_index < 0 or box_index >= len(annotations):
            raise IndexError(f"box_index={box_index} is out of range for {label_json}")

        ann = annotations[box_index]
        box = safe_box(ann, width=image_rgb.shape[1], height=image_rgb.shape[0])
        if box is None:
            raise ValueError(f"Annotation box is invalid in {label_json}")
        image_rgb = crop_with_padding(image_rgb, box)
        true_label = int(ann["label"]) if "label" in ann else None

    pil_image = Image.fromarray(image_rgb)
    return pil_image, true_label


def save_prediction_preview(
    pil_image: Image.Image,
    predictions: List[Dict[str, object]],
    output_path: Path,
    is_correct: bool | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(pil_image)
    ax.axis("off")

    top = predictions[0]
    title = f"Pred: {top['label_id']} ({top['probability']:.2%})"
    if is_correct is True:
        title += " | CORRECT"
    elif is_correct is False:
        title += " | WRONG"
    ax.set_title(title, fontsize=13, fontweight="bold")

    text = "\n".join(
        f"Top {item['rank']}: label {item['label_id']} | {item['probability']:.2%}"
        for item in predictions
    )
    fig.text(
        0.02,
        0.02,
        text,
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def predict_single_image(
    model: nn.Module,
    checkpoint: Dict[str, object],
    idx_to_class: Dict[int, int],
    device: torch.device,
    image_path: Path,
    output_dir: Path,
    true_label: int | None = None,
    label_json: Path | None = None,
    box_index: int = 0,
    top_k: int = 5,
    enable_gradcam: bool = True,
    gradcam_topk: int = 3,
) -> None:
    pil_image, inferred_true_label = read_image_for_inference(
        image_path=image_path,
        label_json=label_json,
        box_index=box_index,
    )
    if true_label is None:
        true_label = inferred_true_label

    image_size = int(checkpoint["image_size"])
    color_bins = int(checkpoint.get("color_bins", DEFAULT_COLOR_BINS))
    tensor = build_transforms(image_size=image_size, is_train=False)(pil_image)
    color_features = torch.from_numpy(extract_color_histogram(np.asarray(pil_image), bins=color_bins))
    predictions, gradcam_items = predict_with_optional_gradcam(
        model=model,
        tensor=tensor,
        color_features=color_features,
        idx_to_class=idx_to_class,
        device=device,
        top_k=top_k,
        enable_gradcam=enable_gradcam,
        gradcam_topk=gradcam_topk,
    )

    predicted_label = int(predictions[0]["label_id"])
    is_correct = None if true_label is None else predicted_label == int(true_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "single_image_prediction.json"
    preview_path = output_dir / "single_image_prediction.png"
    gradcam_path = output_dir / "single_image_gradcam.png"
    payload = {
        "image_path": str(image_path),
        "label_json": str(label_json) if label_json is not None else None,
        "predicted_label": predicted_label,
        "predicted_probability": float(predictions[0]["probability"]),
        "true_label": true_label,
        "is_correct": is_correct,
        "top_k": predictions,
        "gradcam_enabled": bool(enable_gradcam),
        "gradcam_path": str(gradcam_path) if enable_gradcam and gradcam_items else None,
        "gradcam_targets": [
            {
                "rank": int(item["rank"]),
                "label_id": int(item["label_id"]),
                "probability": float(item["probability"]),
            }
            for item in gradcam_items
        ],
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    save_prediction_preview(pil_image, predictions, preview_path, is_correct=is_correct)
    if enable_gradcam and gradcam_items:
        save_gradcam_preview(pil_image.resize((image_size, image_size)), gradcam_items, gradcam_path)

    print("=" * 80)
    print("SINGLE IMAGE PREDICTION")
    print("=" * 80)
    print(f"Model variant: {checkpoint.get('model_variant', 'unknown')}")
    print(f"Image: {image_path}")
    print(f"Predicted label: {predicted_label}")
    print(f"Confidence: {float(predictions[0]['probability']):.2%}")
    if true_label is not None:
        print(f"True label: {true_label}")
        print(f"Result: {'CORRECT' if is_correct else 'WRONG'}")
    print(f"Saved JSON: {result_path}")
    print(f"Saved preview: {preview_path}")
    if enable_gradcam and gradcam_items:
        print(f"Saved Grad-CAM: {gradcam_path}")
    print()
    for item in predictions:
        print(f"Top {item['rank']}: label {item['label_id']} | {item['probability']:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or run inference with the VAIPE classifier.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT_DIR / "best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to public_train/pill")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "analysis_outputs",
        help="Directory for evaluation or prediction outputs",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4, help="DataLoader workers")
    parser.add_argument("--image", type=Path, default=None, help="Path to a single pill image for inference")
    parser.add_argument("--true-label", type=int, default=None, help="Optional ground-truth VAIPE label ID for the image")
    parser.add_argument("--label-json", type=Path, default=None, help="Optional VAIPE label JSON for cropping one bbox from a pill image")
    parser.add_argument("--box-index", type=int, default=0, help="Which annotation box to crop from --label-json")
    parser.add_argument("--top-k", type=int, default=5, help="How many predictions to show for single-image inference")
    parser.add_argument(
        "--gradcam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Grad-CAM visualization for single-image inference.",
    )
    parser.add_argument("--gradcam-topk", type=int, default=3, help="How many top predictions to visualize with Grad-CAM")
    parser.add_argument("--support-threshold", type=int, default=20, help="Tail-class threshold for low-support analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, checkpoint, class_to_idx, idx_to_class = load_checkpoint(args.checkpoint, device)

    if args.image is None:
        evaluate_saved_model(
            model=model,
            checkpoint=checkpoint,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            device=device,
            data_root=args.data_root,
            output_dir=args.output_dir,
            cache_root=args.checkpoint.parent,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            support_threshold=args.support_threshold,
        )
        return

    predict_single_image(
        model=model,
        checkpoint=checkpoint,
        idx_to_class=idx_to_class,
        device=device,
        image_path=args.image,
        output_dir=args.output_dir,
        true_label=args.true_label,
        label_json=args.label_json,
        box_index=args.box_index,
        top_k=args.top_k,
        enable_gradcam=args.gradcam,
        gradcam_topk=args.gradcam_topk,
    )


if __name__ == "__main__":
    main()
