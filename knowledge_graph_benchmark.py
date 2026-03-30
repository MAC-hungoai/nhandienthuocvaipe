from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import box_iou

from detection_test import crop_detection_region, resolve_test_records
from detection_utils import (
    DEFAULT_DETECTION_DATA_ROOT,
    DEFAULT_DETECTION_IOU_THRESHOLD,
    DEFAULT_DETECTION_OUTPUT_DIR,
    DEFAULT_DETECTION_RECORDS_CACHE,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    filter_prediction,
    load_detection_checkpoint,
    match_detections,
    prepare_inference_image,
    save_json,
    scale_boxes_back,
)
from knowledge_graph import (
    DEFAULT_KG_ANCHOR_WEIGHT,
    DEFAULT_KG_CANDIDATES,
    DEFAULT_KG_CONTEXT_WEIGHT,
    DEFAULT_KG_MAX_ANCHOR_PROBABILITY,
    DEFAULT_KG_MAX_DETECTOR_SCORE,
    DEFAULT_KG_MIN_CANDIDATE_PROBABILITY,
    DEFAULT_KG_SELECTIVE_OVERRIDE,
    DEFAULT_KG_VISUAL_WEIGHT,
    DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR,
    DEFAULT_KNOWLEDGE_GRAPH_PATH,
    classify_crop_candidates,
    load_classifier_checkpoint,
    load_or_build_knowledge_graph,
    select_candidate_with_graph,
)


def match_boxes_ignore_labels(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    matched_gt: set[int] = set()
    matched_pairs: List[Tuple[int, int]] = []
    matched_ious: List[float] = []

    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return matched_pairs, matched_ious

    order = torch.argsort(pred_scores, descending=True)
    for pred_index in order.tolist():
        remaining_gt = [gt_index for gt_index in range(gt_boxes.shape[0]) if gt_index not in matched_gt]
        if not remaining_gt:
            break
        candidate_boxes = gt_boxes[remaining_gt]
        ious = torch.zeros((len(remaining_gt),), dtype=torch.float32)
        if candidate_boxes.numel() > 0:
            ious = box_iou(pred_boxes[pred_index].unsqueeze(0), candidate_boxes)[0]
        best_local = int(torch.argmax(ious).item()) if ious.numel() > 0 else -1
        best_iou = float(ious[best_local].item()) if ious.numel() > 0 else 0.0
        if best_iou < iou_threshold or best_local < 0:
            continue
        gt_index = remaining_gt[best_local]
        matched_gt.add(gt_index)
        matched_pairs.append((pred_index, gt_index))
        matched_ious.append(best_iou)

    return matched_pairs, matched_ious


def empty_method_stats() -> Dict[str, object]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "predictions": 0,
        "ground_truth_boxes": 0,
        "matched_ious": [],
        "matched_label_pairs": 0,
        "matched_label_correct": 0,
        "matched_label_iou": [],
        "per_class_support": defaultdict(int),
        "per_class_correct": defaultdict(int),
    }


def finalize_method_stats(stats: Dict[str, object]) -> Dict[str, object]:
    tp = int(stats["tp"])
    fp = int(stats["fp"])
    fn = int(stats["fn"])
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    matched_label_pairs = int(stats["matched_label_pairs"])
    matched_label_correct = int(stats["matched_label_correct"])
    per_class_metrics = []
    for label_id in sorted(stats["per_class_support"]):
        support = int(stats["per_class_support"][label_id])
        correct = int(stats["per_class_correct"][label_id])
        per_class_metrics.append(
            {
                "label_id": int(label_id),
                "support": support,
                "correct": correct,
                "label_accuracy": correct / max(1, support),
            }
        )
    return {
        "predictions": int(stats["predictions"]),
        "ground_truth_boxes": int(stats["ground_truth_boxes"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": float(np.mean(stats["matched_ious"])) if stats["matched_ious"] else 0.0,
        "matched_label_pairs": matched_label_pairs,
        "matched_label_correct": matched_label_correct,
        "matched_label_accuracy": matched_label_correct / max(1, matched_label_pairs),
        "mean_iou_matched_label_pairs": float(np.mean(stats["matched_label_iou"])) if stats["matched_label_iou"] else 0.0,
        "per_class_label_accuracy": per_class_metrics,
    }


def update_method_stats(
    stats: Dict[str, object],
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> None:
    tp, fp, fn, matched_ious, _ = match_detections(
        pred_boxes=pred_boxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        iou_threshold=iou_threshold,
    )
    stats["tp"] += int(tp)
    stats["fp"] += int(fp)
    stats["fn"] += int(fn)
    stats["predictions"] += int(pred_boxes.shape[0])
    stats["ground_truth_boxes"] += int(gt_boxes.shape[0])
    stats["matched_ious"].extend(float(value) for value in matched_ious)

    matched_pairs, label_ious = match_boxes_ignore_labels(
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        gt_boxes=gt_boxes,
        iou_threshold=iou_threshold,
    )
    stats["matched_label_pairs"] += len(matched_pairs)
    stats["matched_label_iou"].extend(float(value) for value in label_ious)
    for pred_index, gt_index in matched_pairs:
        gt_label = int(gt_labels[gt_index].item())
        pred_label = int(pred_labels[pred_index].item())
        stats["per_class_support"][gt_label] += 1
        if pred_label == gt_label:
            stats["matched_label_correct"] += 1
            stats["per_class_correct"][gt_label] += 1


def empty_refinement_delta() -> Dict[str, object]:
    return {
        "matched_pairs": 0,
        "changed_predictions": 0,
        "fixed_mistakes": 0,
        "new_errors": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
        "per_class_fixed": defaultdict(int),
        "per_class_new_errors": defaultdict(int),
    }


def update_refinement_delta(
    delta: Dict[str, object],
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
    gt_labels: torch.Tensor,
    matched_pairs: Sequence[Tuple[int, int]],
) -> None:
    for pred_index, gt_index in matched_pairs:
        source_label = int(source_labels[pred_index].item())
        target_label = int(target_labels[pred_index].item())
        gt_label = int(gt_labels[gt_index].item())
        delta["matched_pairs"] += 1
        if source_label != target_label:
            delta["changed_predictions"] += 1
        source_correct = source_label == gt_label
        target_correct = target_label == gt_label
        if not source_correct and target_correct:
            delta["fixed_mistakes"] += 1
            delta["per_class_fixed"][gt_label] += 1
        elif source_correct and not target_correct:
            delta["new_errors"] += 1
            delta["per_class_new_errors"][gt_label] += 1
        elif source_correct and target_correct:
            delta["unchanged_correct"] += 1
        else:
            delta["unchanged_wrong"] += 1


def finalize_refinement_delta(delta: Dict[str, object]) -> Dict[str, object]:
    fixed = sorted(delta["per_class_fixed"].items(), key=lambda item: (-item[1], item[0]))[:10]
    new_errors = sorted(delta["per_class_new_errors"].items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "matched_pairs": int(delta["matched_pairs"]),
        "changed_predictions": int(delta["changed_predictions"]),
        "fixed_mistakes": int(delta["fixed_mistakes"]),
        "new_errors": int(delta["new_errors"]),
        "unchanged_correct": int(delta["unchanged_correct"]),
        "unchanged_wrong": int(delta["unchanged_wrong"]),
        "net_gain": int(delta["fixed_mistakes"]) - int(delta["new_errors"]),
        "top_fixed_labels": [{"label_id": int(label_id), "count": int(count)} for label_id, count in fixed],
        "top_new_error_labels": [{"label_id": int(label_id), "count": int(count)} for label_id, count in new_errors],
    }


def build_refined_label_sets(
    image_rgb: np.ndarray,
    boxes: Sequence[List[float]],
    detector_labels: Sequence[int],
    detector_scores: Sequence[float],
    classifier_model: torch.nn.Module,
    classifier_checkpoint: Dict[str, object],
    classifier_idx_to_class: Dict[int, int],
    knowledge_graph: Dict[str, object],
    device: torch.device,
    candidate_top_k: int,
    visual_weight: float,
    context_weight: float,
    anchor_weight: float,
    selective_override: bool,
    max_detector_score: float,
    min_candidate_probability: float,
    max_anchor_probability: float,
) -> Tuple[List[int], List[float], List[int], List[float]]:
    classifier_labels = list(detector_labels)
    classifier_probs = list(detector_scores)
    kg_labels = list(detector_labels)
    kg_probs = list(detector_scores)

    order = sorted(range(len(boxes)), key=lambda index: float(detector_scores[index]), reverse=True)
    accepted_labels: List[int] = []
    for index in order:
        crop_rgb = crop_detection_region(image_rgb, boxes[index])
        classifier_candidates = classify_crop_candidates(
            model=classifier_model,
            checkpoint=classifier_checkpoint,
            idx_to_class=classifier_idx_to_class,
            device=device,
            crop_rgb=crop_rgb,
            top_k=candidate_top_k,
        )
        if classifier_candidates:
            classifier_labels[index] = int(classifier_candidates[0]["label_id"])
            classifier_probs[index] = float(classifier_candidates[0]["probability"])

        selection = select_candidate_with_graph(
            crop_rgb=crop_rgb,
            candidates=classifier_candidates,
            graph=knowledge_graph,
            detector_label=int(detector_labels[index]),
            detector_score=float(detector_scores[index]),
            context_labels=accepted_labels,
            visual_weight=visual_weight,
            context_weight=context_weight,
            anchor_weight=anchor_weight,
            selective_override=selective_override,
            max_detector_score=max_detector_score,
            min_candidate_probability=min_candidate_probability,
            max_anchor_probability=max_anchor_probability,
        )
        kg_labels[index] = int(selection["final_label_id"])
        kg_probs[index] = float(selection["final_probability"])
        accepted_labels.append(int(kg_labels[index]))

    return classifier_labels, classifier_probs, kg_labels, kg_probs


def plot_benchmark_comparison(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    methods = list(results.keys())
    labels = [
        "Detector only",
        "Detector + classifier",
        "Detector + classifier + KG",
    ]
    f1_scores = [float(results[name]["f1"]) for name in methods]
    label_accuracies = [float(results[name]["matched_label_accuracy"]) for name in methods]
    precisions = [float(results[name]["precision"]) for name in methods]
    recalls = [float(results[name]["recall"]) for name in methods]

    x = np.arange(len(methods))
    width = 0.22

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].bar(x - width, precisions, width=width, label="Precision", color="#2563EB")
    axes[0].bar(x, recalls, width=width, label="Recall", color="#16A34A")
    axes[0].bar(x + width, f1_scores, width=width, label="F1", color="#EA580C")
    axes[0].set_title("Detection Metrics on Fixed Detector Boxes")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=12, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[0].legend()

    bars = axes[1].bar(x, label_accuracies, color=["#475569", "#7C3AED", "#DC2626"], alpha=0.90)
    axes[1].set_title("IoU-matched Label Accuracy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=12, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    for bar, accuracy in zip(bars, label_accuracies):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{accuracy:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the real impact of knowledge graph reranking on held-out multi-pill detection.")
    parser.add_argument("--detector-checkpoint", type=Path, required=True, help="Path to detector checkpoint")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True, help="Path to single-pill classifier checkpoint")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DETECTION_DATA_ROOT, help="Path to public_train/pill")
    parser.add_argument("--records-cache", type=Path, default=DEFAULT_DETECTION_RECORDS_CACHE, help="Cached detection records JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DETECTION_OUTPUT_DIR / "knowledge_graph_benchmark", help="Directory for benchmark outputs")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_DETECTION_SCORE_THRESHOLD, help="Detector confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_DETECTION_IOU_THRESHOLD, help="IoU threshold for matching")
    parser.add_argument("--knowledge-graph-artifact", type=Path, default=DEFAULT_KNOWLEDGE_GRAPH_PATH, help="Path to saved knowledge graph JSON")
    parser.add_argument("--knowledge-graph-cache-dir", type=Path, default=DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR, help="Crop cache used for graph prototypes")
    parser.add_argument("--build-knowledge-graph", action=argparse.BooleanOptionalAction, default=False, help="Rebuild graph artifact before benchmark")
    parser.add_argument("--kg-top-k", type=int, default=DEFAULT_KG_CANDIDATES, help="Classifier top-k candidates before graph reranking")
    parser.add_argument("--kg-visual-weight", type=float, default=DEFAULT_KG_VISUAL_WEIGHT, help="Visual prototype weight")
    parser.add_argument("--kg-context-weight", type=float, default=DEFAULT_KG_CONTEXT_WEIGHT, help="Prescription context weight")
    parser.add_argument("--kg-anchor-weight", type=float, default=DEFAULT_KG_ANCHOR_WEIGHT, help="Detector anchor compatibility weight")
    parser.add_argument("--kg-selective-override", action=argparse.BooleanOptionalAction, default=DEFAULT_KG_SELECTIVE_OVERRIDE, help="Only allow KG to override detector labels under conservative gating.")
    parser.add_argument("--kg-max-detector-score", type=float, default=DEFAULT_KG_MAX_DETECTOR_SCORE, help="Maximum detector score that still allows KG override.")
    parser.add_argument("--kg-min-candidate-probability", type=float, default=DEFAULT_KG_MIN_CANDIDATE_PROBABILITY, help="Minimum classifier probability required before KG can override detector.")
    parser.add_argument("--kg-max-anchor-probability", type=float, default=DEFAULT_KG_MAX_ANCHOR_PROBABILITY, help="Maximum classifier support for detector label before KG override is blocked.")
    parser.add_argument("--limit-images", type=int, default=0, help="Optional image limit for quick experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector_model, detector_checkpoint, _, detector_index_to_label = load_detection_checkpoint(args.detector_checkpoint, device)
    classifier_model, classifier_checkpoint, _, classifier_idx_to_class = load_classifier_checkpoint(args.classifier_checkpoint, device)

    confusion_metrics_path = args.classifier_checkpoint.parent / "test_metrics.json"
    if not confusion_metrics_path.exists():
        confusion_metrics_path = None
    knowledge_graph = load_or_build_knowledge_graph(
        artifact_path=args.knowledge_graph_artifact,
        data_root=args.data_root,
        cache_dir=args.knowledge_graph_cache_dir,
        image_size=int(classifier_checkpoint["image_size"]),
        color_bins=int(classifier_checkpoint.get("color_bins", 8)),
        confusion_metrics_path=confusion_metrics_path,
        rebuild=bool(args.build_knowledge_graph),
    )

    test_records = resolve_test_records(
        checkpoint=detector_checkpoint,
        checkpoint_path=args.detector_checkpoint,
        data_root=args.data_root,
        records_cache=args.records_cache,
    )
    if args.limit_images > 0:
        test_records = test_records[: args.limit_images]

    method_stats = {
        "detector_only": empty_method_stats(),
        "detector_plus_classifier": empty_method_stats(),
        "detector_plus_classifier_plus_kg": empty_method_stats(),
    }
    refinement_deltas = {
        "classifier_vs_detector": empty_refinement_delta(),
        "kg_vs_classifier": empty_refinement_delta(),
        "kg_vs_detector": empty_refinement_delta(),
    }

    started_at = time.perf_counter()
    total_images = len(test_records)
    for image_index, record in enumerate(test_records, start=1):
        image_path = Path(str(record["image_path"]))
        original_image_rgb, image_tensor, scale = prepare_inference_image(
            image_path=image_path,
            resize_long_side=int(detector_checkpoint["resize_long_side"]),
        )

        with torch.no_grad():
            prediction = detector_model([image_tensor.to(device)])[0]
        pred_boxes_resized, pred_labels_index, pred_scores = filter_prediction(
            prediction=prediction,
            score_threshold=args.score_threshold,
        )
        box_list = scale_boxes_back(pred_boxes_resized, scale=scale)
        detector_labels = [detector_index_to_label[int(label)] for label in pred_labels_index.tolist()]
        detector_scores = [float(score) for score in pred_scores.tolist()]

        classifier_labels, classifier_probs, kg_labels, kg_probs = build_refined_label_sets(
            image_rgb=original_image_rgb,
            boxes=box_list,
            detector_labels=detector_labels,
            detector_scores=detector_scores,
            classifier_model=classifier_model,
            classifier_checkpoint=classifier_checkpoint,
            classifier_idx_to_class=classifier_idx_to_class,
            knowledge_graph=knowledge_graph,
            device=device,
            candidate_top_k=args.kg_top_k,
            visual_weight=args.kg_visual_weight,
            context_weight=args.kg_context_weight,
            anchor_weight=args.kg_anchor_weight,
            selective_override=args.kg_selective_override,
            max_detector_score=args.kg_max_detector_score,
            min_candidate_probability=args.kg_min_candidate_probability,
            max_anchor_probability=args.kg_max_anchor_probability,
        )

        pred_boxes = torch.as_tensor(box_list, dtype=torch.float32).reshape(-1, 4)
        pred_scores_tensor = torch.as_tensor(detector_scores, dtype=torch.float32)
        detector_labels_tensor = torch.as_tensor(detector_labels, dtype=torch.int64)
        classifier_labels_tensor = torch.as_tensor(classifier_labels, dtype=torch.int64)
        kg_labels_tensor = torch.as_tensor(kg_labels, dtype=torch.int64)
        gt_boxes = torch.as_tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
        gt_labels = torch.as_tensor(record["labels"], dtype=torch.int64)

        update_method_stats(
            method_stats["detector_only"],
            pred_boxes=pred_boxes,
            pred_labels=detector_labels_tensor,
            pred_scores=pred_scores_tensor,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_threshold=args.iou_threshold,
        )
        update_method_stats(
            method_stats["detector_plus_classifier"],
            pred_boxes=pred_boxes,
            pred_labels=classifier_labels_tensor,
            pred_scores=pred_scores_tensor,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_threshold=args.iou_threshold,
        )
        update_method_stats(
            method_stats["detector_plus_classifier_plus_kg"],
            pred_boxes=pred_boxes,
            pred_labels=kg_labels_tensor,
            pred_scores=pred_scores_tensor,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_threshold=args.iou_threshold,
        )

        matched_pairs, _ = match_boxes_ignore_labels(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores_tensor,
            gt_boxes=gt_boxes,
            iou_threshold=args.iou_threshold,
        )
        update_refinement_delta(
            refinement_deltas["classifier_vs_detector"],
            source_labels=detector_labels_tensor,
            target_labels=classifier_labels_tensor,
            gt_labels=gt_labels,
            matched_pairs=matched_pairs,
        )
        update_refinement_delta(
            refinement_deltas["kg_vs_classifier"],
            source_labels=classifier_labels_tensor,
            target_labels=kg_labels_tensor,
            gt_labels=gt_labels,
            matched_pairs=matched_pairs,
        )
        update_refinement_delta(
            refinement_deltas["kg_vs_detector"],
            source_labels=detector_labels_tensor,
            target_labels=kg_labels_tensor,
            gt_labels=gt_labels,
            matched_pairs=matched_pairs,
        )

        if image_index % 100 == 0 or image_index == total_images:
            elapsed = time.perf_counter() - started_at
            print(f"Processed {image_index}/{total_images} images | elapsed {elapsed / 60:.1f} min", flush=True)

    finalized_methods = {name: finalize_method_stats(stats) for name, stats in method_stats.items()}
    finalized_deltas = {name: finalize_refinement_delta(delta) for name, delta in refinement_deltas.items()}
    detector_only = finalized_methods["detector_only"]
    detector_plus_classifier = finalized_methods["detector_plus_classifier"]
    detector_plus_classifier_plus_kg = finalized_methods["detector_plus_classifier_plus_kg"]

    summary = {
        "images": total_images,
        "score_threshold": float(args.score_threshold),
        "iou_threshold": float(args.iou_threshold),
        "detector_checkpoint": str(args.detector_checkpoint),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "knowledge_graph_artifact": str(args.knowledge_graph_artifact),
        "kg_settings": {
            "kg_top_k": int(args.kg_top_k),
            "kg_visual_weight": float(args.kg_visual_weight),
            "kg_context_weight": float(args.kg_context_weight),
            "kg_anchor_weight": float(args.kg_anchor_weight),
            "kg_selective_override": bool(args.kg_selective_override),
            "kg_max_detector_score": float(args.kg_max_detector_score),
            "kg_min_candidate_probability": float(args.kg_min_candidate_probability),
            "kg_max_anchor_probability": float(args.kg_max_anchor_probability),
        },
        "methods": finalized_methods,
        "refinement_deltas": finalized_deltas,
        "improvements": {
            "classifier_vs_detector_f1_delta": float(detector_plus_classifier["f1"] - detector_only["f1"]),
            "kg_vs_classifier_f1_delta": float(detector_plus_classifier_plus_kg["f1"] - detector_plus_classifier["f1"]),
            "kg_vs_detector_f1_delta": float(detector_plus_classifier_plus_kg["f1"] - detector_only["f1"]),
            "classifier_vs_detector_label_accuracy_delta": float(detector_plus_classifier["matched_label_accuracy"] - detector_only["matched_label_accuracy"]),
            "kg_vs_classifier_label_accuracy_delta": float(detector_plus_classifier_plus_kg["matched_label_accuracy"] - detector_plus_classifier["matched_label_accuracy"]),
            "kg_vs_detector_label_accuracy_delta": float(detector_plus_classifier_plus_kg["matched_label_accuracy"] - detector_only["matched_label_accuracy"]),
        },
    }

    save_json(args.output_dir / "knowledge_graph_benchmark.json", summary)
    plot_benchmark_comparison(finalized_methods, args.output_dir / "knowledge_graph_benchmark.png")

    print("=" * 80)
    print("KNOWLEDGE GRAPH BENCHMARK")
    print("=" * 80)
    for name, metrics in finalized_methods.items():
        pretty_name = name.replace("_", " ")
        print(f"{pretty_name}:")
        print(f"  Precision: {float(metrics['precision']):.2%}")
        print(f"  Recall: {float(metrics['recall']):.2%}")
        print(f"  F1: {float(metrics['f1']):.2%}")
        print(f"  IoU-matched label accuracy: {float(metrics['matched_label_accuracy']):.2%}")
    print()
    print("Impact summary:")
    print(f"  Classifier vs detector F1 delta: {summary['improvements']['classifier_vs_detector_f1_delta']:+.2%}")
    print(f"  KG vs classifier F1 delta: {summary['improvements']['kg_vs_classifier_f1_delta']:+.2%}")
    print(f"  KG vs detector F1 delta: {summary['improvements']['kg_vs_detector_f1_delta']:+.2%}")
    print(f"  Classifier vs detector label accuracy delta: {summary['improvements']['classifier_vs_detector_label_accuracy_delta']:+.2%}")
    print(f"  KG vs classifier label accuracy delta: {summary['improvements']['kg_vs_classifier_label_accuracy_delta']:+.2%}")
    print(f"  KG vs detector label accuracy delta: {summary['improvements']['kg_vs_detector_label_accuracy_delta']:+.2%}")
    print(f"Saved: {args.output_dir / 'knowledge_graph_benchmark.json'}")
    print(f"Saved: {args.output_dir / 'knowledge_graph_benchmark.png'}")


if __name__ == "__main__":
    main()
