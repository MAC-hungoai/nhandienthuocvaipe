from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from detection_utils import (
    DEFAULT_DETECTION_DATA_ROOT,
    DEFAULT_DETECTION_IOU_THRESHOLD,
    DEFAULT_DETECTION_OUTPUT_DIR,
    DEFAULT_DETECTION_RECORDS_CACHE,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_DETECTION_SPLIT_MANIFEST,
    VAIPEMultiPillDetectionDataset,
    build_detection_records,
    create_detection_loader,
    create_label_mappings,
    draw_detection_annotations,
    evaluate_detection_model,
    load_detection_checkpoint,
    maybe_load_detection_split_manifest,
    prepare_inference_image,
    safe_box,
    save_json,
    scale_boxes_back,
    split_detection_records,
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


def resolve_test_records(
    checkpoint: Dict[str, object],
    checkpoint_path: Path,
    data_root: Path,
    records_cache: Path,
) -> List[Dict[str, object]]:
    manifest_name = str(checkpoint.get("split_manifest", DEFAULT_DETECTION_SPLIT_MANIFEST))
    manifest_path = checkpoint_path.parent / manifest_name
    manifest_payload = maybe_load_detection_split_manifest(manifest_path)
    if manifest_payload is not None:
        test_records = manifest_payload.get("test_records", [])
        if test_records:
            print(f"Loaded exact detection test split from {manifest_path}")
            return list(test_records)

    records = build_detection_records(data_root, cache_path=records_cache)
    _, _, test_records = split_detection_records(
        records=records,
        seed=int(checkpoint["seed"]),
        val_ratio=float(checkpoint.get("val_ratio", 0.10)),
        test_ratio=float(checkpoint.get("test_ratio", 0.10)),
    )
    return test_records


def evaluate_saved_detector(
    model: torch.nn.Module,
    checkpoint: Dict[str, object],
    checkpoint_path: Path,
    device: torch.device,
    data_root: Path,
    records_cache: Path,
    output_dir: Path,
    score_threshold: float,
    iou_threshold: float,
    num_workers: int,
    batch_size: int,
) -> None:
    test_records = resolve_test_records(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        data_root=data_root,
        records_cache=records_cache,
    )
    all_records = build_detection_records(data_root, cache_path=records_cache)
    label_to_index, index_to_label = create_label_mappings(all_records)

    dataset = VAIPEMultiPillDetectionDataset(
        records=test_records,
        label_to_index=label_to_index,
        resize_long_side=int(checkpoint["resize_long_side"]),
        is_train=False,
    )
    loader = create_detection_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        generator=None,
    )
    metrics = evaluate_detection_model(
        model=model,
        loader=loader,
        device=device,
        index_to_label=index_to_label,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.json"
    per_class_path = output_dir / "per_class_metrics.json"
    save_json(metrics_path, metrics)
    save_json(per_class_path, {"per_class_metrics": metrics["per_class_metrics"]})

    print("=" * 80)
    print("HELD-OUT MULTI-PILL DETECTION EVALUATION")
    print("=" * 80)
    print(f"Images: {metrics['images']}")
    print(f"Precision: {float(metrics['precision']):.2%}")
    print(f"Recall: {float(metrics['recall']):.2%}")
    print(f"F1: {float(metrics['f1']):.2%}")
    print(f"Mean matched IoU: {float(metrics['mean_matched_iou']):.2%}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {per_class_path}")


def load_ground_truth(
    image_path: Path,
    label_json: Path,
) -> Tuple[List[List[float]], List[int]]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image_bgr.shape[:2]

    with open(label_json, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    boxes: List[List[float]] = []
    labels: List[int] = []
    for ann in annotations:
        if "label" not in ann:
            continue
        box = safe_box(ann, width=width, height=height)
        if box is None:
            continue
        boxes.append([float(coord) for coord in box])
        labels.append(int(ann["label"]))
    return boxes, labels


def crop_detection_region(image_rgb: np.ndarray, box: Sequence[float]) -> np.ndarray:
    image_rgb_np = np.asarray(image_rgb).copy()
    height, width = image_rgb_np.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    crop = image_rgb_np[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_rgb_np
    return crop


def refine_detections_with_classifier_graph(
    *,
    image_rgb: object,
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
) -> List[Dict[str, object]]:
    order = sorted(range(len(boxes)), key=lambda index: float(detector_scores[index]), reverse=True)
    accepted_labels: List[int] = []
    refinements: Dict[int, Dict[str, object]] = {}

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
        accepted_labels.append(int(selection["final_label_id"]))
        refinements[index] = {
            "box_xyxy": list(boxes[index]),
            "detector_label_id": int(detector_labels[index]),
            "detector_score": float(detector_scores[index]),
            "final_label_id": int(selection["final_label_id"]),
            "final_probability": float(selection["final_probability"]),
            "final_score": float(selection["final_score"]),
            "selected_source": str(selection["selected_source"]),
            "override_applied": bool(selection["override_applied"]),
            "override_checks": dict(selection["override_checks"]),
            "anchor_probability": float(selection["anchor_probability"]),
            "anchor_in_classifier_top_k": bool(selection["anchor_in_classifier_top_k"]),
            "classifier_candidates": selection["classifier_candidates"],
            "knowledge_graph_candidates": selection["knowledge_graph_candidates"],
        }

    return [refinements[index] for index in range(len(boxes))]


def predict_single_image(
    model: torch.nn.Module,
    checkpoint: Dict[str, object],
    index_to_label: Dict[int, int],
    device: torch.device,
    image_path: Path,
    output_dir: Path,
    score_threshold: float,
    label_json: Path | None = None,
    classifier_model: torch.nn.Module | None = None,
    classifier_checkpoint: Dict[str, object] | None = None,
    classifier_idx_to_class: Dict[int, int] | None = None,
    knowledge_graph: Dict[str, object] | None = None,
    kg_top_k: int = DEFAULT_KG_CANDIDATES,
    kg_visual_weight: float = DEFAULT_KG_VISUAL_WEIGHT,
    kg_context_weight: float = DEFAULT_KG_CONTEXT_WEIGHT,
    kg_anchor_weight: float = DEFAULT_KG_ANCHOR_WEIGHT,
    kg_selective_override: bool = DEFAULT_KG_SELECTIVE_OVERRIDE,
    kg_max_detector_score: float = DEFAULT_KG_MAX_DETECTOR_SCORE,
    kg_min_candidate_probability: float = DEFAULT_KG_MIN_CANDIDATE_PROBABILITY,
    kg_max_anchor_probability: float = DEFAULT_KG_MAX_ANCHOR_PROBABILITY,
) -> Dict[str, object]:
    original_image_rgb, image_tensor, scale = prepare_inference_image(
        image_path=image_path,
        resize_long_side=int(checkpoint["resize_long_side"]),
    )

    with torch.no_grad():
        predictions = model([image_tensor.to(device)])
    prediction = predictions[0]
    keep = prediction["scores"] >= score_threshold
    pred_boxes = prediction["boxes"][keep]
    pred_labels = prediction["labels"][keep]
    pred_scores = prediction["scores"][keep]

    box_list = scale_boxes_back(pred_boxes, scale=scale)
    label_list = [index_to_label[int(label)] for label in pred_labels.detach().cpu().tolist()]
    score_list = [float(score) for score in pred_scores.detach().cpu().tolist()]

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "single_image_detection.json"
    preview_path = output_dir / "single_image_detection.png"
    kg_preview_path = output_dir / "single_image_detection_knowledge_graph.png"

    refinements: List[Dict[str, object]] = []
    if (
        classifier_model is not None
        and classifier_checkpoint is not None
        and classifier_idx_to_class is not None
        and knowledge_graph is not None
        and box_list
    ):
        refinements = refine_detections_with_classifier_graph(
            image_rgb=original_image_rgb,
            boxes=box_list,
            detector_labels=label_list,
            detector_scores=score_list,
            classifier_model=classifier_model,
            classifier_checkpoint=classifier_checkpoint,
            classifier_idx_to_class=classifier_idx_to_class,
            knowledge_graph=knowledge_graph,
            device=device,
            candidate_top_k=kg_top_k,
            visual_weight=kg_visual_weight,
            context_weight=kg_context_weight,
            anchor_weight=kg_anchor_weight,
            selective_override=kg_selective_override,
            max_detector_score=kg_max_detector_score,
            min_candidate_probability=kg_min_candidate_probability,
            max_anchor_probability=kg_max_anchor_probability,
        )

    payload = {
        "image_path": str(image_path),
        "score_threshold": score_threshold,
        "detections": [
            {
                "label_id": label_id,
                "score": score,
                "box_xyxy": box,
            }
            for label_id, score, box in zip(label_list, score_list, box_list)
        ],
        "knowledge_graph_refinement": {
            "enabled": bool(refinements),
            "classifier_model_variant": classifier_checkpoint.get("model_variant") if classifier_checkpoint is not None else None,
            "kg_top_k": int(kg_top_k),
            "kg_visual_weight": float(kg_visual_weight),
            "kg_context_weight": float(kg_context_weight),
            "kg_anchor_weight": float(kg_anchor_weight),
            "kg_selective_override": bool(kg_selective_override),
            "kg_max_detector_score": float(kg_max_detector_score),
            "kg_min_candidate_probability": float(kg_min_candidate_probability),
            "kg_max_anchor_probability": float(kg_max_anchor_probability),
            "refined_detections": refinements,
        },
        "artifacts": {
            "result_json": str(result_path),
            "detector_preview": str(preview_path),
            "knowledge_graph_preview": str(kg_preview_path) if refinements else None,
            "ground_truth_preview": None,
        },
    }
    save_json(result_path, payload)
    draw_detection_annotations(
        image_rgb=original_image_rgb,
        boxes=box_list,
        labels=label_list,
        scores=score_list,
        title="Predicted multi-pill detections",
        output_path=preview_path,
    )
    if refinements:
        draw_detection_annotations(
            image_rgb=original_image_rgb,
            boxes=[item["box_xyxy"] for item in refinements],
            labels=[int(item["final_label_id"]) for item in refinements],
            scores=[float(item["final_probability"]) for item in refinements],
            title="Detections refined with classifier + knowledge graph",
            output_path=kg_preview_path,
        )

    print("=" * 80)
    print("SINGLE IMAGE MULTI-PILL DETECTION")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Detections: {len(label_list)}")
    print(f"Saved JSON: {result_path}")
    print(f"Saved preview: {preview_path}")
    if refinements:
        print(f"Saved KG preview: {kg_preview_path}")
    for label_id, score, box in zip(label_list[:10], score_list[:10], box_list[:10]):
        print(f"  label {label_id} | score={score:.2%} | box={box}")
    if refinements:
        print("Knowledge graph refinements:")
        for item in refinements[:10]:
            print(
                "  "
                f"detector={item['detector_label_id']} ({item['detector_score']:.2%}) -> "
                f"final={item['final_label_id']} ({item['final_probability']:.2%})"
            )

    if label_json is not None:
        gt_boxes, gt_labels = load_ground_truth(image_path=image_path, label_json=label_json)
        gt_preview_path = output_dir / "single_image_ground_truth.png"
        draw_detection_annotations(
            image_rgb=original_image_rgb,
            boxes=gt_boxes,
            labels=gt_labels,
            scores=None,
            title="Ground-truth multi-pill boxes",
            output_path=gt_preview_path,
        )
        payload["artifacts"]["ground_truth_preview"] = str(gt_preview_path)
        save_json(result_path, payload)
        print(f"Saved ground-truth preview: {gt_preview_path}")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or run inference with the VAIPE multi-pill detector.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DETECTION_OUTPUT_DIR / "best_model.pth", help="Path to detector checkpoint")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DETECTION_DATA_ROOT, help="Path to public_train/pill")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DETECTION_OUTPUT_DIR / "analysis_outputs",
        help="Directory for evaluation or prediction outputs",
    )
    parser.add_argument(
        "--records-cache",
        type=Path,
        default=DEFAULT_DETECTION_RECORDS_CACHE,
        help="Path to cached detection metadata built from image-level labels.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--image", type=Path, default=None, help="Path to a full pill image for inference")
    parser.add_argument("--label-json", type=Path, default=None, help="Optional ground-truth label JSON for visualization")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_DETECTION_SCORE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_DETECTION_IOU_THRESHOLD, help="IoU threshold for evaluation matching")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None, help="Optional single-pill classifier checkpoint for post-detection reranking")
    parser.add_argument("--knowledge-graph-artifact", type=Path, default=DEFAULT_KNOWLEDGE_GRAPH_PATH, help="Path to saved knowledge graph JSON")
    parser.add_argument("--knowledge-graph-cache-dir", type=Path, default=DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR, help="Crop cache directory used to build knowledge graph prototypes")
    parser.add_argument(
        "--build-knowledge-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild knowledge graph artifact before inference.",
    )
    parser.add_argument("--kg-top-k", type=int, default=DEFAULT_KG_CANDIDATES, help="Classifier top-k candidates before graph reranking")
    parser.add_argument("--kg-visual-weight", type=float, default=DEFAULT_KG_VISUAL_WEIGHT, help="Weight for visual prototype matching during graph reranking")
    parser.add_argument("--kg-context-weight", type=float, default=DEFAULT_KG_CONTEXT_WEIGHT, help="Weight for cross-pill prescription context during graph reranking")
    parser.add_argument("--kg-anchor-weight", type=float, default=DEFAULT_KG_ANCHOR_WEIGHT, help="Weight for detector-label anchor compatibility during graph reranking")
    parser.add_argument("--kg-selective-override", action=argparse.BooleanOptionalAction, default=DEFAULT_KG_SELECTIVE_OVERRIDE, help="Only allow KG to override detector labels when the detector is uncertain and classifier evidence is strong.")
    parser.add_argument("--kg-max-detector-score", type=float, default=DEFAULT_KG_MAX_DETECTOR_SCORE, help="Maximum detector confidence that still allows KG override.")
    parser.add_argument("--kg-min-candidate-probability", type=float, default=DEFAULT_KG_MIN_CANDIDATE_PROBABILITY, help="Minimum classifier probability required before KG can override detector.")
    parser.add_argument("--kg-max-anchor-probability", type=float, default=DEFAULT_KG_MAX_ANCHOR_PROBABILITY, help="Maximum classifier probability assigned to the detector label before KG override is blocked.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, _, index_to_label = load_detection_checkpoint(args.checkpoint, device)

    classifier_model = None
    classifier_checkpoint = None
    classifier_idx_to_class = None
    knowledge_graph = None
    if args.image is not None and args.classifier_checkpoint is not None:
        if not args.classifier_checkpoint.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {args.classifier_checkpoint}")
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

    if args.image is None:
        evaluate_saved_detector(
            model=model,
            checkpoint=checkpoint,
            checkpoint_path=args.checkpoint,
            device=device,
            data_root=args.data_root,
            records_cache=args.records_cache,
            output_dir=args.output_dir,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        return

    predict_single_image(
        model=model,
        checkpoint=checkpoint,
        index_to_label=index_to_label,
        device=device,
        image_path=args.image,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        label_json=args.label_json,
        classifier_model=classifier_model,
        classifier_checkpoint=classifier_checkpoint,
        classifier_idx_to_class=classifier_idx_to_class,
        knowledge_graph=knowledge_graph,
        kg_top_k=args.kg_top_k,
        kg_visual_weight=args.kg_visual_weight,
        kg_context_weight=args.kg_context_weight,
        kg_anchor_weight=args.kg_anchor_weight,
        kg_selective_override=args.kg_selective_override,
        kg_max_detector_score=args.kg_max_detector_score,
        kg_min_candidate_probability=args.kg_min_candidate_probability,
        kg_max_anchor_probability=args.kg_max_anchor_probability,
    )


if __name__ == "__main__":
    main()
