from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from detection_test import predict_single_image
from detection_utils import (
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    save_json,
    load_detection_checkpoint,
)
from knowledge_graph import (
    DEFAULT_KG_CANDIDATES,
    DEFAULT_KG_CONTEXT_WEIGHT,
    DEFAULT_KG_ANCHOR_WEIGHT,
    DEFAULT_KG_MAX_ANCHOR_PROBABILITY,
    DEFAULT_KG_MAX_DETECTOR_SCORE,
    DEFAULT_KG_MIN_CANDIDATE_PROBABILITY,
    DEFAULT_KG_SELECTIVE_OVERRIDE,
    DEFAULT_KG_VISUAL_WEIGHT,
    DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR,
    DEFAULT_KNOWLEDGE_GRAPH_PATH,
    build_label_display_names,
    load_classifier_checkpoint,
    load_or_build_knowledge_graph,
)


DEFAULT_DEMO_DETECTOR_CHECKPOINT = Path("checkpoints") / "detection_mnv3_hardmining_ft_v2" / "best_model.pth"
# Project-wide default classifier: keep app/demo/docs aligned on the clean
# held-out test checkpoint stored at checkpoints/best_model.pth.
DEFAULT_DEMO_CLASSIFIER_CHECKPOINT = Path("checkpoints") / "best_model.pth"
DEFAULT_DEMO_OUTPUT_DIR = Path("checkpoints") / "demo_app_output"
DEFAULT_APP_RESPONSE_IOU_THRESHOLD = 0.80


def _decorate_label(label_id: int, label_display_names: Dict[int, str] | None) -> Dict[str, str]:
    label_name = str((label_display_names or {}).get(int(label_id), "")).strip()
    display_label = f"{int(label_id)} | {label_name}" if label_name else str(int(label_id))
    return {
        "label_name": label_name,
        "display_label": display_label,
    }


def _box_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _suppress_overlapping_detections(
    detections: List[Dict[str, object]],
    iou_threshold: float = DEFAULT_APP_RESPONSE_IOU_THRESHOLD,
) -> tuple[List[Dict[str, object]], int]:
    kept: List[Dict[str, object]] = []
    suppressed_count = 0

    for item in sorted(detections, key=lambda detection: float(detection["score"]), reverse=True):
        has_duplicate = any(
            _box_iou(list(item["box_xyxy"]), list(existing["box_xyxy"])) >= iou_threshold
            for existing in kept
        )
        if has_duplicate:
            suppressed_count += 1
            continue
        kept.append(item)

    return kept, suppressed_count


def build_app_response(
    payload: Dict[str, object],
    *,
    label_display_names: Dict[int, str] | None = None,
) -> Dict[str, object]:
    kg_payload = payload.get("knowledge_graph_refinement", {})
    refined = list(kg_payload.get("refined_detections", []))
    if refined:
        detections = [
            {
                "box_xyxy": list(item["box_xyxy"]),
                "label_id": int(item["final_label_id"]),
                "score": float(item["final_probability"]),
                "source": str(item.get("selected_source", "detector")),
                "detector_label_id": int(item["detector_label_id"]),
                "detector_score": float(item["detector_score"]),
                "override_applied": bool(item.get("override_applied", False)),
                "override_checks": dict(item.get("override_checks", {})),
                "anchor_probability": float(item.get("anchor_probability", 0.0)),
                **_decorate_label(int(item["final_label_id"]), label_display_names),
                "detector_label_name": _decorate_label(int(item["detector_label_id"]), label_display_names)["label_name"],
                "detector_display_label": _decorate_label(int(item["detector_label_id"]), label_display_names)["display_label"],
            }
            for item in refined
        ]
    else:
        detections = [
            {
                "box_xyxy": list(item["box_xyxy"]),
                "label_id": int(item["label_id"]),
                "score": float(item["score"]),
                "source": "detector",
                "detector_label_id": int(item["label_id"]),
                "detector_score": float(item["score"]),
                "override_applied": False,
                "override_checks": {},
                "anchor_probability": 0.0,
                **_decorate_label(int(item["label_id"]), label_display_names),
                "detector_label_name": _decorate_label(int(item["label_id"]), label_display_names)["label_name"],
                "detector_display_label": _decorate_label(int(item["label_id"]), label_display_names)["display_label"],
            }
            for item in payload.get("detections", [])
        ]

    raw_num_detections = len(detections)
    detections, suppressed_detections = _suppress_overlapping_detections(detections)
    overrides = [item for item in detections if bool(item["override_applied"])]
    top_labels: List[int] = [int(item["label_id"]) for item in detections]
    top_label_displays: List[str] = [str(item["display_label"]) for item in detections]
    return {
        "image_path": str(payload["image_path"]),
        "num_detections": len(detections),
        "raw_num_detections": raw_num_detections,
        "suppressed_detections": suppressed_detections,
        "num_overrides": len(overrides),
        "top_labels": top_labels,
        "top_label_displays": top_label_displays,
        "knowledge_graph_enabled": bool(kg_payload.get("enabled", False)),
        "knowledge_graph_selective_override": bool(kg_payload.get("kg_selective_override", False)),
        "detections": detections,
        "artifacts": dict(payload.get("artifacts", {})),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="App-ready single-image multi-pill inference with selective knowledge graph enabled by default.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a full pill image")
    parser.add_argument("--label-json", type=Path, default=None, help="Optional ground-truth JSON for comparison preview")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DEMO_OUTPUT_DIR, help="Directory for JSON + preview outputs")
    parser.add_argument("--detector-checkpoint", type=Path, default=DEFAULT_DEMO_DETECTOR_CHECKPOINT, help="Detector checkpoint to load")
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_DEMO_CLASSIFIER_CHECKPOINT, help="Classifier checkpoint used for post-detection reranking")
    parser.add_argument("--knowledge-graph-artifact", type=Path, default=DEFAULT_KNOWLEDGE_GRAPH_PATH, help="Path to saved knowledge graph JSON")
    parser.add_argument("--knowledge-graph-cache-dir", type=Path, default=DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR, help="Crop cache directory used to build graph prototypes")
    parser.add_argument("--build-knowledge-graph", action=argparse.BooleanOptionalAction, default=False, help="Rebuild the graph before inference")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_DETECTION_SCORE_THRESHOLD, help="Detector confidence threshold")
    parser.add_argument("--kg-top-k", type=int, default=DEFAULT_KG_CANDIDATES, help="Classifier top-k candidates before graph reranking")
    parser.add_argument("--kg-visual-weight", type=float, default=DEFAULT_KG_VISUAL_WEIGHT, help="Weight for visual prototype matching")
    parser.add_argument("--kg-context-weight", type=float, default=DEFAULT_KG_CONTEXT_WEIGHT, help="Weight for prescription context")
    parser.add_argument("--kg-anchor-weight", type=float, default=DEFAULT_KG_ANCHOR_WEIGHT, help="Weight for detector-label anchor compatibility")
    parser.add_argument("--kg-selective-override", action=argparse.BooleanOptionalAction, default=DEFAULT_KG_SELECTIVE_OVERRIDE, help="Keep detector labels unless classifier + KG evidence is strong")
    parser.add_argument("--kg-max-detector-score", type=float, default=DEFAULT_KG_MAX_DETECTOR_SCORE, help="Maximum detector confidence that still allows override")
    parser.add_argument("--kg-min-candidate-probability", type=float, default=DEFAULT_KG_MIN_CANDIDATE_PROBABILITY, help="Minimum classifier probability required before override")
    parser.add_argument("--kg-max-anchor-probability", type=float, default=DEFAULT_KG_MAX_ANCHOR_PROBABILITY, help="Maximum classifier support for the detector label before override is blocked")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.detector_checkpoint.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {args.detector_checkpoint}")
    if not args.classifier_checkpoint.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {args.classifier_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector_model, detector_checkpoint, _, detector_index_to_label = load_detection_checkpoint(
        args.detector_checkpoint,
        device,
    )
    classifier_model, classifier_checkpoint, _, classifier_idx_to_class = load_classifier_checkpoint(
        args.classifier_checkpoint,
        device,
    )

    confusion_metrics_path = args.classifier_checkpoint.parent / "test_metrics.json"
    if not confusion_metrics_path.exists():
        confusion_metrics_path = None
    knowledge_graph = load_or_build_knowledge_graph(
        artifact_path=args.knowledge_graph_artifact,
        cache_dir=args.knowledge_graph_cache_dir,
        image_size=int(classifier_checkpoint["image_size"]),
        color_bins=int(classifier_checkpoint.get("color_bins", 8)),
        confusion_metrics_path=confusion_metrics_path,
        rebuild=bool(args.build_knowledge_graph),
    )
    label_display_names = build_label_display_names()

    payload = predict_single_image(
        model=detector_model,
        checkpoint=detector_checkpoint,
        index_to_label=detector_index_to_label,
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

    app_response = build_app_response(payload, label_display_names=label_display_names)
    app_response_path = args.output_dir / "app_response.json"
    app_response.setdefault("artifacts", {})["app_response_json"] = str(app_response_path)
    save_json(app_response_path, app_response)
    print(f"Saved app response: {app_response_path}")


if __name__ == "__main__":
    main()
