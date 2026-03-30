from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from demo_infer import (
    DEFAULT_DEMO_CLASSIFIER_CHECKPOINT,
    DEFAULT_DEMO_DETECTOR_CHECKPOINT,
    build_app_response,
)
from detection_test import predict_single_image
from detection_utils import DEFAULT_DETECTION_SCORE_THRESHOLD, load_detection_checkpoint
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
    build_label_display_names,
    load_classifier_checkpoint,
    load_or_build_knowledge_graph,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".PNG")
DEFAULT_REAL_DATA_ROOT = Path("data") / "user_real_photos" / "pill"


def save_json(path: Path, payload: Dict[str, object] | List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap draft labels for your real photos using the current detector + classifier.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_REAL_DATA_ROOT, help="Custom real-photo dataset root")
    parser.add_argument("--detector-checkpoint", type=Path, default=DEFAULT_DEMO_DETECTOR_CHECKPOINT, help="Detector checkpoint to load")
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_DEMO_CLASSIFIER_CHECKPOINT, help="Classifier checkpoint used for reranking")
    parser.add_argument("--knowledge-graph-artifact", type=Path, default=DEFAULT_KNOWLEDGE_GRAPH_PATH, help="Path to saved knowledge graph JSON")
    parser.add_argument("--knowledge-graph-cache-dir", type=Path, default=DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR, help="Crop cache directory used to build graph prototypes")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_DETECTION_SCORE_THRESHOLD, help="Detector confidence threshold")
    parser.add_argument("--kg-top-k", type=int, default=DEFAULT_KG_CANDIDATES, help="Classifier top-k candidates before graph reranking")
    parser.add_argument("--kg-visual-weight", type=float, default=DEFAULT_KG_VISUAL_WEIGHT, help="Weight for visual prototype matching")
    parser.add_argument("--kg-context-weight", type=float, default=DEFAULT_KG_CONTEXT_WEIGHT, help="Weight for cross-pill context")
    parser.add_argument("--kg-anchor-weight", type=float, default=DEFAULT_KG_ANCHOR_WEIGHT, help="Weight for detector-label compatibility")
    parser.add_argument("--kg-selective-override", action=argparse.BooleanOptionalAction, default=DEFAULT_KG_SELECTIVE_OVERRIDE, help="Only override uncertain detector labels")
    parser.add_argument("--kg-max-detector-score", type=float, default=DEFAULT_KG_MAX_DETECTOR_SCORE, help="Max detector confidence that still allows override")
    parser.add_argument("--kg-min-candidate-probability", type=float, default=DEFAULT_KG_MIN_CANDIDATE_PROBABILITY, help="Minimum classifier probability required before override")
    parser.add_argument("--kg-max-anchor-probability", type=float, default=DEFAULT_KG_MAX_ANCHOR_PROBABILITY, help="Maximum classifier support for the detector label before override is blocked")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite existing JSON labels")
    parser.add_argument("--max-images", type=int, default=0, help="Optional limit for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.data_root / "image"
    label_dir = args.data_root / "label"
    output_dir = args.data_root / "bootstrap_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(sorted(image_dir.glob(f"*{ext}")))
    image_paths = sorted({path for path in image_paths})
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found under {image_dir}")

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
        rebuild=False,
    )
    label_display_names = build_label_display_names()

    written = 0
    skipped = 0
    summary: List[Dict[str, object]] = []

    print("=" * 80)
    print("BOOTSTRAP REAL PHOTO LABELS")
    print("=" * 80)
    print(f"Image dir: {image_dir}")
    print(f"Label dir: {label_dir}")
    print(f"Detector checkpoint: {args.detector_checkpoint}")
    print(f"Classifier checkpoint: {args.classifier_checkpoint}")

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.json"
        if label_path.exists() and not args.overwrite:
            skipped += 1
            summary.append({"image": str(image_path), "status": "skipped_existing"})
            continue

        payload = predict_single_image(
            model=detector_model,
            checkpoint=detector_checkpoint,
            index_to_label=detector_index_to_label,
            device=device,
            image_path=image_path,
            output_dir=output_dir / image_path.stem,
            score_threshold=args.score_threshold,
            label_json=None,
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
        detections = list(app_response.get("detections", []))

        annotations: List[Dict[str, object]] = []
        for item in detections:
            x1, y1, x2, y2 = [int(round(float(coord))) for coord in item["box_xyxy"]]
            annotations.append(
                {
                    "x": x1,
                    "y": y1,
                    "w": max(1, x2 - x1),
                    "h": max(1, y2 - y1),
                    "label": int(item["label_id"]),
                }
            )

        save_json(label_path, annotations)
        written += 1
        summary.append(
            {
                "image": str(image_path),
                "label_json": str(label_path),
                "num_annotations": len(annotations),
                "status": "written",
            }
        )
        print(f"{image_path.name}: wrote {len(annotations)} draft labels")

    save_json(args.data_root / "bootstrap_summary.json", summary)
    print(f"Done. Written: {written}, skipped existing: {skipped}")
    print("Review the generated JSON labels before training.")


if __name__ == "__main__":
    main()
