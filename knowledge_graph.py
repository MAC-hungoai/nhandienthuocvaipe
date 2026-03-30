from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from train import (
    DEFAULT_COLOR_BINS,
    DEFAULT_DATA_ROOT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OUTPUT_DIR,
    build_model,
    build_transforms,
    extract_color_histogram,
    forward_model,
    infer_model_variant_from_state_dict,
    prepare_crop_cache,
)


DEFAULT_KNOWLEDGE_GRAPH_PATH = DEFAULT_OUTPUT_DIR / "knowledge_graph_vaipe.json"
DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR = DEFAULT_OUTPUT_DIR / f"crop_cache_{DEFAULT_IMAGE_SIZE}"
DEFAULT_KNOWLEDGE_MAX_SAMPLES_PER_LABEL = 96
DEFAULT_KNOWLEDGE_TOP_NEIGHBORS = 10
DEFAULT_KG_CANDIDATES = 5
DEFAULT_KG_VISUAL_WEIGHT = 0.35
DEFAULT_KG_CONTEXT_WEIGHT = 0.20
DEFAULT_KG_ANCHOR_WEIGHT = 0.15
DEFAULT_KG_SELECTIVE_OVERRIDE = True
DEFAULT_KG_MAX_DETECTOR_SCORE = 0.90
DEFAULT_KG_MIN_CANDIDATE_PROBABILITY = 0.90
DEFAULT_KG_MAX_ANCHOR_PROBABILITY = 0.02


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.upper()
    normalized = re.sub(r"[^A-Z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def resolve_prescription_label_dir(data_root: Path) -> Path:
    candidate = data_root / "prescription" / "label"
    if candidate.exists():
        return candidate
    parent_candidate = data_root.parent / "prescription" / "label"
    if parent_candidate.exists():
        return parent_candidate
    return candidate


def tokenize_text(text: str, min_len: int = 3) -> List[str]:
    stopwords = {
        "VIEN",
        "UONG",
        "SAU",
        "KHI",
        "AN",
        "GHI",
        "CHU",
        "SANG",
        "TOI",
        "SL",
        "MG",
        "ML",
        "OTHER",
    }
    tokens = []
    for token in normalize_text(text).split():
        if len(token) < min_len:
            continue
        if token in stopwords:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return tokens


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    a_vec = np.asarray(a, dtype=np.float32)
    b_vec = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(a_vec, b_vec) / denom, -1.0, 1.0))


def distance_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    a_vec = np.asarray(a, dtype=np.float32)
    b_vec = np.asarray(b, dtype=np.float32)
    if a_vec.size == 0 or b_vec.size == 0:
        return 0.0
    distance = float(np.linalg.norm(a_vec - b_vec))
    return float(math.exp(-distance))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_index = int(np.argmax(areas)) + 1
    component = (labels == best_index).astype(np.uint8)
    return component


def extract_pill_mask(image_rgb: np.ndarray) -> np.ndarray:
    image_uint8 = np.asarray(image_rgb, dtype=np.uint8)
    height, width = image_uint8.shape[:2]
    border = max(3, int(round(min(height, width) * 0.10)))

    lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
    border_pixels = np.concatenate(
        [
            lab[:border, :, :].reshape(-1, 3),
            lab[-border:, :, :].reshape(-1, 3),
            lab[:, :border, :].reshape(-1, 3),
            lab[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border_pixels, axis=0, keepdims=True)
    distance = np.linalg.norm(lab.reshape(-1, 3).astype(np.float32) - background.astype(np.float32), axis=1)
    distance_map = distance.reshape(height, width)
    threshold = max(10.0, float(np.percentile(distance, 70)))
    mask = (distance_map >= threshold).astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = _largest_component(mask)

    area_ratio = float(mask.mean())
    if area_ratio < 0.08:
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        _, otsu_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(otsu_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = _largest_component(mask)

    if float(mask.mean()) < 0.05:
        mask = np.ones((height, width), dtype=np.uint8)
    return mask


def extract_shape_features(image_rgb: np.ndarray) -> np.ndarray:
    mask = extract_pill_mask(image_rgb)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(6, dtype=np.float32)

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))

    image_area = float(image_rgb.shape[0] * image_rgb.shape[1])
    aspect_ratio = float(w / max(1, h))
    fill_ratio = area / max(1.0, image_area)
    bbox_fill = area / max(1.0, float(w * h))
    solidity = area / max(1.0, hull_area)
    circularity = 0.0 if perimeter <= 1e-6 else float((4.0 * math.pi * area) / (perimeter * perimeter))
    equivalent_radius = math.sqrt(area / math.pi) if area > 0.0 else 0.0
    radius_ratio = equivalent_radius / max(1.0, 0.5 * max(w, h))
    return np.asarray(
        [aspect_ratio, fill_ratio, bbox_fill, solidity, circularity, radius_ratio],
        dtype=np.float32,
    )


def extract_imprint_signature(image_rgb: np.ndarray) -> np.ndarray:
    mask = extract_pill_mask(image_rgb)
    gray = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, threshold1=40, threshold2=120)
    edges = cv2.bitwise_and(edges, edges, mask=(mask * 255).astype(np.uint8))

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    orientation = (cv2.phase(grad_x, grad_y, angleInDegrees=False) % math.pi).astype(np.float32)

    edge_pixels = edges > 0
    if not np.any(edge_pixels):
        return np.zeros(9, dtype=np.float32)

    edge_density = float(edge_pixels.mean())
    magnitudes = magnitude[edge_pixels]
    orientations = orientation[edge_pixels]
    histogram, _ = np.histogram(
        orientations,
        bins=8,
        range=(0.0, math.pi),
        weights=magnitudes,
    )
    histogram = histogram.astype(np.float32)
    histogram = histogram / (float(histogram.sum()) + 1e-8)
    return np.concatenate([[edge_density], histogram], axis=0).astype(np.float32)


def load_classifier_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object], Dict[int, int], Dict[int, int]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "class_to_idx" not in checkpoint:
        raise ValueError(f"Unsupported classifier checkpoint format: {checkpoint_path}")

    class_to_idx = {int(label_id): int(class_index) for label_id, class_index in checkpoint["class_to_idx"].items()}
    idx_payload = checkpoint.get("idx_to_class")
    if idx_payload is None:
        idx_to_class = {class_index: label_id for label_id, class_index in class_to_idx.items()}
    else:
        idx_to_class = {int(class_index): int(label_id) for class_index, label_id in idx_payload.items()}

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


def classify_crop_candidates(
    model: nn.Module,
    checkpoint: Dict[str, object],
    idx_to_class: Dict[int, int],
    device: torch.device,
    crop_rgb: np.ndarray,
    top_k: int = DEFAULT_KG_CANDIDATES,
) -> List[Dict[str, float]]:
    pil_image = Image.fromarray(np.asarray(crop_rgb, dtype=np.uint8))
    image_size = int(checkpoint["image_size"])
    color_bins = int(checkpoint.get("color_bins", DEFAULT_COLOR_BINS))
    tensor = build_transforms(image_size=image_size, is_train=False)(pil_image)
    color_features = torch.from_numpy(extract_color_histogram(np.asarray(pil_image), bins=color_bins))

    with torch.no_grad():
        logits = forward_model(
            model,
            tensor.unsqueeze(0).to(device),
            color_features.unsqueeze(0).to(device),
        )
        probabilities = torch.softmax(logits, dim=1)[0]
        top_count = min(int(top_k), probabilities.shape[0])
        top_probs, top_indices = probabilities.topk(top_count)

    return [
        {
            "rank": int(rank),
            "label_id": int(idx_to_class[int(class_index)]),
            "probability": float(probability),
        }
        for rank, (probability, class_index) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1)
    ]


def _iter_prescription_rows(data_root: Path) -> Iterable[Tuple[str, List[Dict[str, object]]]]:
    label_dir = resolve_prescription_label_dir(data_root)
    for label_path in sorted(label_dir.glob("*.json")):
        with open(label_path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        yield label_path.name, rows


def collect_prescription_knowledge(
    data_root: Path,
) -> Dict[str, object]:
    label_to_names: Dict[int, Counter[str]] = defaultdict(Counter)
    label_to_name_tokens: Dict[int, Counter[str]] = defaultdict(Counter)
    label_to_diagnosis_terms: Dict[int, Counter[str]] = defaultdict(Counter)
    co_occurrence: Dict[int, Counter[int]] = defaultdict(Counter)
    prescription_count: Counter[int] = Counter()

    for _, rows in _iter_prescription_rows(data_root):
        prescription_labels: List[int] = []
        diagnosis_terms: Counter[str] = Counter()
        for row in rows:
            row_label = str(row.get("label", "")).lower()
            if row_label == "diagnose":
                diagnosis_terms.update(tokenize_text(str(row.get("text", ""))))

        for row in rows:
            if str(row.get("label", "")).lower() != "drugname":
                continue
            if "mapping" not in row:
                continue
            label_id = int(row["mapping"])
            prescription_labels.append(label_id)
            raw_name = normalize_text(str(row.get("text", "")))
            if raw_name:
                label_to_names[label_id][raw_name] += 1
                label_to_name_tokens[label_id].update(tokenize_text(raw_name))

        unique_labels = sorted(set(prescription_labels))
        for label_id in unique_labels:
            prescription_count[label_id] += 1
            label_to_diagnosis_terms[label_id].update(diagnosis_terms)

        for source_label, target_label in combinations(unique_labels, 2):
            co_occurrence[source_label][target_label] += 1
            co_occurrence[target_label][source_label] += 1

    return {
        "label_to_names": label_to_names,
        "label_to_name_tokens": label_to_name_tokens,
        "label_to_diagnosis_terms": label_to_diagnosis_terms,
        "co_occurrence": co_occurrence,
        "prescription_count": prescription_count,
    }


def build_label_display_names(
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Dict[int, str]:
    label_to_names: Dict[int, Counter[str]] = defaultdict(Counter)
    for _, rows in _iter_prescription_rows(data_root):
        for row in rows:
            if str(row.get("label", "")).lower() != "drugname":
                continue
            if "mapping" not in row:
                continue
            label_id = int(row["mapping"])
            raw_name = str(row.get("text", "")).strip()
            raw_name = re.sub(r"\s+", " ", raw_name)
            if raw_name:
                label_to_names[label_id][raw_name] += 1
    return {
        int(label_id): str(counter.most_common(1)[0][0])
        for label_id, counter in label_to_names.items()
        if counter
    }


def collect_confusion_knowledge(metrics_path: Path | None) -> Dict[int, Dict[int, float]]:
    confusion: Dict[int, Dict[int, float]] = defaultdict(dict)
    if metrics_path is None or not metrics_path.exists():
        return confusion

    with open(metrics_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for item in payload.get("per_class_metrics", []):
        label_id = int(item["label_id"])
        support = max(1, int(item.get("support", 0)))
        for confused in item.get("top_confusions", []):
            predicted_label = int(confused["predicted_label_id"])
            score = float(confused.get("count", 0)) / float(support)
            current = confusion[label_id].get(predicted_label, 0.0)
            confusion[label_id][predicted_label] = max(current, score)
    return confusion


def _sample_records(records: Sequence[Dict[str, object]], max_samples: int, seed: int) -> List[Dict[str, object]]:
    if len(records) <= max_samples:
        return list(records)
    rng = random.Random(seed)
    sampled = list(records)
    rng.shuffle(sampled)
    return sampled[:max_samples]


def build_knowledge_graph(
    data_root: Path = DEFAULT_DATA_ROOT,
    cache_dir: Path = DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR,
    artifact_path: Path = DEFAULT_KNOWLEDGE_GRAPH_PATH,
    image_size: int = DEFAULT_IMAGE_SIZE,
    color_bins: int = DEFAULT_COLOR_BINS,
    max_samples_per_label: int = DEFAULT_KNOWLEDGE_MAX_SAMPLES_PER_LABEL,
    top_neighbors: int = DEFAULT_KNOWLEDGE_TOP_NEIGHBORS,
    confusion_metrics_path: Path | None = None,
) -> Dict[str, object]:
    records = prepare_crop_cache(data_root, cache_dir=cache_dir, image_size=image_size)
    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[int(record["label_id"])].append(record)

    node_features: Dict[int, Dict[str, object]] = {}
    for label_id in sorted(grouped):
        sampled_records = _sample_records(grouped[label_id], max_samples=max_samples_per_label, seed=label_id + 17)
        color_vectors: List[np.ndarray] = []
        shape_vectors: List[np.ndarray] = []
        imprint_vectors: List[np.ndarray] = []

        for record in sampled_records:
            image = cv2.imread(str(record["crop_path"]))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_vectors.append(extract_color_histogram(image_rgb, bins=color_bins))
            shape_vectors.append(extract_shape_features(image_rgb))
            imprint_vectors.append(extract_imprint_signature(image_rgb))

        if not color_vectors:
            continue

        node_features[label_id] = {
            "sample_count": len(color_vectors),
            "color_prototype": np.mean(np.stack(color_vectors), axis=0).astype(np.float32).tolist(),
            "shape_prototype": np.mean(np.stack(shape_vectors), axis=0).astype(np.float32).tolist(),
            "imprint_signature": np.mean(np.stack(imprint_vectors), axis=0).astype(np.float32).tolist(),
        }

    prescription = collect_prescription_knowledge(data_root)
    confusion = collect_confusion_knowledge(confusion_metrics_path)

    nodes: Dict[str, Dict[str, object]] = {}
    label_ids = sorted(node_features)
    for label_id in label_ids:
        name_counter = prescription["label_to_names"].get(label_id, Counter())
        token_counter = prescription["label_to_name_tokens"].get(label_id, Counter())
        diagnosis_counter = prescription["label_to_diagnosis_terms"].get(label_id, Counter())
        nodes[str(label_id)] = {
            "label_id": label_id,
            "sample_count": int(node_features[label_id]["sample_count"]),
            "prescription_count": int(prescription["prescription_count"].get(label_id, 0)),
            "drug_names": [name for name, _ in name_counter.most_common(5)],
            "name_tokens": [token for token, _ in token_counter.most_common(8)],
            "diagnosis_terms": [term for term, _ in diagnosis_counter.most_common(8)],
            "color_prototype": node_features[label_id]["color_prototype"],
            "shape_prototype": node_features[label_id]["shape_prototype"],
            "imprint_signature": node_features[label_id]["imprint_signature"],
            "neighbors": [],
        }

    for source_label in label_ids:
        source_node = nodes[str(source_label)]
        neighbor_rows: List[Dict[str, object]] = []
        for target_label in label_ids:
            if target_label == source_label:
                continue
            target_node = nodes[str(target_label)]
            color_score = (cosine_similarity(source_node["color_prototype"], target_node["color_prototype"]) + 1.0) * 0.5
            shape_score = distance_similarity(source_node["shape_prototype"], target_node["shape_prototype"])
            imprint_score = (cosine_similarity(source_node["imprint_signature"], target_node["imprint_signature"]) + 1.0) * 0.5
            visual_score = 0.45 * color_score + 0.25 * shape_score + 0.30 * imprint_score

            co_count = int(prescription["co_occurrence"].get(source_label, {}).get(target_label, 0))
            max_support = max(
                1,
                min(
                    int(prescription["prescription_count"].get(source_label, 0)),
                    int(prescription["prescription_count"].get(target_label, 0)),
                ),
            )
            co_score = float(co_count / max_support)

            source_terms = set(source_node["diagnosis_terms"])
            target_terms = set(target_node["diagnosis_terms"])
            diagnosis_score = 0.0
            if source_terms or target_terms:
                diagnosis_score = len(source_terms & target_terms) / max(1, len(source_terms | target_terms))

            source_name_tokens = set(source_node["name_tokens"])
            target_name_tokens = set(target_node["name_tokens"])
            name_score = 0.0
            if source_name_tokens or target_name_tokens:
                name_score = len(source_name_tokens & target_name_tokens) / max(1, len(source_name_tokens | target_name_tokens))

            confusion_score = 0.5 * (
                float(confusion.get(source_label, {}).get(target_label, 0.0))
                + float(confusion.get(target_label, {}).get(source_label, 0.0))
            )

            total_score = (
                0.50 * visual_score
                + 0.20 * co_score
                + 0.10 * diagnosis_score
                + 0.10 * name_score
                + 0.10 * confusion_score
            )

            neighbor_rows.append(
                {
                    "label_id": int(target_label),
                    "total_score": float(total_score),
                    "visual_score": float(visual_score),
                    "color_score": float(color_score),
                    "shape_score": float(shape_score),
                    "imprint_score": float(imprint_score),
                    "co_occurrence_score": float(co_score),
                    "diagnosis_score": float(diagnosis_score),
                    "name_score": float(name_score),
                    "confusion_score": float(confusion_score),
                }
            )

        neighbor_rows.sort(key=lambda item: item["total_score"], reverse=True)
        source_node["neighbors"] = neighbor_rows[:top_neighbors]

    artifact = {
        "data_root": str(data_root),
        "cache_dir": str(cache_dir),
        "image_size": int(image_size),
        "color_bins": int(color_bins),
        "num_labels": len(nodes),
        "feature_dims": {"color": color_bins * 3, "shape": 6, "imprint": 9},
        "confusion_metrics_path": str(confusion_metrics_path) if confusion_metrics_path is not None else None,
        "nodes": nodes,
    }
    save_json(artifact_path, artifact)
    return artifact


def load_knowledge_graph(artifact_path: Path) -> Dict[str, object]:
    with open(artifact_path, "r", encoding="utf-8") as handle:
        graph = json.load(handle)
    for node in graph.get("nodes", {}).values():
        neighbors = node.get("neighbors", [])
        node["neighbor_lookup"] = {str(item["label_id"]): item for item in neighbors}
    return graph


def load_or_build_knowledge_graph(
    artifact_path: Path = DEFAULT_KNOWLEDGE_GRAPH_PATH,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    cache_dir: Path = DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR,
    image_size: int = DEFAULT_IMAGE_SIZE,
    color_bins: int = DEFAULT_COLOR_BINS,
    confusion_metrics_path: Path | None = None,
    rebuild: bool = False,
) -> Dict[str, object]:
    if rebuild or not artifact_path.exists():
        build_knowledge_graph(
            data_root=data_root,
            cache_dir=cache_dir,
            artifact_path=artifact_path,
            image_size=image_size,
            color_bins=color_bins,
            confusion_metrics_path=confusion_metrics_path,
        )
    return load_knowledge_graph(artifact_path)


def rerank_candidates_with_graph(
    crop_rgb: np.ndarray,
    candidates: Sequence[Dict[str, float]],
    graph: Dict[str, object],
    *,
    context_labels: Sequence[int] | None = None,
    anchor_label: int | None = None,
    visual_weight: float = DEFAULT_KG_VISUAL_WEIGHT,
    context_weight: float = DEFAULT_KG_CONTEXT_WEIGHT,
    anchor_weight: float = DEFAULT_KG_ANCHOR_WEIGHT,
) -> List[Dict[str, object]]:
    context_labels = [int(label) for label in context_labels or []]
    color_bins = int(graph.get("color_bins", DEFAULT_COLOR_BINS))
    query_color = extract_color_histogram(crop_rgb, bins=color_bins)
    query_shape = extract_shape_features(crop_rgb)
    query_imprint = extract_imprint_signature(crop_rgb)

    reranked: List[Dict[str, object]] = []
    for item in candidates:
        label_id = int(item["label_id"])
        node = graph.get("nodes", {}).get(str(label_id))
        probability = float(item["probability"])

        visual_score = 0.0
        color_score = 0.0
        shape_score = 0.0
        imprint_score = 0.0
        context_score = 0.0
        anchor_score = 0.0
        drug_names: List[str] = []

        if node is not None:
            color_score = (cosine_similarity(query_color, node["color_prototype"]) + 1.0) * 0.5
            shape_score = distance_similarity(query_shape, node["shape_prototype"])
            imprint_score = (cosine_similarity(query_imprint, node["imprint_signature"]) + 1.0) * 0.5
            visual_score = 0.45 * color_score + 0.25 * shape_score + 0.30 * imprint_score

            if context_labels:
                context_edges = []
                for context_label in context_labels:
                    edge = node.get("neighbor_lookup", {}).get(str(int(context_label)))
                    if edge is None:
                        continue
                    context_edges.append(0.75 * float(edge["co_occurrence_score"]) + 0.25 * float(edge["diagnosis_score"]))
                if context_edges:
                    context_score = float(np.mean(context_edges))

            if anchor_label is not None:
                if int(anchor_label) == label_id:
                    anchor_score = 1.0
                else:
                    edge = node.get("neighbor_lookup", {}).get(str(int(anchor_label)))
                    if edge is not None:
                        anchor_score = float(edge["total_score"])

            drug_names = list(node.get("drug_names", []))

        final_score = math.log(probability + 1e-8) + visual_weight * visual_score + context_weight * context_score + anchor_weight * anchor_score
        reranked.append(
            {
                "label_id": label_id,
                "probability": probability,
                "final_score": float(final_score),
                "visual_score": float(visual_score),
                "color_score": float(color_score),
                "shape_score": float(shape_score),
                "imprint_score": float(imprint_score),
                "context_score": float(context_score),
                "anchor_score": float(anchor_score),
                "drug_names": drug_names,
            }
        )

    reranked.sort(key=lambda item: item["final_score"], reverse=True)
    for index, item in enumerate(reranked, start=1):
        item["rank"] = int(index)
    return reranked


def _find_candidate_by_label(
    candidates: Sequence[Dict[str, object]],
    label_id: int | None,
) -> Dict[str, object] | None:
    if label_id is None:
        return None
    for item in candidates:
        if int(item["label_id"]) == int(label_id):
            return dict(item)
    return None


def select_candidate_with_graph(
    *,
    crop_rgb: np.ndarray,
    candidates: Sequence[Dict[str, float]],
    graph: Dict[str, object],
    detector_label: int,
    detector_score: float,
    context_labels: Sequence[int] | None = None,
    visual_weight: float = DEFAULT_KG_VISUAL_WEIGHT,
    context_weight: float = DEFAULT_KG_CONTEXT_WEIGHT,
    anchor_weight: float = DEFAULT_KG_ANCHOR_WEIGHT,
    selective_override: bool = DEFAULT_KG_SELECTIVE_OVERRIDE,
    max_detector_score: float = DEFAULT_KG_MAX_DETECTOR_SCORE,
    min_candidate_probability: float = DEFAULT_KG_MIN_CANDIDATE_PROBABILITY,
    max_anchor_probability: float = DEFAULT_KG_MAX_ANCHOR_PROBABILITY,
) -> Dict[str, object]:
    detector_label = int(detector_label)
    detector_score = float(detector_score)
    reranked = rerank_candidates_with_graph(
        crop_rgb=crop_rgb,
        candidates=candidates,
        graph=graph,
        context_labels=context_labels,
        anchor_label=detector_label,
        visual_weight=visual_weight,
        context_weight=context_weight,
        anchor_weight=anchor_weight,
    )

    top_candidate = dict(reranked[0]) if reranked else None
    anchor_classifier_candidate = _find_candidate_by_label(candidates, detector_label)
    anchor_graph_candidate = _find_candidate_by_label(reranked, detector_label)
    anchor_probability = float(anchor_classifier_candidate["probability"]) if anchor_classifier_candidate is not None else 0.0
    anchor_in_classifier_top_k = anchor_classifier_candidate is not None
    anchor_in_graph_candidates = anchor_graph_candidate is not None

    override_checks = {
        "label_changed": bool(top_candidate is not None and int(top_candidate["label_id"]) != detector_label),
        "detector_score_ok": detector_score <= float(max_detector_score),
        "candidate_probability_ok": bool(
            top_candidate is not None and float(top_candidate["probability"]) >= float(min_candidate_probability)
        ),
        "anchor_probability_ok": anchor_probability <= float(max_anchor_probability),
    }
    should_override = bool(
        top_candidate is not None
        and override_checks["label_changed"]
        and (
            (not selective_override)
            or all(
                override_checks[key]
                for key in ("detector_score_ok", "candidate_probability_ok", "anchor_probability_ok")
            )
        )
    )

    if should_override and top_candidate is not None:
        selected_label = int(top_candidate["label_id"])
        selected_probability = float(top_candidate["probability"])
        selected_final_score = float(top_candidate["final_score"])
        selected_source = "knowledge_graph"
    else:
        selected_label = detector_label
        selected_probability = detector_score
        selected_final_score = float(anchor_graph_candidate["final_score"]) if anchor_graph_candidate is not None else detector_score
        selected_source = "detector"

    return {
        "final_label_id": selected_label,
        "final_probability": float(selected_probability),
        "final_score": float(selected_final_score),
        "selected_source": selected_source,
        "override_applied": bool(should_override),
        "override_checks": override_checks,
        "selective_override": bool(selective_override),
        "detector_label_id": detector_label,
        "detector_score": detector_score,
        "anchor_probability": float(anchor_probability),
        "anchor_in_classifier_top_k": bool(anchor_in_classifier_top_k),
        "anchor_in_graph_candidates": bool(anchor_in_graph_candidates),
        "classifier_candidates": [dict(item) for item in candidates],
        "knowledge_graph_candidates": reranked,
    }
