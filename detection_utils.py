from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


DEFAULT_DETECTION_DATA_ROOT = Path("archive (1)") / "public_train" / "pill"
DEFAULT_DETECTION_OUTPUT_DIR = Path("checkpoints") / "detection"
DEFAULT_DETECTION_RECORDS_CACHE = Path("checkpoints") / "detection_records_cache.json"
DEFAULT_DETECTION_MODEL = "fasterrcnn_mobilenet_v3_large_fpn"
DEFAULT_DETECTION_EPOCHS = 12
DEFAULT_DETECTION_PATIENCE = 4
DEFAULT_DETECTION_BATCH_SIZE = 2
DEFAULT_DETECTION_SEED = 42
DEFAULT_DETECTION_LR = 1e-4
DEFAULT_DETECTION_WEIGHT_DECAY = 1e-4
DEFAULT_DETECTION_VAL_RATIO = 0.10
DEFAULT_DETECTION_TEST_RATIO = 0.10
DEFAULT_DETECTION_RESIZE_LONG_SIDE = 1024
DEFAULT_DETECTION_MIN_SIZE = 640
DEFAULT_DETECTION_MAX_SIZE = 1024
DEFAULT_DETECTION_SCORE_THRESHOLD = 0.30
DEFAULT_DETECTION_IOU_THRESHOLD = 0.50
DEFAULT_DETECTION_SAMPLER_POWER = 0.50
DEFAULT_DETECTION_HARD_MINING_TOPK = 0.15
DEFAULT_DETECTION_HARD_MINING_BOOST = 1.75
DEFAULT_DETECTION_HARD_MINING_WARMUP = 1
DEFAULT_DETECTION_SPLIT_MANIFEST = "detection_split_manifest.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".PNG")


def normalize_data_roots(
    primary_root: Path | str,
    extra_roots: Sequence[Path | str] | None = None,
) -> List[Path]:
    roots = [Path(primary_root)]
    roots.extend(Path(root) for root in (extra_roots or []))

    normalized: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(root)
    return normalized


def make_dataset_root_key(root: Path) -> str:
    parts = [part for part in root.parts[-3:] if part not in ("\\", "/", "")]
    stem = "_".join(parts) if parts else root.name or "dataset"
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_") or "dataset"
    digest = hashlib.md5(str(root).encode("utf-8")).hexdigest()[:8]
    return f"{stem[:40]}_{digest}"


def set_seed(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=deterministic)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_image_path(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def safe_box(ann: Dict[str, int], width: int, height: int) -> Tuple[int, int, int, int] | None:
    x = int(ann.get("x", 0))
    y = int(ann.get("y", 0))
    w = int(ann.get("w", 0))
    h = int(ann.get("h", 0))
    if w < 2 or h < 2:
        return None

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def choose_anchor_label(labels: Sequence[int]) -> int:
    counter = Counter(int(label) for label in labels)
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_detection_records(
    data_root: Path | Sequence[Path],
    cache_path: Path | None = DEFAULT_DETECTION_RECORDS_CACHE,
) -> List[Dict[str, object]]:
    if isinstance(data_root, Sequence) and not isinstance(data_root, (Path, str)):
        data_roots = normalize_data_roots(data_root[0], data_root[1:])
    else:
        data_roots = normalize_data_roots(data_root)

    if cache_path is not None and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cached_roots = payload.get("data_roots") if isinstance(payload, dict) else None
        if not cached_roots and isinstance(payload, dict) and payload.get("data_root"):
            cached_roots = [payload["data_root"]]
        cached_records = payload.get("records") if isinstance(payload, dict) else None
        if cached_roots == [str(root) for root in data_roots] and isinstance(cached_records, list) and cached_records:
            print(f"Loaded detection records cache from {cache_path}")
            return cached_records

    records: List[Dict[str, object]] = []
    for root in data_roots:
        image_dir = root / "image"
        label_dir = root / "label"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Could not find VAIPE pill data under: {root}")

        for label_path in sorted(label_dir.glob("*.json")):
            image_path = resolve_image_path(image_dir, label_path.stem)
            if image_path is None:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]

            with open(label_path, "r", encoding="utf-8") as handle:
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

            if not boxes:
                continue

            records.append(
                {
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "source_root": str(root),
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                    "labels": labels,
                    "num_boxes": len(boxes),
                    "anchor_label": choose_anchor_label(labels),
                }
            )

    if not records:
        raise RuntimeError("No usable detection records were created from the dataset.")
    if cache_path is not None:
        save_json(
            cache_path,
            {
                "data_roots": [str(root) for root in data_roots],
                "records": records,
            },
        )
    return records


def split_detection_records(
    records: Sequence[Dict[str, object]],
    seed: int,
    val_ratio: float = DEFAULT_DETECTION_VAL_RATIO,
    test_ratio: float = DEFAULT_DETECTION_TEST_RATIO,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[int(record["anchor_label"])].append(dict(record))

    rng = random.Random(seed)
    train_records: List[Dict[str, object]] = []
    val_records: List[Dict[str, object]] = []
    test_records: List[Dict[str, object]] = []

    for anchor_label in sorted(grouped):
        samples = list(grouped[anchor_label])
        rng.shuffle(samples)
        count = len(samples)

        if count == 1:
            train_records.extend(samples)
            continue
        if count == 2:
            train_records.append(samples[0])
            val_records.append(samples[1])
            continue

        test_count = max(1, int(round(count * test_ratio)))
        val_count = max(1, int(round(count * val_ratio)))
        while count - val_count - test_count < 1:
            if test_count > val_count and test_count > 1:
                test_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                break

        train_end = count - val_count - test_count
        val_end = count - test_count

        train_records.extend(samples[:train_end])
        val_records.extend(samples[train_end:val_end])
        test_records.extend(samples[val_end:])

    return train_records, val_records, test_records


def save_detection_split_manifest(
    output_dir: Path,
    train_records: Sequence[Dict[str, object]],
    val_records: Sequence[Dict[str, object]],
    test_records: Sequence[Dict[str, object]],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Path:
    manifest_path = output_dir / DEFAULT_DETECTION_SPLIT_MANIFEST
    payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train_records": list(train_records),
        "val_records": list(val_records),
        "test_records": list(test_records),
    }
    save_json(manifest_path, payload)
    return manifest_path


def maybe_load_detection_split_manifest(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def create_label_mappings(records: Sequence[Dict[str, object]]) -> Tuple[Dict[int, int], Dict[int, int]]:
    label_ids = sorted({int(label) for record in records for label in record["labels"]})
    label_to_index = {label_id: index + 1 for index, label_id in enumerate(label_ids)}
    index_to_label = {index: label_id for label_id, index in label_to_index.items()}
    return label_to_index, index_to_label


def resize_image_and_boxes(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    resize_long_side: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    height, width = image_rgb.shape[:2]
    if max(height, width) <= resize_long_side:
        return image_rgb, boxes.astype(np.float32), 1.0

    scale = resize_long_side / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_boxes = boxes.astype(np.float32) * scale
    return resized_image, resized_boxes, scale


class VAIPEMultiPillDetectionDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        label_to_index: Dict[int, int],
        resize_long_side: int,
        is_train: bool,
    ) -> None:
        self.records = [dict(record) for record in records]
        self.label_to_index = dict(label_to_index)
        self.resize_long_side = resize_long_side
        self.is_train = is_train
        self.color_jitter = ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        record = self.records[index]
        image_bgr = cv2.imread(str(record["image_path"]))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {record['image_path']}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes = np.asarray(record["boxes"], dtype=np.float32)
        labels = np.asarray([self.label_to_index[int(label)] for label in record["labels"]], dtype=np.int64)

        image_rgb, boxes, _ = resize_image_and_boxes(image_rgb, boxes, self.resize_long_side)
        height, width = image_rgb.shape[:2]

        if self.is_train and random.random() < 0.5:
            image_rgb = np.ascontiguousarray(image_rgb[:, ::-1])
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        if self.is_train and random.random() < 0.20:
            image_rgb = np.ascontiguousarray(image_rgb[::-1, :])
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]

        image = Image.fromarray(image_rgb)
        if self.is_train:
            image = self.color_jitter(image)
        image_tensor = TF.to_tensor(image)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.as_tensor((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate_fn(batch: Sequence[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_detection_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
    sample_weights: Sequence[float] | None = None,
    num_samples: int | None = None,
) -> DataLoader:
    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=int(num_samples or len(sample_weights)),
            replacement=True,
            generator=generator,
        )
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": bool(shuffle and sampler is None),
        "sampler": sampler,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": detection_collate_fn,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def compute_detection_sample_weights(
    records: Sequence[Dict[str, object]],
    sampler_power: float,
    hard_example_multipliers: Dict[int, float] | None = None,
) -> List[float] | None:
    use_balanced_sampling = sampler_power > 0.0
    use_hard_mining = bool(hard_example_multipliers)
    if not use_balanced_sampling and not use_hard_mining:
        return None

    weights = np.ones(len(records), dtype=np.float64)
    if use_balanced_sampling:
        label_counts = Counter(int(label) for record in records for label in set(int(label) for label in record["labels"]))
        for record_index, record in enumerate(records):
            unique_labels = sorted(set(int(label) for label in record["labels"]))
            rarity = [(1.0 / max(1, label_counts[label_id])) ** sampler_power for label_id in unique_labels]
            weights[record_index] = float(np.mean(rarity)) if rarity else 1.0

    if use_hard_mining:
        for record_index, multiplier in hard_example_multipliers.items():
            if 0 <= int(record_index) < len(weights):
                weights[int(record_index)] *= float(multiplier)

    mean_weight = float(np.mean(weights))
    if mean_weight <= 0.0:
        return [1.0] * len(records)
    return (weights / mean_weight).tolist()


def select_hard_example_multipliers(
    image_losses: Dict[int, float],
    topk_ratio: float,
    boost_factor: float,
) -> Tuple[Dict[int, float], Dict[str, float]]:
    if topk_ratio <= 0.0 or boost_factor <= 1.0 or not image_losses:
        return {}, {"selected_count": 0.0, "loss_threshold": 0.0, "max_loss": 0.0}

    ranked = sorted(image_losses.items(), key=lambda item: item[1], reverse=True)
    selected_count = max(1, int(round(len(ranked) * topk_ratio)))
    selected = ranked[:selected_count]
    multipliers = {int(image_id): float(boost_factor) for image_id, _ in selected}
    summary = {
        "selected_count": float(selected_count),
        "loss_threshold": float(selected[-1][1]),
        "max_loss": float(selected[0][1]),
    }
    return multipliers, summary


def build_detector(
    model_name: str,
    num_classes: int,
    min_size: int,
    max_size: int,
    pretrained_backbone: bool = True,
) -> nn.Module:
    if model_name == "fasterrcnn_mobilenet_v3_large_fpn":
        builder = fasterrcnn_mobilenet_v3_large_fpn
        detection_weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        backbone_weights = MobileNet_V3_Large_Weights.DEFAULT
    elif model_name == "fasterrcnn_resnet50_fpn_v2":
        builder = fasterrcnn_resnet50_fpn_v2
        detection_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        backbone_weights = ResNet50_Weights.DEFAULT
    else:
        raise ValueError(f"Unsupported detector: {model_name}")

    kwargs = {
        "min_size": min_size,
        "max_size": max_size,
    }
    try:
        if pretrained_backbone:
            model = builder(weights=detection_weights, **kwargs)
        else:
            model = builder(weights=None, weights_backbone=None, **kwargs)
    except Exception as exc:
        if pretrained_backbone:
            print(f"Warning: failed to load pretrained detector weights ({exc}). Trying pretrained backbone only.")
            try:
                model = builder(weights=None, weights_backbone=backbone_weights, **kwargs)
            except Exception as backbone_exc:
                print(
                    "Warning: failed to load pretrained backbone weights "
                    f"({backbone_exc}). Falling back to random init."
                )
                model = builder(weights=None, weights_backbone=None, **kwargs)
        else:
            print(f"Warning: failed to build detector without pretrained weights ({exc}). Falling back to random init.")
            model = builder(weights=None, weights_backbone=None, **kwargs)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def move_targets_to_device(targets: Sequence[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    moved_targets: List[Dict[str, torch.Tensor]] = []
    for target in targets:
        moved_targets.append({key: value.to(device) for key, value in target.items()})
    return moved_targets


def train_detection_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> Dict[str, float]:
    model.train()
    use_amp = scaler is not None and device.type == "cuda"

    total_loss = 0.0
    total_batches = 0
    image_loss_totals: Dict[int, float] = defaultdict(float)
    image_loss_counts: Dict[int, int] = defaultdict(int)

    for images, targets in loader:
        image_ids = [int(target["image_id"].item()) for target in targets]
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        if scaler is None:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        batch_loss = float(losses.item())
        total_loss += batch_loss
        total_batches += 1
        for image_id in image_ids:
            image_loss_totals[image_id] += batch_loss
            image_loss_counts[image_id] += 1

    image_losses = {
        int(image_id): float(image_loss_totals[image_id] / max(1, image_loss_counts[image_id]))
        for image_id in image_loss_totals
    }
    return {
        "loss": total_loss / max(1, total_batches),
        "image_losses": image_losses,
    }


def filter_prediction(
    prediction: Dict[str, torch.Tensor],
    score_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction["boxes"].numel() == 0:
        empty_boxes = prediction["boxes"].reshape(0, 4).cpu()
        empty_labels = prediction["labels"].reshape(0).cpu()
        empty_scores = prediction["scores"].reshape(0).cpu()
        return empty_boxes, empty_labels, empty_scores

    keep = prediction["scores"] >= score_threshold
    return (
        prediction["boxes"][keep].detach().cpu(),
        prediction["labels"][keep].detach().cpu(),
        prediction["scores"][keep].detach().cpu(),
    )


def match_detections(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> Tuple[int, int, int, List[float], List[Tuple[int, int]]]:
    matched_gt: set[int] = set()
    matched_ious: List[float] = []
    matched_pairs: List[Tuple[int, int]] = []
    tp = 0
    fp = 0

    if pred_boxes.numel() == 0:
        return 0, 0, int(gt_boxes.shape[0]), matched_ious, matched_pairs

    order = torch.argsort(pred_scores, descending=True)
    for pred_index in order.tolist():
        same_label_candidates = [
            gt_index
            for gt_index in range(gt_boxes.shape[0])
            if gt_index not in matched_gt and int(gt_labels[gt_index]) == int(pred_labels[pred_index])
        ]
        if not same_label_candidates:
            fp += 1
            continue

        candidate_boxes = gt_boxes[same_label_candidates]
        ious = box_iou(pred_boxes[pred_index].unsqueeze(0), candidate_boxes)[0]
        best_local = int(torch.argmax(ious).item())
        best_iou = float(ious[best_local].item())
        if best_iou < iou_threshold:
            fp += 1
            continue

        gt_index = same_label_candidates[best_local]
        matched_gt.add(gt_index)
        tp += 1
        matched_ious.append(best_iou)
        matched_pairs.append((pred_index, gt_index))

    fn = int(gt_boxes.shape[0]) - len(matched_gt)
    return tp, fp, fn, matched_ious, matched_pairs


def evaluate_detection_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    index_to_label: Dict[int, int],
    score_threshold: float,
    iou_threshold: float,
) -> Dict[str, object]:
    model.eval()
    use_amp = device.type == "cuda"

    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious: List[float] = []
    per_class: Dict[int, Dict[str, int]] = defaultdict(lambda: {"support": 0, "tp": 0, "fp": 0, "fn": 0})
    total_predictions = 0
    total_ground_truths = 0
    images_evaluated = 0

    with torch.no_grad():
        for images, targets in loader:
            images_on_device = [image.to(device, non_blocking=True) for image in images]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                predictions = model(images_on_device)

            for prediction, target in zip(predictions, targets):
                pred_boxes, pred_labels, pred_scores = filter_prediction(prediction, score_threshold=score_threshold)
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                for gt_label in gt_labels.tolist():
                    label_id = index_to_label[int(gt_label)]
                    per_class[label_id]["support"] += 1

                tp, fp, fn, batch_ious, matched_pairs = match_detections(
                    pred_boxes=pred_boxes,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    iou_threshold=iou_threshold,
                )
                matched_ious.extend(batch_ious)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_predictions += int(pred_boxes.shape[0])
                total_ground_truths += int(gt_boxes.shape[0])
                images_evaluated += 1

                matched_gt_indexes = {gt_index for _, gt_index in matched_pairs}
                for pred_index, gt_index in matched_pairs:
                    label_id = index_to_label[int(gt_labels[gt_index])]
                    per_class[label_id]["tp"] += 1

                for pred_index in range(pred_boxes.shape[0]):
                    label_id = index_to_label[int(pred_labels[pred_index])]
                    if any(pair[0] == pred_index for pair in matched_pairs):
                        continue
                    per_class[label_id]["fp"] += 1

                for gt_index in range(gt_boxes.shape[0]):
                    if gt_index in matched_gt_indexes:
                        continue
                    label_id = index_to_label[int(gt_labels[gt_index])]
                    per_class[label_id]["fn"] += 1

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    per_class_metrics: List[Dict[str, object]] = []
    for label_id in sorted(per_class):
        stats = per_class[label_id]
        class_precision = stats["tp"] / max(1, stats["tp"] + stats["fp"])
        class_recall = stats["tp"] / max(1, stats["tp"] + stats["fn"])
        class_f1 = 2.0 * class_precision * class_recall / max(1e-8, class_precision + class_recall)
        per_class_metrics.append(
            {
                "label_id": int(label_id),
                "support": int(stats["support"]),
                "tp": int(stats["tp"]),
                "fp": int(stats["fp"]),
                "fn": int(stats["fn"]),
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1": float(class_f1),
            }
        )

    return {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "images": images_evaluated,
        "predictions": total_predictions,
        "ground_truth_boxes": total_ground_truths,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": mean_iou,
        "per_class_metrics": per_class_metrics,
    }


def plot_detection_training_curves(
    history: Dict[str, List[float]],
    output_path: Path,
    best_epoch: int,
) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("VAIPE Multi-Pill Detection - Training Curves", fontsize=18, fontweight="bold")

    axes[0, 0].plot(epochs, history["train_loss"], marker="o", linewidth=2.2, color="#0F766E")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(epochs, history["val_f1"], marker="s", linewidth=2.2, color="#EA580C", label="Val F1")
    axes[0, 1].plot(epochs, history["val_precision"], marker="^", linewidth=1.8, color="#2563EB", label="Val Precision")
    axes[0, 1].plot(epochs, history["val_recall"], marker="D", linewidth=1.8, color="#7C3AED", label="Val Recall")
    axes[0, 1].axvline(best_epoch, linestyle="--", color="#334155", alpha=0.8, label=f"Best epoch {best_epoch}")
    axes[0, 1].set_title("Validation Detection Metrics")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history["val_mean_iou"], marker="o", linewidth=2.2, color="#16A34A")
    axes[1, 0].axvline(best_epoch, linestyle="--", color="#334155", alpha=0.8)
    axes[1, 0].set_title("Mean Matched IoU")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("IoU")
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))

    axes[1, 1].plot(epochs, history["lr"], marker="o", linewidth=2.2, color="#DC2626")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_detection_annotations(
    image_rgb: np.ndarray,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Sequence[float] | None = None,
    title: str | None = None,
    output_path: Path | None = None,
) -> None:
    image = image_rgb.copy()
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(value) for value in box]
        width = x2 - x1
        height = y2 - y1
        rect = plt.Rectangle((x1, y1), width, height, fill=False, edgecolor="#10B981", linewidth=2.0)
        ax.add_patch(rect)
        text = f"label {labels[index]}"
        if scores is not None:
            text += f" | {float(scores[index]):.2%}"
        ax.text(
            x1,
            max(0.0, y1 - 4.0),
            text,
            color="white",
            fontsize=9,
            bbox={"facecolor": "#0F766E", "alpha": 0.85, "pad": 2},
        )

    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def load_detection_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object], Dict[int, int], Dict[int, int]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Unsupported detection checkpoint format: {checkpoint_path}")

    label_to_index = {int(label_id): int(index) for label_id, index in checkpoint["label_to_index"].items()}
    index_to_label = {int(index): int(label_id) for index, label_id in checkpoint["index_to_label"].items()}

    model = build_detector(
        model_name=str(checkpoint["model_name"]),
        num_classes=int(checkpoint["num_classes"]),
        min_size=int(checkpoint["min_size"]),
        max_size=int(checkpoint["max_size"]),
        pretrained_backbone=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, label_to_index, index_to_label


def prepare_inference_image(
    image_path: Path,
    resize_long_side: int,
) -> Tuple[np.ndarray, torch.Tensor, float]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_image, _, scale = resize_image_and_boxes(
        image_rgb=image_rgb,
        boxes=np.zeros((0, 4), dtype=np.float32),
        resize_long_side=resize_long_side,
    )
    image_tensor = TF.to_tensor(Image.fromarray(resized_image))
    return image_rgb, image_tensor, scale


def scale_boxes_back(boxes: torch.Tensor, scale: float) -> List[List[float]]:
    if boxes.numel() == 0:
        return []
    scaled = boxes.cpu().numpy().astype(np.float32)
    if scale > 0:
        scaled = scaled / scale
    return scaled.tolist()
