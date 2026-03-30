"""
Minimal VAIPE training pipeline for single-pill classification.

Project focus:
- Task: classify a single pill crop into a VAIPE label ID
- Source data: archive (1)/public_train/pill/{image,label}
- Training: 50 epochs, patience=6, early stopping
- Outputs: best_model.pth, final_model.pth, history.json, test_metrics.json,
  training_curves.png, confusion_matrix.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


DEFAULT_DATA_ROOT = Path("archive (1)") / "public_train" / "pill"
DEFAULT_OUTPUT_DIR = Path("checkpoints")
DEFAULT_IMAGE_SIZE = 160
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 6
DEFAULT_SEED = 42
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.10
DEFAULT_SAMPLER_POWER = 0.5
DEFAULT_COLOR_BINS = 8
DEFAULT_MODEL_VARIANT = "cg_imif_color_fusion"
DEFAULT_SPLIT_MANIFEST = "split_manifest.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".PNG")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def crop_with_padding(image: np.ndarray, box: Sequence[int], padding_ratio: float = 0.12) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image
    return crop


def build_records(data_root: Path) -> List[Dict[str, object]]:
    image_dir = data_root / "image"
    label_dir = data_root / "label"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Could not find VAIPE pill data under: {data_root}")

    records: List[Dict[str, object]] = []
    label_files = sorted(label_dir.glob("*.json"))
    for label_path in label_files:
        image_path = resolve_image_path(image_dir, label_path.stem)
        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        with open(label_path, "r", encoding="utf-8") as handle:
            annotations = json.load(handle)

        for ann_idx, ann in enumerate(annotations):
            if "label" not in ann:
                continue
            box = safe_box(ann, width, height)
            if box is None:
                continue
            label_id = int(ann["label"])
            records.append(
                {
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "box": list(box),
                    "label_id": label_id,
                    "annotation_index": ann_idx,
                }
            )

    if not records:
        raise RuntimeError("No usable crop records were created from the dataset.")
    return records


def prepare_crop_cache(
    data_root: Path | Sequence[Path],
    cache_dir: Path,
    image_size: int,
) -> List[Dict[str, object]]:
    if isinstance(data_root, Sequence) and not isinstance(data_root, (Path, str)):
        data_roots = normalize_data_roots(data_root[0], data_root[1:])
    else:
        data_roots = normalize_data_roots(data_root)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_image_dir = cache_dir / "images"
    cache_image_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("records", [])
        cached_roots = payload.get("source_roots", [])
        if not cached_roots and payload.get("source_root"):
            cached_roots = [payload["source_root"]]
        requested_roots = [str(root) for root in data_roots]
        if requested_roots == list(cached_roots) and records:
            print(f"Loaded crop cache from {metadata_path}", flush=True)
            return records

    records: List[Dict[str, object]] = []
    label_sources: List[Tuple[Path, Path]] = []

    for root in data_roots:
        image_dir = root / "image"
        label_dir = root / "label"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Could not find VAIPE pill data under: {root}")
        label_sources.extend((root, label_path) for label_path in sorted(label_dir.glob("*.json")))

    print(f"Building crop cache at {cache_dir} from {len(data_roots)} source(s) ...", flush=True)
    for index, (root, label_path) in enumerate(label_sources, start=1):
        image_dir = root / "image"
        image_path = resolve_image_path(image_dir, label_path.stem)
        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        with open(label_path, "r", encoding="utf-8") as handle:
            annotations = json.load(handle)

        for ann_idx, ann in enumerate(annotations):
            if "label" not in ann:
                continue
            box = safe_box(ann, width, height)
            if box is None:
                continue

            label_id = int(ann["label"])
            crop = crop_with_padding(image, box)
            crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

            source_key = make_dataset_root_key(root)
            crop_name = f"{source_key}_{label_path.stem}_{ann_idx:02d}_label{label_id}.jpg"
            crop_path = cache_image_dir / crop_name
            if not crop_path.exists():
                cv2.imwrite(str(crop_path), crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            records.append(
                {
                    "crop_path": str(crop_path),
                    "label_id": label_id,
                    "source_root": str(root),
                    "source_image_path": str(image_path),
                    "label_path": str(label_path),
                    "annotation_index": ann_idx,
                }
            )

        if index % 500 == 0:
            print(f"  cached {index}/{len(label_sources)} label files -> {len(records):,} crops", flush=True)

    payload = {
        "image_size": image_size,
        "source_roots": [str(root) for root in data_roots],
        "records": records,
    }
    save_json(metadata_path, payload)
    print(f"Crop cache ready: {len(records):,} crops", flush=True)
    return records


def save_split_manifest(
    output_dir: Path,
    train_records: Sequence[Dict[str, object]],
    val_records: Sequence[Dict[str, object]],
    test_records: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> Path:
    manifest_path = output_dir / DEFAULT_SPLIT_MANIFEST
    payload = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "image_size": args.image_size,
        "deterministic": bool(args.deterministic),
        "data_roots": [str(args.data_root), *[str(root) for root in args.extra_data_root]],
        "train_records": list(train_records),
        "val_records": list(val_records),
        "test_records": list(test_records),
    }
    save_json(manifest_path, payload)
    return manifest_path


def maybe_load_split_manifest(manifest_path: Path) -> Dict[str, object] | None:
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    return payload


def split_records(
    records: Sequence[Dict[str, object]],
    seed: int,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[int(record["label_id"])].append(record)

    rng = random.Random(seed)
    train_records: List[Dict[str, object]] = []
    val_records: List[Dict[str, object]] = []
    test_records: List[Dict[str, object]] = []

    for label_id in sorted(grouped):
        samples = list(grouped[label_id])
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


def extract_color_histogram(image: np.ndarray, bins: int = 8) -> np.ndarray:
    """
    Trích xuất histogram màu từ ảnh để dùng làm color stream.
    
    Đây là một phần của kỹ thuật CG-IMIF (Color & Grey Image-wise Multi-channel Fusion).
    Thay vì dùng CNN để học color features, ta tính trực tiếp histogram HSV.
    
    Args:
        image: ảnh RGB dạng numpy array [H, W, 3]
        bins: số bins cho mỗi channel (H, S, V)
        
    Returns:
        histogram vector shape [bins*3,] normalized (sum=1)
        
    Steps:
        1. Convert RGB → HSV (tách thông tin màu ra)
        2. Tính histogram cho từng channel H, S, V
        3. Nối chúng lại thành 1 vector
        4. Normalize sao cho sum = 1
    """
    # Chuyển ảnh từ RGB sang HSV
    # H (Hue): 0-180, S (Saturation): 0-255, V (Value): 0-255
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Tính histogram cho mỗi channel
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])  # Hue: 0-180
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])  # Saturation: 0-255
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])  # Value: 0-255
    
    # Nối 3 histograms thành 1 vector [bins*3]
    hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    
    # Normalize: chia cho tổng để được distribution
    # Thêm 1e-8 để tránh division by zero
    hist = hist / (hist.sum() + 1e-8)
    
    return hist.astype(np.float32)


def build_transforms(image_size: int, is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.25),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.4, 2.4), value="random"),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class VAIPECropDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        class_to_idx: Dict[int, int],
        image_size: int,
        is_train: bool,
        color_bins: int = DEFAULT_COLOR_BINS,
    ) -> None:
        self.records = list(records)
        self.class_to_idx = class_to_idx
        self.transform = build_transforms(image_size=image_size, is_train=is_train)
        self.color_bins = color_bins

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        record = self.records[index]
        crop_path = record.get("crop_path")
        if crop_path is not None:
            image_path = Path(str(crop_path))
            image = cv2.imread(str(image_path))
        else:
            image_path = Path(str(record["image_path"]))
            image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if crop_path is not None:
            crop = image
        else:
            crop = crop_with_padding(image, record["box"])
        color_hist = torch.from_numpy(extract_color_histogram(crop, bins=self.color_bins))
        crop_image = Image.fromarray(crop)
        tensor = self.transform(crop_image)
        target = self.class_to_idx[int(record["label_id"])]
        return tensor, color_hist, target


class CGIMIFColorFusionClassifier(nn.Module):
    """
    Mô hình phân loại viên thuốc dùng Color Fusion.
    
    Kiến trúc:
    ┌─────────────────────┐
    │  Input image (RGB)  │  [B, 3, 160, 160]
    └────────┬────────────┘
             │
      ┌──────┴─────────────────────┬──────────────────┐
      │                            │                  │
      ▼                            ▼                  ▼
    ResNet18             Color Histogram        Color Stream
    image_backbone       Extraction             Processing
    [B,512]              [B, 24]              [B, 64]
      │                                         │
      └──────────────────┬──────────────────────┘
                         │
                         ▼
                    Concatenate
                    [B, 576]
                         │
                         ▼
                 Classification Head
                 (2 FC layers + dropout)
                         │
                         ▼
                    Output logits
                    [B, num_classes]
    
    CG-IMIF = Color & Grey Image-wise Multi-channel Image Fusion
    Ý tưởi: Kết hợp CNN features (RGB) + handcrafted color features (HSV)
    """

    def __init__(
        self,
        num_classes: int,      # Số lớp thuốc (ví dụ 200+)
        color_feature_dim: int,  # Chiều của color histogram (bins*3, thường 24)
        pretrained: bool = True,  # Dùng ImageNet pretrained weights?
    ) -> None:
        super().__init__()
        
        # ========== IMAGE STREAM (ResNet18) ==========
        # Load ResNet18 pretrained trên ImageNet
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.image_backbone = models.resnet18(weights=weights)
        
        # Lấy số features từ FC layer cuối
        in_features = self.image_backbone.fc.in_features  # 512
        
        # Xóa FC layer (chỉ dùng features từ global average pool)
        self.image_backbone.fc = nn.Identity()
        
        # Flag để forward() biết dùng color stream
        self.uses_color_stream = True

        # ========== COLOR STREAM (Handcrafted) ==========
        # Process handcrafted HSV histogram features
        self.color_head = nn.Sequential(
            nn.LayerNorm(color_feature_dim),        # Normalize input
            nn.Linear(color_feature_dim, 64),       # 24 → 64
            nn.ReLU(inplace=True),                  # Activation
            nn.Dropout(p=0.15),                     # Regularization
            nn.Linear(64, 64),                      # 64 → 64 (dùng lại sau ReLU)
            nn.ReLU(inplace=True),
        )
        # Output: [B, 64]

        # ========== CLASSIFICATION HEAD ==========
        # Fusion: nối image features + color features
        # [512] + [64] = [576]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.35),                     # Aggressive dropout
            nn.Linear(in_features + 64, 256),      # 576 → 256 (bottleneck)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes),            # 256 → num_classes
        )

    def forward(self, images: torch.Tensor, color_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: [B, 3, 160, 160] - normalized RGB images
            color_features: [B, 24] - HSV histograms
            
        Returns:
            logits: [B, num_classes]
        """
        # Trích xuất image features qua ResNet18
        image_features = self.image_backbone(images)  # [B, 512]
        
        # Process color histogram features
        color_embed = self.color_head(color_features)  # [B, 24] → [B, 64]
        
        # Fusion: concatenate cả 2 feature streams
        fused_features = torch.cat([image_features, color_embed], dim=1)  # [B, 576]
        
        # Phân loại dựa trên fused features
        return self.classifier(fused_features)  # [B, num_classes]


def infer_model_variant_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    if any(key.startswith("image_backbone.") for key in state_dict):
        return "cg_imif_color_fusion"
    return "resnet18"


def build_model(
    num_classes: int,
    model_variant: str = DEFAULT_MODEL_VARIANT,
    color_feature_dim: int = DEFAULT_COLOR_BINS * 3,
    pretrained: bool = True,
) -> nn.Module:
    if model_variant == "cg_imif_color_fusion":
        return CGIMIFColorFusionClassifier(
            num_classes=num_classes,
            color_feature_dim=color_feature_dim,
            pretrained=pretrained,
        )

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.20),
        nn.Linear(256, num_classes),
    )
    model.uses_color_stream = False
    return model


def forward_model(
    model: nn.Module,
    images: torch.Tensor,
    color_features: torch.Tensor,
) -> torch.Tensor:
    if bool(getattr(model, "uses_color_stream", False)):
        return model(images, color_features)
    return model(images)


def create_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
    return DataLoader(dataset, **kwargs)


def compute_sampler(
    records: Sequence[Dict[str, object]],
    power: float = DEFAULT_SAMPLER_POWER,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    label_counts = Counter(int(record["label_id"]) for record in records)
    weights = [1.0 / float(label_counts[int(record["label_id"])]) ** power for record in records]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def compute_class_weights(
    records: Sequence[Dict[str, object]],
    class_to_idx: Dict[int, int],
    power: float = 1.0,
) -> torch.Tensor:
    label_counts = Counter(int(record["label_id"]) for record in records)
    counts = torch.tensor([label_counts[label_id] for label_id, _ in sorted(class_to_idx.items(), key=lambda item: item[1])], dtype=torch.float32)
    weights = counts.max() / counts.clamp(min=1.0)
    weights = weights.pow(power)
    weights = weights / weights.mean()
    return weights


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]), persistent=True)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha if self.alpha.numel() > 0 else None,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt).pow(self.gamma)
        return (focal_term * ce_loss).mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, float]:
    """
    Chạy 1 epoch training hoặc validation.
    
    Args:
        model: Neural network model
        loader: DataLoader (chứa batches of data)
        criterion: Loss function (ví dụ CrossEntropyLoss)
        device: torch.device('cuda' or 'cpu')
        optimizer: Optimizer (None nếu validation)
        scaler: GradScaler for AMP (Automatic Mixed Precision)
        
    Returns:
        Dict với keys: loss, acc, top3_acc (metrics)
        
    Logic:
        - Nếu optimizer != None → Training mode (update weights)
        - Nếu optimizer == None → Validation/test mode (no update)
    """
    # Determine mode từ sự có/vắng mặt của optimizer
    is_train = optimizer is not None
    
    # Bật training mode nếu training, ngược lại bật eval mode
    # (điều này ảnh hưởng dropout, batch norm, etc)
    model.train(is_train)

    # ========== Accumulators ==========
    total_loss = 0.0      # Tích lũy loss
    total_correct = 0     # Số lần dự đoán đúng
    total_top3 = 0        # Số lần đúng trong top-3 predictions
    total_samples = 0     # Tổng số samples

    # AMP enabled nếu có scaler và device là CUDA
    autocast_enabled = scaler is not None and device.type == "cuda"

    # ========== Main Training/Validation Loop ==========
    for images, color_features, targets in loader:
        # Chuyển dữ liệu sang device (GPU/CPU) + non_blocking=True để async transfer
        images = images.to(device, non_blocking=True)
        color_features = color_features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            # Xóa gradient từ backward() lần trước
            optimizer.zero_grad(set_to_none=True)

        # Forward pass với autocast (mixed precision) nếu enabled
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = forward_model(model, images, color_features)  # [B, num_classes]
            loss = criterion(logits, targets)

        # ========== Backward Pass (chỉ nếu training) ==========
        if is_train:
            if scaler is None:
                # Backward without AMP
                loss.backward()
                # Clip gradients để tránh exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                # Backward with AMP
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        # ========== Compute Metrics ==========
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size  # Accumulate
        
        # Top-1 accuracy
        preds = logits.argmax(dim=1)  # [B] - predicted class indices
        total_correct += (preds == targets).sum().item()
        
        # Top-3 accuracy
        # Lấy top 3 predicted classes
        topk = logits.topk(k=min(3, logits.size(1)), dim=1).indices  # [B, 3]
        # Check nếu target nằm trong top 3
        total_top3 += topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
        
        total_samples += batch_size

    # ========== Return averaged metrics ==========
    return {
        "loss": total_loss / max(1, total_samples),      # Trung bình loss
        "acc": total_correct / max(1, total_samples),    # Top-1 accuracy
        "top3_acc": total_top3 / max(1, total_samples),  # Top-3 accuracy
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    idx_to_class: Dict[int, int],
) -> Dict[str, object]:
    model.eval()
    autocast_enabled = device.type == "cuda"

    total_loss = 0.0
    total_correct = 0
    total_top3 = 0
    total_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    per_class_total: Dict[int, int] = Counter()
    per_class_correct: Dict[int, int] = Counter()

    with torch.no_grad():
        for images, color_features, targets in loader:
            images = images.to(device, non_blocking=True)
            color_features = color_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = forward_model(model, images, color_features)
                loss = criterion(logits, targets)

            preds = logits.argmax(dim=1)
            topk = logits.topk(k=min(3, logits.size(1)), dim=1).indices

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == targets).sum().item()
            total_top3 += topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += batch_size

            targets_cpu = targets.cpu().tolist()
            preds_cpu = preds.cpu().tolist()
            for target_idx, pred_idx in zip(targets_cpu, preds_cpu):
                label_id = idx_to_class[target_idx]
                per_class_total[label_id] += 1
                if target_idx == pred_idx:
                    per_class_correct[label_id] += 1

            y_true.extend(targets_cpu)
            y_pred.extend(preds_cpu)

    per_class_accuracy = {
        str(label_id): per_class_correct[label_id] / total
        for label_id, total in sorted(per_class_total.items())
        if total > 0
    }

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
        "top3_accuracy": total_top3 / max(1, total_samples),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples": total_samples,
        "per_class_accuracy": per_class_accuracy,
    }


def evaluate_model_detailed(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    idx_to_class: Dict[int, int],
) -> Dict[str, object]:
    model.eval()
    autocast_enabled = device.type == "cuda"

    total_loss = 0.0
    total_correct = 0
    total_top3 = 0
    total_samples = 0
    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []
    top3_hits: List[bool] = []

    with torch.no_grad():
        for images, color_features, targets in loader:
            images = images.to(device, non_blocking=True)
            color_features = color_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = forward_model(model, images, color_features)
                loss = criterion(logits, targets)

            preds = logits.argmax(dim=1)
            topk = logits.topk(k=min(3, logits.size(1)), dim=1).indices
            batch_top3_hits = topk.eq(targets.unsqueeze(1)).any(dim=1)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == targets).sum().item()
            total_top3 += batch_top3_hits.sum().item()
            total_samples += batch_size

            y_true_idx.extend(targets.cpu().tolist())
            y_pred_idx.extend(preds.cpu().tolist())
            top3_hits.extend(batch_top3_hits.cpu().tolist())

    labels_idx = list(range(len(idx_to_class)))
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=labels_idx)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_idx,
        y_pred_idx,
        labels=labels_idx,
        zero_division=0,
    )

    per_class_metrics: List[Dict[str, object]] = []
    for idx in labels_idx:
        label_id = idx_to_class[idx]
        support_count = int(support[idx])
        correct_count = int(cm[idx, idx])
        row = cm[idx].copy()
        row[idx] = 0
        top_confusions = []
        for pred_idx in np.argsort(row)[::-1][:5]:
            if row[pred_idx] <= 0:
                continue
            top_confusions.append(
                {
                    "predicted_label_id": idx_to_class[int(pred_idx)],
                    "count": int(row[pred_idx]),
                }
            )

        per_class_metrics.append(
            {
                "label_id": label_id,
                "support": support_count,
                "correct": correct_count,
                "accuracy": float(correct_count / support_count) if support_count > 0 else 0.0,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "top_confusions": top_confusions,
            }
        )

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
        "top3_accuracy": total_top3 / max(1, total_samples),
        "macro_f1": float(f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)),
        "samples": total_samples,
        "per_class_accuracy": {
            str(item["label_id"]): item["accuracy"]
            for item in per_class_metrics
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "y_true_idx": y_true_idx,
        "y_pred_idx": y_pred_idx,
        "y_true_label_ids": [idx_to_class[idx] for idx in y_true_idx],
        "y_pred_label_ids": [idx_to_class[idx] for idx in y_pred_idx],
    }


def checkpoint_payload(
    model: nn.Module,
    class_to_idx: Dict[int, int],
    best_epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    extra_metrics: Dict[str, float] | None = None,
) -> Dict[str, object]:
    idx_to_class = {index: label_id for label_id, index in class_to_idx.items()}
    payload = {
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "image_size": args.image_size,
        "model_name": "resnet18",
        "model_variant": args.model_variant,
        "color_bins": args.color_bins,
        "color_feature_dim": args.color_bins * 3,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "deterministic": bool(args.deterministic),
        "split_manifest": DEFAULT_SPLIT_MANIFEST,
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "data_roots": [str(args.data_root), *[str(root) for root in args.extra_data_root]],
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
    }
    if extra_metrics:
        payload["metrics"] = extra_metrics
    return payload


def save_json(path: Path, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def plot_training_curves(
    history: Dict[str, List[float]],
    test_metrics: Dict[str, object],
    output_path: Path,
    best_epoch: int,
) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    gap = np.array(history["train_acc"]) - np.array(history["val_acc"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("VAIPE Pill Classification - Training Curves", fontsize=18, fontweight="bold")

    axes[0, 0].plot(epochs, history["train_loss"], marker="o", linewidth=2.2, color="#0F766E", label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], marker="s", linewidth=2.2, color="#EA580C", label="Validation")
    axes[0, 0].axvline(best_epoch, linestyle="--", color="#334155", alpha=0.8, label=f"Best epoch {best_epoch}")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-entropy loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["train_acc"], marker="o", linewidth=2.2, color="#059669", label="Train")
    axes[0, 1].plot(epochs, history["val_acc"], marker="s", linewidth=2.2, color="#F59E0B", label="Validation")
    axes[0, 1].axvline(best_epoch, linestyle="--", color="#334155", alpha=0.8)
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Top-1 accuracy")
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, gap, marker="D", linewidth=2.2, color="#DC2626")
    axes[1, 0].axhline(0.0, linestyle="--", color="#334155", alpha=0.8)
    axes[1, 0].axvline(best_epoch, linestyle="--", color="#334155", alpha=0.8)
    axes[1, 0].set_title("Generalization Gap")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Train acc - Val acc")
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))

    axes[1, 1].plot(epochs, history["lr"], marker="o", linewidth=2.2, color="#2563EB")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")

    summary = (
        f"Best epoch: {best_epoch}\n"
        f"Best val acc: {max(history['val_acc']):.2%}\n"
        f"Test acc: {float(test_metrics['accuracy']):.2%}\n"
        f"Test top-3: {float(test_metrics['top3_accuracy']):.2%}\n"
        f"Test macro-F1: {float(test_metrics['macro_f1']):.2%}"
    )
    fig.text(
        0.73,
        0.17,
        summary,
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(metrics: Dict[str, object], output_path: Path, normalize: bool = True) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float32)
    label_ids = [int(item["label_id"]) for item in metrics["per_class_metrics"]]

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, np.maximum(row_sums, 1.0), out=np.zeros_like(matrix), where=row_sums > 0)

    plt.figure(figsize=(18, 16))
    plt.imshow(matrix, cmap="magma", aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Confusion Matrix (row-normalized)" if normalize else "Confusion Matrix")
    plt.xlabel("Predicted label ID")
    plt.ylabel("True label ID")

    step = max(1, len(label_ids) // 18)
    ticks = np.arange(0, len(label_ids), step)
    tick_labels = [str(label_ids[index]) for index in ticks]
    plt.xticks(ticks, tick_labels, rotation=90, fontsize=7)
    plt.yticks(ticks, tick_labels, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-pill classifier on VAIPE crops.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to public_train/pill")
    parser.add_argument(
        "--extra-data-root",
        type=Path,
        action="append",
        default=[],
        help="Additional pill dataset roots with matching image/label folders. Repeat this flag to add more sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for model outputs")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional existing crop cache directory to reuse instead of rebuilding under output-dir.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional classifier checkpoint to load before fine-tuning.",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Input image size")
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=["resnet18", "cg_imif_color_fusion"],
        default=DEFAULT_MODEL_VARIANT,
        help="Backbone variant. cg_imif_color_fusion adds a CG-IMIF-inspired color stream.",
    )
    parser.add_argument("--color-bins", type=int, default=DEFAULT_COLOR_BINS, help="HSV histogram bins per channel")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["cross_entropy", "weighted_ce", "focal"],
        default="cross_entropy",
        help="Loss function for class imbalance handling",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help="Power applied to inverse-frequency class weights",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma for focal loss",
    )
    parser.add_argument(
        "--sampler-power",
        type=float,
        default=DEFAULT_SAMPLER_POWER,
        help="Sampling strength. 0.5=sqrt inverse frequency, 1.0=full inverse frequency",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic training/evaluation where possible. Use --no-deterministic to opt out.",
    )
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Test split ratio")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0 if os.name == "nt" else 4,
        help="DataLoader workers",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main training pipeline cho single-pill classification.
    
    Quy trình chính:
    1. Parse arguments & setup device
    2. Load & cache crop data
    3. Split train/val/test
    4. Create DataLoaders
    5. Build model & optimizer
    6. Training loop với early stopping
    7. Evaluate on test set
    8. Save results (history, metrics, visualizations)
    """
    # ========== SETUP ==========
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed để reproducibility
    set_seed(args.seed, deterministic=args.deterministic)
    
    # Float32 matmul precision tuning (có thể tăng tốc độ)
    torch.set_float32_matmul_precision("high")

    # Detect device (GPU nếu available, ngược lại CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # Dùng Automatic Mixed Precision trên GPU
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # Cho AMP training

    # ========== PRINT INFO ==========
    print("=" * 80)
    print("VAIPE SINGLE-PILL CLASSIFICATION")
    print("=" * 80)
    data_roots = normalize_data_roots(args.data_root, args.extra_data_root)
    print(f"Device: {device}", flush=True)
    print(f"Data root: {args.data_root}", flush=True)
    if len(data_roots) > 1:
        print("Extra data roots:", flush=True)
        for root in data_roots[1:]:
            print(f"  - {root}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Patience: {args.patience}", flush=True)
    print(f"Model variant: {args.model_variant}", flush=True)
    print(f"Deterministic: {args.deterministic}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(flush=True)

    # ========== DATA PREPARATION ==========
    # Nếu chưa có cache crop, tạo mới; nếu có rồi thì load cache
    cache_dir = args.cache_dir if args.cache_dir is not None else args.output_dir / f"crop_cache_{args.image_size}"
    print(f"Cache dir: {cache_dir}", flush=True)
    print(flush=True)
    
    # Prepare crop cache (resize tất cả crops về 160×160 để training nhanh)
    records = prepare_crop_cache(data_roots, cache_dir=cache_dir, image_size=args.image_size)
    
    # Split train/val/test theo class (để balanced)
    train_records, val_records, test_records = split_records(
        records=records,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    
    # Lưu split manifest để reproduce exact split sau này
    split_manifest_path = save_split_manifest(
        output_dir=args.output_dir,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        args=args,
    )

    # ========== BUILD CLASS MAPPING ==========
    # Lấy tất cả unique class IDs (0, 1, 2, ..., num_classes-1)
    class_ids = sorted({int(record["label_id"]) for record in records})
    # Map class_id → index (0 to num_classes-1)
    class_to_idx = {label_id: idx for idx, label_id in enumerate(class_ids)}
    # Reverse mapping
    idx_to_class = {idx: label_id for label_id, idx in class_to_idx.items()}

    # ========== DATASET SUMMARY ==========
    dataset_summary = {
        "total_crops": len(records),
        "train_crops": len(train_records),
        "val_crops": len(val_records),
        "test_crops": len(test_records),
        "num_classes": len(class_ids),
        "deterministic": bool(args.deterministic),
        "split_manifest": str(split_manifest_path),
        "data_roots": [str(root) for root in data_roots],
        "class_distribution": {str(label): count for label, count in Counter(int(r["label_id"]) for r in records).items()},
    }
    save_json(args.output_dir / "dataset_summary.json", dataset_summary)

    print(f"Total pill crops: {len(records):,}", flush=True)
    print(f"Classes: {len(class_ids)}", flush=True)
    print(f"Split → train: {len(train_records):,}, val: {len(val_records):,}, test: {len(test_records):,}", flush=True)
    print(f"Split manifest: {split_manifest_path}", flush=True)
    print(flush=True)

    # ========== CREATE DATASETS & LOADERS ==========
    # Tạo generators cho reproducibility
    train_generator = make_generator(args.seed + 101)
    val_generator = make_generator(args.seed + 202)
    test_generator = make_generator(args.seed + 303)

    # Instantiate datasets
    train_dataset = VAIPECropDataset(
        train_records,
        class_to_idx,
        args.image_size,
        is_train=True,  # Data augmentation enabled
        color_bins=args.color_bins,
    )
    val_dataset = VAIPECropDataset(
        val_records,
        class_to_idx,
        args.image_size,
        is_train=False,  # No augmentation
        color_bins=args.color_bins,
    )
    test_dataset = VAIPECropDataset(
        test_records,
        class_to_idx,
        args.image_size,
        is_train=False,
        color_bins=args.color_bins,
    )

    # Create dataloaders
    # Training: balanced sampler + shuffle
    train_loader = create_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=compute_sampler(train_records, power=args.sampler_power, generator=train_generator),
        generator=train_generator,
    )
    # Validation & Test: sequential (shuffle=False)
    val_loader = create_loader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=val_generator,
    )
    test_loader = create_loader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=test_generator,
    )

    # ========== BUILD MODEL & OPTIMIZER ==========
    model = build_model(
        num_classes=len(class_to_idx),
        model_variant=args.model_variant,
        color_feature_dim=args.color_bins * 3,
        pretrained=True,
    ).to(device)
    if args.init_checkpoint is not None:
        init_checkpoint = torch.load(args.init_checkpoint, map_location=device)
        init_payload = init_checkpoint if isinstance(init_checkpoint, dict) else {}
        state_dict = init_payload.get("state_dict", init_checkpoint)
        checkpoint_variant = str(init_payload.get("model_variant", infer_model_variant_from_state_dict(state_dict)))
        checkpoint_class_to_idx = init_payload.get("class_to_idx")
        if checkpoint_class_to_idx is not None:
            checkpoint_class_to_idx = {int(label_id): int(index) for label_id, index in checkpoint_class_to_idx.items()}
            if checkpoint_class_to_idx != class_to_idx:
                raise ValueError("Checkpoint class mapping does not match the current dataset labels.")
        if checkpoint_variant != args.model_variant:
            print(
                f"Warning: checkpoint variant '{checkpoint_variant}' differs from requested '{args.model_variant}'.",
                flush=True,
            )
        model.load_state_dict(state_dict, strict=True)
        print(f"Initialized model from checkpoint: {args.init_checkpoint}", flush=True)
    
    # Optimizer: AdamW with weight decay for L2 regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler: Reduce LR nếu validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    
    # ========== LOSS FUNCTION ==========
    # Compute class weights cho weighted loss (biased towards minority classes)
    class_weights = compute_class_weights(train_records, class_to_idx, power=args.class_weight_power).to(device)
    
    # Select loss based on argument
    if args.loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    elif args.loss_type == "weighted_ce":
        # Weight loss by inverse class frequency
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    else:
        # Focal loss: focus on hard examples
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma, label_smoothing=0.02)

    # ========== TRAINING LOOP ==========
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_top3_acc": [],
        "val_top3_acc": [],
        "lr": [],
        "epoch_time_sec": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    min_delta = 1e-4  # Minimum improvement to reset patience
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        
        # Run training & validation
        train_metrics = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer,
            scaler=scaler if use_amp else None
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start

        # ========== HISTORY TRACKING ==========
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_top3_acc"].append(train_metrics["top3_acc"])
        history["val_top3_acc"].append(val_metrics["top3_acc"])
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)

        gap = train_metrics["acc"] - val_metrics["acc"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['acc']:.2%} | val_acc={val_metrics['acc']:.2%} | "
            f"gap={gap:+.2%} | lr={current_lr:.2e} | time={epoch_time:.1f}s",
            end="",
            flush=True,
        )

        # ========== EARLY STOPPING LOGIC ==========
        # Nếu val_loss giảm nhiều hơn min_delta
        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            
            # Lưu checkpoint tốt nhất
            torch.save(
                checkpoint_payload(
                    model=model,
                    class_to_idx=class_to_idx,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    args=args,
                    extra_metrics={"val_acc": val_metrics["acc"], "val_top3_acc": val_metrics["top3_acc"]},
                ),
                args.output_dir / "best_model.pth",
            )
            print(" | saved best_model.pth", flush=True)
        else:
            # No improvement, increment patience counter
            patience_counter += 1
            print(f" | patience={patience_counter}/{args.patience}", flush=True)
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.", flush=True)
                break

    total_train_time = time.perf_counter() - train_start

    # ========== SAVE FINAL MODEL ==========
    torch.save(
        checkpoint_payload(
            model=model,
            class_to_idx=class_to_idx,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            args=args,
        ),
        args.output_dir / "final_model.pth",
    )

    # ========== FINAL EVALUATION ON TEST SET ==========
    # Load best checkpoint
    best_checkpoint = torch.load(args.output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["state_dict"])
    
    # Evaluate with detailed metrics (confusion matrix, per-class metrics)
    test_metrics = evaluate_model_detailed(model, test_loader, criterion, device, idx_to_class)

    # ========== SAVE RESULTS ==========
    history_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": max(history["val_acc"]) if history["val_acc"] else 0.0,
        "final_train_accuracy": history["train_acc"][-1] if history["train_acc"] else 0.0,
        "final_val_accuracy": history["val_acc"][-1] if history["val_acc"] else 0.0,
        "epochs_ran": len(history["train_loss"]),
        "total_train_time_sec": total_train_time,
        "avg_epoch_time_sec": float(np.mean(history["epoch_time_sec"])) if history["epoch_time_sec"] else 0.0,
    }

    save_json(args.output_dir / "history.json", {**history, "summary": history_summary})
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    plot_training_curves(history, test_metrics, args.output_dir / "training_curves.png", best_epoch)
    plot_confusion_matrix(test_metrics, args.output_dir / "confusion_matrix.png", normalize=True)

    print(flush=True)
    print("=" * 80, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best val loss: {best_val_loss:.4f}", flush=True)
    print(f"Best val acc: {history_summary['best_val_accuracy']:.2%}", flush=True)
    print(f"Test acc: {float(test_metrics['accuracy']):.2%}", flush=True)
    print(f"Test top-3 acc: {float(test_metrics['top3_accuracy']):.2%}", flush=True)
    print(f"Test macro-F1: {float(test_metrics['macro_f1']):.2%}", flush=True)
    print(f"Training time: {total_train_time / 60:.1f} minutes", flush=True)
    print(flush=True)
    print("Saved files:", flush=True)
    print(f"  - {args.output_dir / 'best_model.pth'}", flush=True)
    print(f"  - {args.output_dir / 'final_model.pth'}", flush=True)
    print(f"  - {args.output_dir / 'history.json'}", flush=True)
    print(f"  - {args.output_dir / 'test_metrics.json'}", flush=True)
    print(f"  - {args.output_dir / 'training_curves.png'}", flush=True)
    print(f"  - {args.output_dir / 'confusion_matrix.png'}", flush=True)
    print(f"  - {split_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
