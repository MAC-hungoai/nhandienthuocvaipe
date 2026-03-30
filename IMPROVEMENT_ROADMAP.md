# 🔬 HƯỚNG CẢI THIỆN MÔ HÌNH - VAIPE PILL CLASSIFICATION

**Ngày**: March 29, 2026

---

## 📊 Trạng Thái Hiện Tại

### Hiệu Năng
- **Test Accuracy**: ~92-94%
- **Top-3 Accuracy**: ~97-98%
- **Inference Time**: 10-15ms per image (GPU)
- **Model Size**: 45MB

### Kiến Trúc
- **Classifier**: ResNet18 + Color Fusion (CG-IMIF)
- **Detector**: Faster R-CNN (MobileNetV3)
- **Post-processing**: Knowledge Graph Re-ranking

---

## 🎯 7 Hướng Cải Thiện Chính

### 1️⃣ UPGRADE BACKBONE - Mô Hình Mạnh Hơn

**Hiện Tại**: ResNet18 (18 layers, 11M params)

#### ✅ **Tăng Capacity**

**Option A: ResNet50** (50 layers, 25M params)
```python
# train.py - thay đổi 2 dòng

# Hiện tại:
model = models.resnet18(weights=weights)

# Cải thiện:
model = models.resnet50(weights=weights)
in_features = 2048  # ResNet50 output
```

**Kỳ Vọng**:
- ✅ +2-3% accuracy
- ❌ +2x inference time (20-30ms)
- ❌ +3x model size (135MB)

**Khó độ**: ⭐ Dễ (chỉ thay model name)

---

**Option B: EfficientNet** (cân bằng speed vs accuracy)
```python
from torchvision import models
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
in_features = 1408
```

**Kỳ Vọng**:
- ✅ +1-2% accuracy
- ✅ Tương tự hoặc nhanh hơn ResNet18
- ✅ Nhẹ hơn (32MB)

**Khó độ**: ⭐ Dễ

---

**Option C: Vision Transformer (ViT)** 🔥
```python
from torchvision import models
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
in_features = 768
```

**Kỳ Vọng**:
- ✅✅ +3-5% accuracy (bộ dữ liệu đủ lớn)
- ❌❌ +5x inference time (50-75ms)
- ✅ Xử lý tốt global context

**Khó độ**: ⭐ Dễ (API tương tự)

---

#### 📊 **So Sánh Backbone**

| Model | Accuracy | Speed | Size | Global Context |
|-------|----------|-------|------|-----------------|
| ResNet18 | 92-94% | 10-15ms | 45MB | ❌ Yếu (local) |
| ResNet50 | 94-96% | 20-30ms | 135MB | ❌ Yếu (local) |
| EfficientNet-B2 | 93-95% | 12-18ms | 32MB | ✅ Tốt |
| ViT-B16 | 95-97% | 50-75ms | 330MB | ✅✅ Rất Tốt |
| **Recommended** | **EfficientNet** | for **production** | | |

---

### 2️⃣ ADVANCED COLOR STREAM - Tăng Color Features

**Hiện Tại**: HSV histogram (24 features)

#### ✅ **Option A: Expanded Color Features**

```python
def extract_advanced_color_features(image: np.ndarray) -> np.ndarray:
    """
    Thêm nhiều color features ngoài HSV histogram.
    """
    # 1. HSV Histogram (keeping existing)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
    
    # 2. Color Moments (mean, std of each channel)
    h_mean, h_std = hsv[:,:,0].mean(), hsv[:,:,0].std()
    s_mean, s_std = hsv[:,:,1].mean(), hsv[:,:,1].std()
    v_mean, v_std = hsv[:,:,2].mean(), hsv[:,:,2].std()
    
    # 3. Dominant Color (most common hue)
    h_peak = np.argmax(hist_h) * (180 / 8)
    
    # 4. Color Variance (diversity of colors)
    color_entropy = -np.sum(hist_s * np.log(hist_s + 1e-8))
    
    # Combine all (total: 24 + 6 + 2 = 32 features)
    features = np.concatenate([
        hist_h.flatten(),           # 8
        hist_s.flatten(),           # 8
        hist_v.flatten(),           # 8
        [h_mean, h_std],            # 2
        [s_mean, s_std],            # 2
        [v_mean, v_std],            # 2
        [h_peak, color_entropy],    # 2
    ])
    
    return features / (np.linalg.norm(features) + 1e-8)  # L2 norm
```

**Kỳ Vọng**:
- ✅ +1-2% accuracy
- ✅ Better handling of similar colors
- ❌ Slightly more computation

**Khó độ**: ⭐⭐ Trung bình

---

#### ✅ **Option B: Learned Color Features (CNN-based)**

```python
class LearnedColorStream(nn.Module):
    """
    Thay vì handcrafted features, học color features qua CNN nhỏ.
    """
    def __init__(self):
        super().__init__()
        # Nhỏ CNN riêng cho color
        self.color_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 64),
            nn.ReLU(),
        )
    
    def forward(self, images):
        # Extract color patterns directly
        return self.color_cnn(images)  # [B, 64]
```

**Kỳ Vọng**:
- ✅ +2-3% accuracy
- ✅ Automatic feature learning
- ❌ Thêm parameters, cần tuning

**Khó độ**: ⭐⭐ Trung bình

---

### 3️⃣ MULTI-STREAM FUSION - Kết Hợp Nhiều Features

**Hiện Tại**: 2 streams (image + color)

#### ✅ **Option A: Shape Stream (Contour/Edge Features)**

```python
def extract_shape_features(image: np.ndarray) -> np.ndarray:
    """
    Trích xuất features từ hình dạng viên thuốc.
    Các viên hình tròn vs hình bầu dục khác biệt rõ.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # 2. Contour analysis
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return np.zeros(10, dtype=np.float32)
    
    cnt = max(contours, key=cv2.contourArea)
    
    # 3. Shape descriptors
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2)  # 1=perfect circle
    
    # 4. Eccentricity (elongation)
    (cx, cy), (w, h), angle = cv2.fitEllipse(cnt)
    eccentricity = np.sqrt(1 - (min(w, h) / max(w, h)) ** 2)
    
    # 5. Moments
    M = cv2.moments(cnt)
    hu = cv2.HuMoments(M).flatten()
    
    return np.concatenate([
        [area, perimeter, circularity, eccentricity, angle],
        hu[:5]
    ]).astype(np.float32)
```

**Kỳ Vọng**:
- ✅ +1-2% accuracy
- ✅ Giúp phân biệt viên tròn vs bầu dục
- ❌ Requires image preprocessing

**Khó độ**: ⭐⭐ Trung bình

---

#### ✅ **Option B: Texture Stream (LBP/Gabor)**

```python
from skimage.feature import local_binary_pattern
import numpy as np

def extract_texture_features(image: np.ndarray) -> np.ndarray:
    """
    Trích xuất texture features (bề mặt viên).
    Viên có chữ khắc vs trơn khác biệt.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Local Binary Pattern (LBP) - detect texture patterns
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    
    # Gabor filters - alternative texture approach
    kernels = []
    for theta in np.arange(0, np.pi, np.pi/4):  # 4 orientations
        for lambda_ in [3, 5, 7]:  # 3 scales
            kernel = gabor_kernel(lambda_, theta)
            kernels.append(kernel)
    
    # Response magnitudes
    gabor_features = []
    for kernel in kernels:
        response = cv2.filter2D(gray, cv2.CV_32F, kernel)
        gabor_features.extend([
            response.mean(),
            response.std(),
        ])
    
    return np.concatenate([
        lbp_hist / lbp_hist.sum(),  # normalize
        gabor_features
    ]).astype(np.float32)
```

**Kỳ Vọng**:
- ✅ +1-2% accuracy
- ✅特別 effective for imprinted pills
- ❌ Thêm dependencies (scikit-image)

**Khó độ**: ⭐⭐ Trung bình

---

#### 📊 **Multi-Stream Architecture**

```
Input Image
    ↓
┌───┴────┬───────┬──────────┐
│        │       │          │
▼        ▼       ▼          ▼
ResNet   Color   Shape     Texture
[512]    [32]    [10]      [24]
│        │       │          │
└───┬────┴───────┴──────────┘
    │
    ▼
Concatenate [512+32+10+24 = 578]
    │
    ▼
Classification Head [578 → 256 → num_classes]
    │
    ▼
Output Logits
```

**Tổng Kỳ Vọng**: +2-4% accuracy (combine effect)

**Khó độ**: ⭐⭐⭐ Khó

---

### 4️⃣ KNOWLEDGE GRAPH ENHANCEMENT

**Hiện Tại**: KG with color + shape + imprint similarity

#### ✅ **Option A: Prescription Context**

```python
def build_prescription_aware_kg(
    classifier_checkpoint: Path,
    prescription_data: Dict[int, List[str]],  # label_id → common prescriptions
    cache_dir: Path
) -> Dict:
    """
    Xây dựng KG biết context đơn thuốc.
    Ví dụ: Nếu đơn thuốc yêu cầu hạ sốt (paracetamol),
    nhưng model dự đoán viên kháng sinh → adjust confidence.
    """
    # Load classifier
    model, payload = load_classifier(checkpoint_path)
    
    # Build label embeddings
    label_embeddings = {}
    for label_id in range(num_classes):
        # Average embedding of all samples of this class
        samples = get_class_samples(label_id, cache_dir)
        embeddings = model.image_backbone(to_tensor(samples))
        label_embeddings[label_id] = embeddings.mean(dim=0)
    
    # Build prescription compatibility matrix
    prescription_compatibility = np.zeros((num_classes, num_classes))
    
    for label_id, prescriptions in prescription_data.items():
        for other_label_id, other_prescriptions in prescription_data.items():
            # How many prescription categories overlap?
            overlap = len(set(prescriptions) & set(other_prescriptions))
            compatibility = overlap / max(len(prescriptions), 1)
            prescription_compatibility[label_id, other_label_id] = compatibility
    
    # Save KG with prescription awareness
    kg = {
        "label_embeddings": label_embeddings,
        "visual_similarity": compute_similarity_matrix(label_embeddings),
        "prescription_compatibility": prescription_compatibility,
    }
    
    return kg

# Usage during inference
def inference_with_prescription_context(
    image: np.ndarray,
    prescription_labels: List[int],  # Expected drug categories
    kg: Dict
) -> Tuple[int, float]:
    """
    Dự đoán có xem xét context đơn thuốc.
    """
    logits = classifier(image)
    probs = softmax(logits)
    
    # Initial prediction
    pred = np.argmax(probs)
    
    # Adjust by prescription context
    if prescription_labels:
        context_boost = kg["prescription_compatibility"][pred, prescription_labels].mean()
        adjusted_probs = probs * (1 + 0.3 * context_boost)  # 30% boost
        pred = np.argmax(adjusted_probs)
    
    return pred, max(adjusted_probs)
```

**Kỳ Vọng**:
- ✅ +2-5% accuracy (với prescription context)
- ✅ Giảm error rate cho misclassification
- ❌ Cần thêm prescription database

**Khó độ**: ⭐⭐⭐ Khó (cần dữ liệu đơn thuốc)

---

#### ✅ **Option B: Temporal Consistency (Sequence Learning)**

```python
def build_sequence_aware_kg(
    image_sequences: List[List[np.ndarray]],  # Multi-image sequences of pills
    labels: List[List[int]]  # Ground truth for sequences
):
    """
    Học từ sequences: nếu liên tiếp thấy viên A, B, C
    theo pattern nào đó → có thể predict D tiếp theo.
    """
    # Build a Markov transition matrix
    transition_matrix = np.zeros((num_classes, num_classes))
    
    for sequence, label_seq in zip(image_sequences, labels):
        for i in range(len(label_seq) - 1):
            curr_label = label_seq[i]
            next_label = label_seq[i + 1]
            transition_matrix[curr_label, next_label] += 1
    
    # Normalize
    transition_matrix = transition_matrix / (transition_matrix.sum(axis=1, keepdims=True) + 1e-8)
    
    return {
        "transition_matrix": transition_matrix,  # P(next_class | current_class)
        "sequence_smooth": True
    }

# Usage
def inference_sequence(
    images: List[np.ndarray],
    kg_sequence: Dict
):
    """
    Dự đoán sequence with consistency.
    """
    predictions = []
    
    for i, image in enumerate(images):
        logits = classifier(image)
        probs = softmax(logits)
        
        if i > 0:
            # Boost predictions consistent with previous
            prev_pred = predictions[-1]
            consistency_boost = kg_sequence["transition_matrix"][prev_pred]
            probs = probs * (1 + 0.5 * consistency_boost)  # 50% boost
        
        pred = np.argmax(probs)
        predictions.append(pred)
    
    return predictions
```

**Kỳ Vọng**:
- ✅ +1-3% accuracy (trong sequence context)
- ✅ Batch inference tốt hơn single
- ❌ Cần multi-image data

**Khó độ**: ⭐⭐⭐⭐ Rất Khó

---

### 5️⃣ TRAINING IMPROVEMENTS

**Hiện Tại**: Cross-entropy + early stopping + data aug

#### ✅ **Option A: Advanced Augmentation**

```python
from albumentations import (
    Compose, RandomBrightnessContrast, Rotate, 
    ShiftScaleRotate, Blur, GaussNoise, CoarseDropout,
    Cutout, Mixup
)

def build_strong_transforms():
    """
    Albumentations: stronger augmentations than torchvision.
    """
    return Compose([
        # Geometric
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.7),
        Rotate(limit=45, p=0.5),
        
        # Color
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        
        # Artifacts
        Blur(blur_limit=3, p=0.3),
        GaussNoise(p=0.2),
        
        # Pixels
        CoarseDropout(max_holes=4, max_height=10, max_width=10, p=0.3),
        
        # Mixup: blend 2 images
        Mixup(alpha=0.2, p=0.2),
    ], bbox_params=...)
```

**Kỳ Vọng**:
- ✅ +1-2% accuracy
- ✅ Better generalization
- ❌ Need albumentations library

**Khó độ**: ⭐ Dễ

---

#### ✅ **Option B: Mixup + CutMix Strategies**

```python
def mixup(x1, x2, y1, y2, alpha=1.0):
    """
    Mix 2 images & labels: x_mix = λ*x1 + (1-λ)*x2
    """
    lam = np.random.beta(alpha, alpha)
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = lam * y1 + (1 - lam) * y2  # soft labels
    return x_mix, y_mix

def cutmix(x1, x2, y1, y2, alpha=1.0):
    """
    Cut rectangular region from x2 and paste to x1.
    """
    lam = np.random.beta(alpha, alpha)
    h, w = x1.shape[:2]
    
    # Random cut region
    cut_h = int(h * np.sqrt(1 - lam))
    cut_w = int(w * np.sqrt(1 - lam))
    
    x = np.random.randint(0, w - cut_w)
    y = np.random.randint(0, h - cut_h)
    
    x_mix = x1.copy()
    x_mix[y:y+cut_h, x:x+cut_w] = x2[y:y+cut_h, x:x+cut_w]
    
    # Soft label
    actual_lam = 1 - (cut_h * cut_w) / (h * w)
    y_mix = actual_lam * y1 + (1 - actual_lam) * y2
    
    return x_mix, y_mix
```

**Kỳ Vọng**:
- ✅ +1-2% accuracy
- ✅ Regularization effect
- ✅ Better robustness

**Khó độ**: ⭐⭐ Trung bình

---

#### ✅ **Option C: Ensemble Methods**

```python
class EnsembleClassifier:
    """
    Train multiple models với khác seed/architecture,
    combine predictions → higher accuracy.
    """
    def __init__(self):
        self.models = [
            self.build_model("resnet18", seed=42),
            self.build_model("resnet18", seed=43),
            self.build_model("efficientnet_b2", seed=42),
            self.build_model("resnet50", seed=42),
        ]
    
    def predict(self, image):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            logits = model(image)
            probs = softmax(logits)
            predictions.append(probs)
        
        # Ensemble: average + weighted voting
        ensemble_probs = np.mean(predictions, axis=0)  # Average
        # Or: weighted average based on validation accuracy
        
        return np.argmax(ensemble_probs)
    
    def predict_ensemble_top_k(self, image, k=3):
        """Get top-k predictions with ensemble confidence."""
        ensemble_probs = self.get_ensemble_probs(image)
        top_k_indices = np.argsort(ensemble_probs)[-k:][::-1]
        return top_k_indices, ensemble_probs[top_k_indices]
```

**Kỳ Vọng**:
- ✅ +2-4% accuracy
- ✅ Much more robust
- ❌ 4x inference time, 4x model size

**Khó độ**: ⭐⭐ Trung bình

---

### 6️⃣ POST-PROCESSING & SAFETY

#### ✅ **Option A: Confidence Calibration**

```python
def calibrate_confidence(
    model_predictions: np.ndarray,  # Raw model output
    calibration_data_metrics: Dict  # From validation set
) -> np.ndarray:
    """
    Model confidence ≠ true accuracy.
    Ví dụ: model 90% sure nhưng chỉ 70% đúng.
    Calibrate để confidence match true accuracy.
    """
    # Temperature scaling
    logits = model_predictions
    temperature = get_optimal_temperature(calibration_data_metrics)
    calibrated_probs = softmax(logits / temperature)
    
    return calibrated_probs

def get_optimal_temperature(metrics_dict):
    """
    使用temperature scaling để match predicted confidence
    với actual accuracy.
    
    T=1: no scaling
    T>1: reduce confidence (model overconfident)
    T<1: increase confidence (model underconfident)
    """
    import scipy.optimize as opt
    
    def nll_loss(T):
        probs = softmax(logits / T)
        return -np.mean(np.log(probs[np.arange(len(y)), y]))
    
    T_optimal = opt.minimize_scalar(nll_loss, bounds=(0.1, 10)).x
    
    return T_optimal
```

**Kỳ Vọng**:
- ✅ Better calibrated uncertainty
- ✅ Can use confidence for rejection
- ✅ No retraining needed

**Khó độ**: ⭐⭐ Trung bình

---

#### ✅ **Option B: Uncertainty Quantification (Bayesian)**

```python
class BayesianClassifier(nn.Module):
    """
    Bayesian Neural Networks: predict with uncertainty.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.fc_mean = nn.Linear(512, num_classes)
        self.fc_logvar = nn.Linear(512, num_classes)  # log-variance
        self.backbone = models.resnet18()
    
    def forward(self, x):
        features = self.backbone(x)
        mean = self.fc_mean(features)
        logvar = self.fc_logvar(features)
        var = torch.exp(logvar)
        std = torch.sqrt(var)
        
        return mean, std  # Mean + std deviation
    
    def predict_with_uncertainty(self, x, num_samples=10):
        """
        Monte Carlo Dropout: run inference multiple times,
        measure variance → uncertainty.
        """
        self.train()  # Keep dropout active
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                mean, std = self.forward(x)
                sample = torch.normal(mean, std)
                predictions.append(softmax(sample))
        
        predictions = torch.stack(predictions)
        
        # Mean prediction + uncertainty (std)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty
```

**Kỳ Vọng**:
- ✅ Uncertainty estimates
- ✅ Can reject low-confidence predictions
- ✅ Better for deployment
- ❌ More complex training

**Khó độ**: ⭐⭐⭐ Khó

---

### 7️⃣ DEPLOYMENT OPTIMIZATION

#### ✅ **Option A: Model Quantization**

```python
# Quantize model to INT8 (4x smaller, faster)

# PyTorch static quantization
model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Check size
original_size = sum(p.numel() for p in model.parameters()) * 4 / 1e6  # MB
quantized_size = sum(p.numel() for p in quantized_model.parameters()) * 1 / 1e6
print(f"Compression: {original_size:.1f}MB → {quantized_size:.1f}MB")

# Inference
quantized_model(input_image)  # Same API, 4x faster
```

**Kỳ Vọng**:
- ✅ 4x smaller model (45MB → 12MB)
- ✅ 4x faster inference (10ms → 2.5ms)
- ❌ ~1-2% accuracy loss

**Khó độ**: ⭐ Dễ

---

#### ✅ **Option B: Knowledge Distillation**

```python
class StudentModel(nn.Module):
    """
    Nhỏ hơn model để deploy, học từ teacher model lớn.
    """
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small()
        self.classifier = nn.Linear(576, num_classes)
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

def distillation_loss(student_logits, teacher_logits, labels, temperature=4):
    """
    KL divergence: student tries to match teacher's soft probabilities.
    """
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # Soft targets from teacher
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    # Combined loss
    alpha = 0.5
    loss = alpha * ce_loss + (1 - alpha) * kl_loss
    
    return loss

# Training
teacher = load_best_model()
student = StudentModel().to(device)
optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

for epoch in range(50):
    for batch in dataloader:
        images, _, labels = batch
        
        student_logits = student(images)
        teacher_logits = teacher(images).detach()
        
        loss = distillation_loss(student_logits, teacher_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Kỳ Vọng**:
- ✅ 2x smaller student model (22MB)
- ✅ 2x faster inference (5ms)
- ✅ Only ~1-2% accuracy loss (vs teacher)

**Khó độ**: ⭐⭐⭐ Khó

---

#### ✅ **Option C: ONNX Export + TensorRT**

```python
# Export to ONNX (universal format)
model.eval()
dummy_input = torch.randn(1, 3, 160, 160).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "pill_classifier.onnx",
    input_names=["image"],
    output_names=["logits"],
    opset_version=12,
    export_params=True,
)

# TensorRT optimization (NVIDIA GPUs)
import tensorrt as trt

# Convert ONNX → TensorRT engine
# Automatic optimization for target GPU
# 2-3x speedup on NVIDIA GPUs
```

**Kỳ Vọng**:
- ✅ 2-3x faster on NVIDIA GPUs
- ✅ Cross-platform (mobile, edge)
- ❌ Requires platform-specific builds

**Khó độ**: ⭐⭐⭐ Khó

---

## 📊 COMPARISON TABLE - Tất Cả Hướng Cải Thiện

| Hướng | Difficulty | Accuracy Gain | Speed Impact | Implementation Time |
|------|-----------|--------------|-------------|------------------|
| **1. Upgrade Backbone** | ⭐ | +1-5% | -| 1-2 days |
| **1A. ResNet50** | ⭐ | +2-3% | -1.5x | 1 day |
| **1B. EfficientNet** | ⭐ | +1-2% | 0.8x (better!) | 1 day |
| **1C. Vision Transformer** | ⭐ | +3-5% | -5x | 2 days |
| **2. Advanced Color Stream** | ⭐⭐ | +1-2% | +10% | 2-3 days |
| **2A. Expanded Features** | ⭐⭐ | +1% | +5% | 1 day |
| **2B. Learned Features** | ⭐⭐ | +2% | +15% | 3 days |
| **3. Multi-Stream Fusion** | ⭐⭐⭐ | +2-4% | -10% | 5-7 days |
| **3A. Shape Stream** | ⭐⭐ | +1% | +20% | 2 days |
| **3B. Texture Stream** | ⭐⭐ | +1-2% | +30% | 3 days |
| **4. KG Enhancement** | ⭐⭐⭐ | +1-4% | +5% | 3-5 days |
| **4A. Prescription Context** | ⭐⭐⭐ | +2-4% | +10% | 4 days |
| **4B. Sequence Learning** | ⭐⭐⭐⭐ | +1-3% | -2% | 7-10 days |
| **5. Training Improvements** | ⭐-⭐⭐⭐ | +1-4% | - | 1-5 days |
| **5A. Strong Augmentation** | ⭐ | +1-2% | - | 1 day |
| **5B. Mixup/CutMix** | ⭐⭐ | +1% | - | 2 days |
| **5C. Ensemble** | ⭐⭐ | +2-4% | -3.5x | 3 days |
| **6. Uncertainty** | ⭐⭐⭐ | 0-1% | +50% inference | 4 days |
| **6A. Confidence Calibration** | ⭐⭐ | 0% | - | 1 day |
| **6B. Bayesian NN** | ⭐⭐⭐ | 0-1% | +40% | 4 days |
| **7. Deployment** | ⭐-⭐⭐⭐ | -1-2% | +3-10x | 1-7 days |
| **7A. Quantization** | ⭐ | -1-2% | +4x | 1 day |
| **7B. Knowledge Distillation** | ⭐⭐⭐ | -1-2% | +2x | 5 days |
| **7C. ONNX/TensorRT** | ⭐⭐⭐ | 0% | +2-3x | 3 days |

---

## 🎯 RECOMMENDATION ROADMAP

### Phase 1: Quick Wins (1-2 tuần)
```
1. Strong Augmentation (albumentations) → +1-2%
2. Upgrade backbone to EfficientNet-B2 → +1-2%
3. Temperature calibration → Better uncertainty
Total: +2-3% improvement, high ROI
```

### Phase 2: Medium Effort (2-3 tuần)
```
1. Multi-stream fusion (shape + texture) → +2-3%
2. Advanced color features → +1%
3. Prescription context in KG → +2-4%
Total: +4-7% improvement
```

### Phase 3: Advanced (1-2 tháng)
```
1. Ensemble models → +2-4%
2. Bayesian uncertainty → Better deployment
3. Knowledge distillation for deployment
Total: Production-ready system
```

### Phase 4: Deployment (tuỳ chọn)
```
1. Quantization (4x speed)
2. ONNX export (cross-platform)
3. TensorRT (GPU optimization)
```

---

## 🚀 QUICK START - Implement First Improvement

### EfficientNet (Easiest, +1-2% Accuracy)

```python
# File: train.py, thay đổi function build_model()

def build_model(
    num_classes: int,
    model_variant: str = "efficientnet_b2",  # Changed from resnet18
    color_feature_dim: int = DEFAULT_COLOR_BINS * 3,
    pretrained: bool = True,
) -> nn.Module:
    
    if model_variant == "cg_imif_color_fusion":
        # EfficientNet backbone
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b2(weights=weights)
        
        # Remove classifier head
        in_features = backbone.classifier[1].in_features  # 1408 for B2
        backbone.classifier = nn.Identity()
        
        # Color fusion remains same
        color_head = nn.Sequential(
            nn.LayerNorm(color_feature_dim),
            nn.Linear(color_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        
        classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes),
        )
        
        model = nn.Module()
        model.image_backbone = backbone
        model.color_head = color_head
        model.classifier = classifier
        model.uses_color_stream = True
        
        return model
```

**Command**:
```bash
python train.py --model-variant cg_imif_color_fusion --epochs 60
```

---

## 🎓 References & Papers

1. **EfficientNet**: Tan & Le (2019) - *EfficientNet: Rethinking Model Scaling for CNNs*
2. **Vision Transformer**: Dosovitskiy et al. (2020) - *An Image is Worth 16x16 Words*
3. **Knowledge Distillation**: Hinton et al. (2015) - *Distilling Knowledge in Neural Networks*
4. **Mixup**: Zhang et al. (2017) - *mixup: Beyond Empirical Risk Minimization*
5. **Focal Loss**: Lin et al. (2017) - *Focal Loss for Dense Object Detection*
6. **Bayesian NN**: Gal & Ghahramani (2016) - *Dropout as a Bayesian Approximation*

---

## ✅ Conclusion

**Mô hình của bạn HAS HUGE POTENTIAL** để cải thiện!

**Recommendation**: 
- 🥇 **Ngắn hạn** (1-2 tuần): Upgrade backbone + strong augmentation → **+2-3% accuracy**
- 🥈 **Trung hạn** (3-4 tuần): Multi-stream fusion + KG enhancement → **+4-7% accuracy**
- 🥉 **Dài hạn** (1-2 tháng): Ensemble + Bayesian uncertainty → **Production-ready**

**Trước khi làm**, đánh giá:
1. Accuracy target của bạn là bao nhiêu?
2. Inference speed constraint?
3. Model size limit?
4. Deployment platform (GPU/CPU/Mobile)?

Các yếu tố này sẽ quyết định **hướng phát triển tối ưu** cho dự án của bạn! 🎯
