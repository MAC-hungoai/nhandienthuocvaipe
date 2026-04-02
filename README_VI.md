# DỰ ÁN VAIPE: PHÂN LOẠI VÀ PHÁT HIỆN VIÊN THUỐC

**Tiếng Việt | [English](README.md)**

> Dự án này xây dựng hệ thống **phát hiện (Detection)** và **phân loại (Classification)** viên thuốc từ ảnh, sử dụng các kỹ thuật Deep Learning hiện đại.

## 📋 MỤC LỤC

1. [Tổng Quan Dự Án](#tổng-quan-dự-án)
2. [Lý Thuyết Nền Tảng](#lý-thuyết-nền-tảng)
3. [Công Nghệ & Kỹ Thuật Dùng](#công-nghệ--kỹ-thuật-dùng)
4. [Kiến Trúc Mô Hình](#kiến-trúc-mô-hình)
5. [Cách Hoạt Động Chi Tiết](#cách-hoạt-động-chi-tiết)
6. [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
7. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
8. [Kết Quả & Hiệu Năng](#kết-quả--hiệu-năng)

---

## 🎯 Tổng Quan Dự Án

### Mục Tiêu Chính

Dự án này xây dựng một **pipeline hoàn chỉnh** để:

- **Phát hiện (Detection)**: Tìm ra vị trí tất cả các viên thuốc trong một ảnh qua bounding box (bbox)
- **Phân loại (Classification)**: Xác định chính xác loại thuốc dựa trên ảnh crop từng viên
- **Re-ranking (Knowledge Graph)**: Cải thiện độ chính xác bằng cách sử dụng pattern hoặc thông tin liên quan

### Bộ Dữ Liệu

- **Nguồn**: VAIPE (Việt Nam - Dữ liệu ảnh viên thuốc)
- **Cấu trúc**: 
  - `archive(1)/public_train/pill/image/` - Ảnh viên thuốc gốc (có nhiều viên trong 1 ảnh)
  - `archive(1)/public_train/pill/label/` - Nhãn JSON chứa bbox và label_id cho từng viên
- **Tổng số class**: 108 loại viên thuốc (`label_id`) trong split classifier hiện tại

### Quy Trình Xử Lý

```
Ảnh viên thuốc gốc
        ↓
  [DETECTION - Faster R-CNN]
  → Tìm bbox từng viên thuốc
        ↓
  Crop từng viên thành ảnh riêng
        ↓
  [CLASSIFICATION - ResNet18 + Color Fusion]
  → Dự đoán loại thuốc
        ↓
  [KNOWLEDGE GRAPH - Ranking]
  → Cải thiện dựa trên thông tin màu, hình dạng, imprint
        ↓
  Kết quả: Label + Xác suất
```

---

## 🧠 Lý Thuyết Nền Tảng

### 1. Convolutional Neural Network (CNN)

**Khái niệm**: CNN là một loại mạng nơ-ron sâu thiết kế để xử lý dữ liệu hình ảnh.

**Cấu trúc**:
- **Convolution layers** (Tích chập): Áp dụng các kernel nhỏ để trích xuất features (cạnh, hình dạng, màu)
- **Pooling layers** (Gộp): Giảm kích thước để tiết kiệm tính toán
- **Fully connected layers** (Kết nối đầy đủ): Phân loại dựa trên features đã trích xuất

**Công thức cơ bản**:
$$y = f(\mathbf{W} * \mathbf{x} + \mathbf{b})$$

Trong đó:
- $\mathbf{W}$ là kernel (trọng số)
- $\mathbf{x}$ là đầu vào
- $*$ là phép tích chập
- $f$ là hàm activation (ReLU, Sigmoid...)

### 2. ResNet (Residual Network)

**Ý tưởng**: Sử dụng **kết nối skip** (residual connections) để cho phép huấn luyện mạng sâu hơn mà không bị vanishing gradient.

**Skip Connection**:
$$y = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

Thay vì học: $y = \mathcal{F}(\mathbf{x})$, ResNet học phần dư: $\mathcal{F}(\mathbf{x}) = y - \mathbf{x}$

**Ưu điểm**:
- ✅ Cho phép xây dựng mạng rất sâu (50+ layers)
- ✅ Tốc độ hội tụ nhanh hơn
- ✅ Chống lại vanishing gradient problem

**Trong dự án**: Sử dụng **ResNet18** (18 layers) - nhẹ, nhanh, đủ mạnh cho bài toán này.

### 3. Faster R-CNN (Object Detection)

**Mục đích**: Phát hiện vị trí và phân loại đối tượng trong ảnh cùng lúc.

**Kiến trúc**:
1. **Backbone** (ResNet50): Trích xuất features từ ảnh
2. **RPN (Region Proposal Network)**: Dự đoán những vùng có thể chứa đối tượng
3. **ROI Pooling**: Chuẩn hóa kích thước của các region proposals
4. **Classification & Bounding Box Regression**: 
   - Phân loại từng region
   - Điều chỉnh bbox để chính xác hơn

**Công thức Loss**:
$$L = L_{cls} + \lambda L_{bbox}$$

Trong đó:
- $L_{cls}$: Cross-entropy loss cho phân loại
- $L_{bbox}$: Smooth L1 loss cho điều chỉnh bbox
- $\lambda$: Trọng số cân bằng (thường = 1)

### 4. Transfer Learning

**Ý tưởng**: Sử dụng các mô hình đã được huấn luyện trên tập dữ liệu lớn (ImageNet) làm điểm khởi đầu, sau đó fine-tune cho bài toán cụ thể.

**Lợi ích**:
- 🚀 Giảm thời gian huấn luyện
- 📊 Cần ít dữ liệu hơn
- 📈 Hiệu suất tốt hơn

**Trong dự án**: Tất cả model (ResNet18, MobileNetV3, ResNet50) đều sử dụng pretrained weights từ ImageNet.

### 5. Data Augmentation

**Mục đích**: Tăng tính đa dạng của dữ liệu huấn luyện để mô hình tổng quát hóa tốt hơn.

**Các kỹ thuật dùng trong dự án**:
- **Horizontal Flip**: Lật ngang ảnh (xác suất 50%)
- **Vertical Flip**: Lật dọc ảnh (xác suất 25%)
- **Rotation**: Xoay ảnh ±20°
- **Color Jitter**: Thay đổi brightness, contrast, saturation, hue
- **Random Erasing**: Che một phần ngẫu nhiên của ảnh (xác suất 20%)

### 6. Focal Loss

**Vấn đề**: Trong các bộ dữ liệu không cân bằng, các lớp có nhiều mẫu sẽ dominan, làm các lớp ít mẫu bị "lơ lửng".

**Giải pháp - Focal Loss**:
$$FL(p_t) = -\alpha_t(1-p_t)^{\gamma} \log(p_t)$$

Trong đó:
- $p_t$ = xác suất dự đoán đúng
- $\gamma$ = focusing parameter (thường = 2)
- $(1-p_t)^{\gamma}$ = down-weight các mẫu dễ

**Hiệu quả**: Tập trung vào các mẫu khó (hard examples), cải thiện đặc biệt với dữ liệu không cân bằng.

---

## 🛠️ Công Nghệ & Kỹ Thuật Dùng

### Framework & Thư Viện

| Công Nghệ | Phiên Bản | Mục Đích |
|-----------|----------|---------|
| **PyTorch** | 2.x | Framework deep learning chính |
| **Torchvision** | 0.15+ | Model zoo (ResNet, Faster R-CNN) + tính toán detection |
| **OpenCV** | 4.x | Xử lý ảnh (resize, crop, color conversion) |
| **NumPy** | Latest | Tính toán ma trận và vector |
| **Scikit-learn** | Latest | Metrics (confusion matrix, F1-score, precision, recall) |
| **Matplotlib** | Latest | Visualize training curves & confusion matrix |
| **Pillow** | Latest | Tải & xử lý ảnh PIL |

### Các Kỹ Thuật Quan Trọng

#### 1️⃣ **CG-IMIF Color Fusion** (Mô Hình Đặc Biệt)

**Tên đầy đủ**: Color & Grey Image-wise Multi-channel Image Fusion

**Ý tưởng**: Kết hợp hai stream thông tin:
- **Image Stream**: Features từ ảnh RGB qua ResNet18
- **Color Stream**: Thông tin màu HSV được xử lý riêng

**Công thức**:
```
Image Features: f_img = ResNet18(image)  → [512 dims]

Color Histogram:
  - Tách ảnh sang HSV
  - Tính histogram cho H, S, V (mỗi channel 8 bins)
  - Total: 8*3 = 24 dims
  
Color Features: f_color = MLP(color_hist)  → [64 dims]

Fusion: features = [f_img ⊕ f_color]  → [512+64 = 576 dims]

Classification: logits = FC(features)
```

**Lợi ích**:
- ✅ Ghi nhận thông tin màu một cách rõ ràng (viên p xanh vs i đỏ)
- ✅ Bổ sung thông tin geometric bằng color distribution
- ✅ Nhẹ hơn so với multi-stream CNN phức tạp

**Nguồn gốc**: Lấy cảm hứng từ CG-IMIF paper (~98.8% accuracy trên các bộ dữ liệu tương tự)

#### 2️⃣ **Hard Example Mining** (Cho Detection)

**Ý tưởng**: Tập trung huấn luyện vào những ảnh khó (high loss) để cải thiện các edge cases.

**Cách hoạt động**:
1. Trong training loop: Tính loss cho từng ảnh
2. Chọn top-K% ảnh có loss cao nhất (hard examples)
3. Epoch tiếp theo: Lặp lại các ảnh khó này với xác suất cao (boosting)

**Công thức**:
```python
# Sau epoch:
hard_losses = compute_losses(train_images)  # [num_images]
threshold = np.percentile(hard_losses, 85)  # top 15%
hard_multiplier = 1.75  # Nhân 1.75x cho ảnh khó

# Epoch tiếp (sau warmup):
sampling_probs = np.where(hard_losses >= threshold, 
                          hard_multiplier, 
                          1.0)
# Resample theo probs này
```

**Khi nào bắt đầu**: Sau 1 epoch warmup (để model acclimate)

#### 3️⃣ **Knowledge Graph Re-ranking**

**Mục đích**: Sau khi classification, cải thiện dự đoán bằng thông tin ngữ cảnh.

**Xây dựng KG**:
```
1. Tạo embedding cho mỗi lớp thuốc (dùng best model)
2. Tính similarity ma trận giữa tất cả lớp (color + shape similarity)
3. Lưu dưới dạng JSON graph structure
```

**Sử dụng KG**:
```
Input: ảnh crop
  ↓
Initial classification: P(y|x) từ model
  ↓
Query KG: Lấy top-K lớp tương tự dựa trên:
  - Visual similarity (feature distance)
  - Color similarity (histogram distance)
  - Context similarity (prescription knowledge)
  ↓
Re-rank: Kết hợp initial prob + KG scores
  ↓
Final prediction: argmax(combined_score)
```

**Công thức kết hợp**:
$$score_i = w_v \cdot sim_{visual,i} + w_c \cdot sim_{color,i} + w_{ctx} \cdot sim_{context,i}$$

Trong đó: $w_v=0.35, w_c=0.20, w_{ctx}=0.15$

---

## 🏗️ Kiến Trúc Mô Hình

### Mô Hình 1: Single-Pill Classifier

```
┌─────────────────────────────────────────┐
│   Input Image (160×160 RGB)             │
└──────────────┬──────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  ResNet18 Backbone   │
    │  (pretrained,        │
    │   ImageNet)          │
    │                      │
    │  64→64→128→256→512   │  Conv+ReLU+Skip
    │  Output: [512]       │
    └──────────┬───────────┘
               │
          ┌────┴─────────────────────┐
          │                          │
          ▼                          ▼
    ┌──────────────┐      ┌──────────────────┐
    │ Image&Stream │      │  Color Stream    │
    │   [512]      │      │                  │
    │              │      │ HSV Histogram    │
    │              │      │ (24 dims)        │
    │              │      │       │          │
    │              │      │       ▼          │
    │              │      │   LayerNorm      │
    │              │      │       │          │
    │              │      │       ▼          │
    │              │      │   FC(24→64)      │
    │              │      │   + ReLU         │
    │              │      │   + Dropout(15%) │
    │              │      │       │          │
    │              │      │       ▼          │
    │              │      │   FC(64→64)      │
    │              │      │   + ReLU         │
    │              │      │   [64]           │
    └──────┬───────┘      └─────┬────────────┘
           │                    │
           └────────┬───────────┘
                    │
                    ▼
          ┌──────────────────┐
          │   Concatenate    │
          │   [512+64=576]   │
          └──────┬───────────┘
                 │
                 ▼
          ┌──────────────────────┐
          │  Classification Head │
          │                      │
          │  Dropout(35%)        │
          │       │              │
          │       ▼              │
          │  FC(576→256)         │
          │  + ReLU              │
          │       │              │
          │       ▼              │
          │  Dropout(20%)        │
          │       │              │
          │       ▼              │
          │  FC(256→#classes)    │
          │                      │
          │  Output: [num_cls]   │
          └──────┬────────────────┘
                 │
                 ▼
          ┌─────────────────┐
          │  Softmax + Loss │
          └─────────────────┘
```

### Mô Hình 2: Multi-Pill Detector

```
Backbone Options:
├─ MobileNetV3 (nhẹ, nhanh) → ResNet50 FPN (cân bằng)
├─ ResNet50 FPN (mạnh, chậm hơn)

┌─────────────────────────────────────────┐
│   Input Image (~640×640)                │
└──────────────┬──────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Backbone            │
    │  FPN (Feature        │
    │   Pyramid Network)   │
    │                      │
    │  Tạo features ở      │
    │  nhiều scales:       │
    │  P2, P3, P4, P5      │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  RPN (Region         │
    │  Proposal Network)   │
    │                      │
    │  - Tạo anchors 9×    │
    │    kích thước khác   │
    │  - Predict objectness│
    │  - Adjust bbox       │
    │  - NMS               │
    │  → ~1000 proposals   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  ROI Pooling         │
    │  (chuẩn hóa region)  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Classification &    │
    │  Bbox Regression     │
    │                      │
    │  - FC layers         │
    │  - Predict class     │
    │  - Refine bbox       │
    │  - NMS again         │
    │  → Final detections  │
    └──────────┬───────────┘
               │
               ▼
    ┌─────────────────────────┐
    │  Output:                │
    │  - Boxes: [N, 4]        │
    │  - Scores: [N]          │
    │  - Classes: [N]         │
    │  - Filters by score >α  │
    │    (α=0.3 default)      │
    └─────────────────────────┘
```

---

## 🔍 Cách Hoạt Động Chi Tiết

### A. Training Classifier

#### Bước 1: Chuẩn Bị Dữ Liệu

```python
# Phân tích từ archive(1)/public_train/pill/
#
# Cấu trúc:
#   public_train/pill/
#   ├── image/
#   │   ├── image_001.jpg
#   │   ├── image_002.jpg
#   │   └── ...
#   └── label/
#       ├── image_001.json (chứa list bbox & label_id)
#       ├── image_002.json
#       └── ...

# 1. Đọc tất cả ảnh & nhãn
# 2. Với mỗi ảnh:
#    - Tìm bounding box của từng viên thuốc
#    - Crop viên đó ra (+ padding 12%)
#    - Resize về 160×160
#    - Lưu cache JPEG
#
# Kết quả: ~30,000+ crops cho ~200 classes
```

#### Bước 2: Split Dữ Liệu

```python
# Chia train/val/test theo tỷ lệ 80/10/10
# ĐẶC BIỆT: Chia theo class để đảm bảo cân bằng

for label_id in all_labels:
    samples = all_crops_of_this_label  # N mẫu
    shuffle(samples)
    
    # Chia cân bằng
    train_end = int(N * 0.80)
    val_end = int(N * 0.90)
    
    train.extend(samples[:train_end])
    val.extend(samples[train_end:val_end])
    test.extend(samples[val_end:])
```

#### Bước 3: Huấn Luyện

```
FOR epoch = 1 TO max_epochs:
  FOR batch IN train_loader:
    # Forward pass
    image_tensor = batch.images  # [B, 3, 160, 160]
    color_hist = batch.colors    # [B, 24]
    labels = batch.labels        # [B]
    
    logits = model(image_tensor, color_hist)  # [B, num_classes]
    loss = criterion(logits, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  # Validation
  FOR batch IN val_loader:
    val_loss = compute_loss(model(batch))
  
  # Scheduler: Giảm LR nếu val_loss không giảm
  scheduler.step(val_loss)
  
  # Early stopping
  IF val_loss < best_val_loss:
    save model as "best_model.pth"
    patience = 0
  ELSE:
    patience += 1
    IF patience > 6:
      BREAK (early stopping)

# Load best model & evaluate on test set
```

#### Bước 4: Đánh Giá

```
Metrics tính toán:
- Top-1 Accuracy: nhãn đúng đứng ở hạng 1
- Top-3 Accuracy: nhãn đúng xuất hiện trong 3 dự đoán có xác suất cao nhất
- Per-class Accuracy: Cho từng loại thuốc
- Precision, Recall, F1-score (macro)
- Confusion Matrix
```

### B. Training Detector

```
1. Chuẩn bị dữ liệu
   - Đầu vào: Ảnh gốc (~640×640) + list bbox
   - Không cần crop

2. Data Augmentation cho Detection:
   - Random resize (600-1024)
   - Random horizontal flip
   - Color jitter
   
3. Loss function:
   Loss = Loss_objectness + Loss_bbox_rpn + Loss_classification + Loss_bbox_rcnn
   
4. Optimization:
   - AdamW optimizer
   - Gradient clipping: max norm = 1.0
   - ReduceLROnPlateau scheduler
   
5. Hard Example Mining (sau epoch 1):
   - Tính detection loss trên từng ảnh
   - Chọn top 15% ảnh khó
   - Lặp lại với weight 1.75× trong batch tiếp theo
```

### C. Inference (Dự Đoán)

```
INPUT: ảnh viên thuốc tròn (~160×160)

STEP 1: DETECTION
  model_detector(input_image_large)
  → [(x1,y1,x2,y2, conf, class_id), ...]
  → Filter by confidence > 0.3
  → NMS → ~10-20 boxes

STEP 2: CLASSIFICATION (cho mỗi box)
  FOR each_box:
    crop_image = extract_crop_from_box(input, box)
    crop_image = resize_to_160x160(crop_image)
    color_hist = extract_hsv_histogram(crop_image)
    
    logits = model_classifier(crop_image, color_hist)
    probs = softmax(logits)
    
    pred_class = argmax(logits)
    pred_score = max(probs)

STEP 3: KNOWLEDGE GRAPH RE-RANKING (optional)
  FOR each_box:
    initial_top5 = top-5 dự đoán từ step 2
    
    FOR each candidate IN top5:
      kg_score = KG.query(pred_class, candidate_class)
      combined_score = 0.7 * model_score + 0.3 * kg_score
    
    final_class = argmax(combined_scores)

OUTPUT: [(box, final_class, final_score), ...]
```

---

## 📁 Cấu Trúc Thư Mục

```
doan2/
│
├── src/                        # Source code (mới tổ chức)
│   ├── models/
│   │   ├── train.py           # Training classifier
│   │   └── detection_train.py # Training detector
│   ├── inference/
│   │   ├── test.py            # Eval classifier
│   │   ├── detection_test.py  # Eval detector
│   │   ├── demo_infer.py      # CLI inference
│   │   └── web_demo.py        # Web UI
│   └── utils/
│       ├── detection_utils.py # Detection utilities
│       ├── knowledge_graph.py # KG re-ranking
│       └── knowledge_graph_benchmark.py
│
├── checkpoints/               # Trained models & results
│   ├── best_model.pth        # Best classifier
│   ├── final_model.pth       # Final classifier
│   ├── history.json          # Training history
│   ├── test_metrics.json     # Evaluation results
│   ├── training_curves.png   # Loss/accuracy curves
│   │
│   ├── color_fusion_v1/      # Color fusion experiments
│   ├── crop_cache_160/       # Cached crops for fast loading
│   │   ├── images/           # Resized crop images
│   │   └── metadata.json     # Crop metadata
│   │
│   ├── detector/             # Detection models
│   │   ├── baselines/
│   │   ├── best/
│   │   └── experiments/
│   │
│   ├── benchmarks/           # Benchmark results
│   │   ├── kg_benchmark_full/
│   │   └── kg_benchmark_selective_full/
│   │
│   ├── knowledge_graph_vaipe.json  # Pre-built KG
│   └── detection_records_cache.json # Detection data cache
│
├── archive (1)/              # Dataset
│   ├── public_train/
│   │   └── pill/
│   │       ├── image/        # Training images
│   │       └── label/        # Training labels
│   └── public_test/
│       └── pill/
│           ├── image/        # Test images
│           └── label/        # Test labels (not in train)
│
├── data/                      # Additional data
│   ├── cache/                 # Runtime caches
│   └── user_real_photos/      # Ảnh thật bổ sung để bootstrap / fine-tune
│       └── pill/
│           ├── image/
│           └── label/
│
├── outputs/                   # Inference outputs
│   ├── detections.json
│   ├── classifications.json
│   └── visualizations/
│
├── docs/                      # Documentation
│   ├── README_VI.md          # Vietnamese readme (THIS FILE)
│   └── ARCHITECTURE.md
│
├── scripts/                   # Helper scripts
│   ├── train_classifier.bat
│   ├── train_classifier_real_adapt.bat
│   ├── train_detector_real_adapt.bat
│   ├── bootstrap_real_photo_labels.bat
│   └── validate_real_photo_dataset.bat
│
├── README.md                  # Original English readme
├── app_streamlit_modern.py    # Giao diện Streamlit hiện tại
├── app_streamlit.py           # Giao diện Streamlit cũ
├── train.py                   # Main training script (classifier)
├── detection_train.py         # Main training script (detector)
├── test.py                    # Testing script
├── detection_test.py          # Detection testing
├── detection_utils.py         # Shared detection utilities
├── knowledge_graph.py         # KG implementation
├── knowledge_graph_benchmark.py # KG benchmarking
├── demo_infer.py              # Inference wrapper
├── web_demo.py                # Web UI tối giản
│
└── __pycache__/ & .venv/      # Runtime directories
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Cài Đặt

```bash
# 1. Clone hoặc download project
cd doan2

# 2. Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Cài đặt dependencies cần thiết
pip install -r requirements.txt
```

### Chạy Giao Diện Hiện Tại (Khuyến Nghị)

```powershell
# Giao diện Streamlit hiện tại
.\.venv\Scripts\python.exe -m streamlit run app_streamlit_modern.py --server.port 8515

# Hoặc dùng script có sẵn
scripts\run_app_8515.bat
```

Sau đó mở trình duyệt tại:

```text
http://localhost:8515
```

**Ghi chú**:
- `app_streamlit_modern.py` là giao diện đang được dùng và đã được tối ưu lại phần hero, sidebar, thẻ metric, bảng kết quả và trải nghiệm upload ảnh.
- Nếu lệnh `streamlit` không nhận, dùng `python -m streamlit ...` như trên để ổn định hơn trên Windows.
- Repo không đẩy kèm dataset, checkpoint và `.venv`; xem thêm `docs/SETUP.md`, `checkpoints/README.md`, `data/README.md`.

### Chạy Giao Diện Cũ / Web Demo Tối Giản

```bash
# Streamlit bản cũ
python -m streamlit run app_streamlit.py

# Web app tối giản không cần Streamlit
python web_demo.py --host 127.0.0.1 --port 8501
```

### Training Classifier

```bash
# Default settings
python train.py

# Custom output directory
python train.py --output-dir checkpoints/my_experiment

# Reuse existing crop cache (nhanh hơn)
python train.py --output-dir checkpoints/new_run \
                 --cache-dir checkpoints/crop_cache_160

# Tuning hyperparameters
python train.py --epochs 100 \
                 --patience 8 \
                 --batch-size 128 \
                 --lr 5e-4 \
                 --model-variant cg_imif_color_fusion \
                 --loss-type focal \
                 --focal-gamma 2.5
```

**Các hyperparameter quan trọng**:
- `--image-size`: Kích thước ảnh input (160 default)
- `--model-variant`: "resnet18" hoặc "cg_imif_color_fusion"
- `--color-bins`: Bins của HSV histogram (8 default)
- `--sampler-power`: Class-balanced sampling strength (0.5 default)
- `--loss-type`: "cross_entropy", "weighted_ce", hoặc "focal"
- `--deterministic`: Chế độ reproducible (True default)

### Training Detector

```bash
# Default MultiPill Detection
python detection_train.py

# Custom settings
python detection_train.py --model-name fasterrcnn_resnet50_fpn_v2 \
                          --epochs 15 \
                          --batch-size 4 \
                          --output-dir checkpoints/detector_v2

# Với Hard Mining
python detection_train.py --hard-mining-topk 0.15 \
                          --hard-mining-boost 1.75 \
                          --hard-mining-warmup 2
```

### Testing & Evaluation

```bash
# Test classifier trên held-out test set
python test.py --checkpoint checkpoints/best_model.pth \
               --output-dir outputs/classifier_results

# Test detector
python detection_test.py --checkpoint checkpoints/detection_mnv3_hardmining_ft_v2/best_model.pth \
                         --output-dir outputs/detector_results

# Benchmark với Knowledge Graph
python knowledge_graph_benchmark.py \
  --classifier-checkpoint checkpoints/best_model.pth \
  --detector-checkpoint checkpoints/detection_mnv3_hardmining_ft_v2/best_model.pth \
  --knowledge-graph-artifact checkpoints/knowledge_graph_vaipe.json \
  --build-knowledge-graph
```

### Inference (Dự Đoán)

```powershell
# CLI inference
python demo_infer.py --image path/to/pill_image.jpg \
                     --output-dir checkpoints/demo_app_output

# Giao diện Streamlit hiện tại
.\.venv\Scripts\python.exe -m streamlit run app_streamlit_modern.py --server.port 8515
```

Checkpoint mặc định hiện tại:

- Detector: `checkpoints/detection_mnv3_hardmining_ft_v2/best_model.pth`
- Classifier: `checkpoints/best_model.pth`

Ghi chú về luồng xử lý:

- App ưu tiên chạy `detector + classifier` cho ảnh gốc tải lên.
- Nếu detector chỉ tìm thấy `1` viên, giao diện vẫn có thể hiển thị theo luồng detector; số viên thực tế xem ở trường `Số viên` hoặc `num_detections`.
- Với ảnh crop sẵn một viên, bạn vẫn có thể dùng `test.py --image ...` để chạy thuần classifier.

### Output Files

**Sau train classifier**:
- `best_model.pth` - Checkpoint tốt nhất
- `final_model.pth` - Model cuối cùng
- `history.json` - Loss, accuracy, LR history
- `test_metrics.json` - Test set metrics
- `training_curves.png` - Visualization
- `confusion_matrix.png` - Confusion matrix
- `split_manifest.json` - Train/val/test split
- `dataset_summary.json` - Dataset statistics
- `crop_cache_<image_size>/` - Cache crop viên thuốc để train/test nhanh hơn

**Sau train detector**:
- `best_model.pth` - Checkpoint detector tốt nhất
- `detection_split_manifest.json` - Split theo ảnh để đánh giá lại đúng cùng tập test
- `dataset_summary.json` - Thống kê số ảnh, số box, phân bố class

**Khi bootstrap / fine-tune với ảnh thật**:
- `data/user_real_photos/pill/bootstrap_summary.json` - Tóm tắt nhãn nháp đã sinh
- `data/user_real_photos/pill/bootstrap_outputs/` - Artifact xem nhanh kết quả detector + classifier + KG

### Fine-Tune Với Ảnh Thật Của Bạn

Đặt dữ liệu vào đúng cấu trúc:

```text
data/user_real_photos/pill/
├── image/
└── label/
```

Workflow khuyến nghị:

```bash
# 1. Sinh nhãn nháp bằng model hiện tại
scripts\bootstrap_real_photo_labels.bat

# 2. Sửa lại box/label nếu cần, rồi kiểm tra dữ liệu
scripts\validate_real_photo_dataset.bat

# 3. Fine-tune detector với ảnh thật + dữ liệu VAIPE gốc
scripts\train_detector_real_adapt.bat

# 4. Fine-tune classifier với ảnh thật + dữ liệu VAIPE gốc
scripts\train_classifier_real_adapt.bat
```

### Các Cải Tiến Dataset VAIPE Trong Repo Hiện Tại

#### 1. Chuẩn hóa dữ liệu crop cho bài toán single-pill classification

- Từ ảnh gốc `public_train/pill/image` và label JSON, repo tự cắt từng bounding box thành crop riêng để train classifier.
- Crop được cache trong `crop_cache_<image_size>` để giảm thời gian chuẩn bị dữ liệu ở các lần train/test sau.
- Mỗi lần train đều lưu `split_manifest.json` và `dataset_summary.json` để tái lập đúng train/val/test split.

#### 2. Giảm lệch lớp (class imbalance)

- Classifier hỗ trợ `weighted_ce` và `focal loss`.
- Có `class-balanced sampling` qua các tham số như `--sampler-power` và `--class-weight-power`.
- Cách này giúp các nhãn ít mẫu trong dataset VAIPE không bị chìm so với các nhãn phổ biến hơn.

#### 3. Cải thiện detection trên ảnh nhiều viên bằng hard-example replay

- Detector không chỉ train ngẫu nhiên mà còn replay các ảnh khó sau warmup.
- Các tham số chính:
  - `--hard-mining-topk`
  - `--hard-mining-boost`
  - `--hard-mining-warmup`
- Đây là phần cải tiến quan trọng để tăng độ ổn định trên ảnh nhiều viên, ảnh sát nhau hoặc ảnh có điều kiện chụp khó.

#### 4. Hỗ trợ trộn thêm ảnh thật của người dùng vào dataset gốc

- `train.py` và `detection_train.py` đều hỗ trợ `--extra-data-root`.
- Nhờ đó có thể train trên:
  - dữ liệu VAIPE gốc trong `archive (1)/public_train/pill`
  - dữ liệu ảnh thật bổ sung trong `data/user_real_photos/pill`
- Đây là hướng thích nghi domain quan trọng để model đỡ bị lệch khi triển khai với ảnh chụp thực tế.

#### 5. Bổ sung workflow bootstrap nhãn cho ảnh thật

- `bootstrap_real_photo_labels.py` dùng detector + classifier + knowledge graph hiện tại để sinh nhãn JSON nháp.
- `validate_real_photo_dataset.py` kiểm tra lỗi format, box lỗi, ảnh thiếu cặp JSON, và box out-of-bounds trước khi fine-tune.
- Workflow này giúp mở rộng dataset nhanh hơn thay vì gán nhãn hoàn toàn thủ công từ đầu.

#### 6. Enrich ngữ cảnh dữ liệu bằng Knowledge Graph

- Knowledge graph không chỉ dựa vào ảnh crop mà còn tổng hợp:
  - màu sắc
  - hình dạng
  - imprint / texture signature
  - drug name
  - prescription co-occurrence
- Phần ngữ cảnh này giúp rerank tốt hơn ở những cặp nhãn dễ nhầm trong dataset VAIPE.

#### 7. Theo dõi dữ liệu và thí nghiệm rõ hơn

- Repo hiện lưu rõ artifact cho dữ liệu và split thay vì chỉ lưu checkpoint model.
- Các file như `dataset_summary.json`, `split_manifest.json`, `detection_split_manifest.json`, `bootstrap_summary.json` giúp dễ audit và so sánh giữa các lần chạy.

---

## 📊 Kết Quả & Hiệu Năng

### Benchmark Results (Checkpoint Mặc Định Hiện Tại)

#### Classifier mặc định (`checkpoints/best_model.pth`)
| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 96.23% |
| Top-3 Accuracy | 99.36% |
| Macro F1-score | 90.88% |
| Test samples | 3,285 crop |
| Loss | 0.5625 |
| Model Size | ~45 MB |

#### Detector mặc định (`checkpoints/detection_mnv3_hardmining_ft_v2/best_model.pth`)
| Metric | Value |
|--------|-------|
| Precision | 88.94% |
| Recall | 95.39% |
| F1-score | 92.05% |
| Mean matched IoU | 86.35% |
| Test images | 954 ảnh |
| Model Size | ~120 MB |

#### With Knowledge Graph Re-ranking
| Ghi chú | Giá trị |
|--------|---------|
| Trạng thái | Có thể bật trong pipeline detector + classifier |
| Tác động | Mức cải thiện phụ thuộc checkpoint detector/classifier đang ghép |
| Cách đo | Dùng `knowledge_graph_benchmark.py` để đo trên đúng bộ checkpoint bạn đang chạy |

### Hyperparameter Effects

**Sampler Power (class balancing)**:
- 0.0 = không dùng balancing → bias lớn có nhiều mẫu
- 0.5 = sqrt inverse frequency → cân bằng tốt
- 1.0 = full inverse frequency → có thể thiên lệch lại

**Focal Loss Impact**:
- Đặc biệt hiệu quả với các lớp ít mẫu (~5-15%)
- Hard examples được focus hơn
- Trade-off: Dễ overfit nếu γ quá cao

**Color Fusion Contribution**:
- RGB features alone: ~88-90%
- + Color stream: checkpoint mặc định hiện tại đạt Top-1 96.23% và Top-3 99.36%
- Improvement: +2-4% (đặc biệt với màu rõ)

---

## 🔧 Troubleshooting

### Vấn đề: CUDA Out of Memory

```bash
# Giảm batch size
python train.py --batch-size 32

# Hoặc dùng CPU chạy chậm hơn nhưng ổn định
python train.py --device cpu (nếu code support)
```

### Vấn đề: Training không hội tụ

- ✅ Kiểm tra learning rate (thử 1e-3 đến 1e-5)
- ✅ Dùng smaller model nếu data ít
- ✅ Tăng data augmentation
- ✅ Dùng weighted loss cho class imbalance

### Vấn đề: Accuracy thấp

- ✅ Kiểm tra dữ liệu (mislabeled samples?)
- ✅ Tăng model capacity (layers/channels)
- ✅ Tăng epochs & patience
- ✅ Sử dụng Color Fusion variant

---

## 📚 Tài Liệu Tham Khảo

### Papers

1. **ResNet**: He et al. (2015) - *Deep Residual Learning for Image Recognition*
2. **Faster R-CNN**: Ren et al. (2016) - *Faster R-CNN: Towards Real-Time Object Detection*
3. **Focal Loss**: Lin et al. (2017) - *Focal Loss for Dense Object Detection*
4. **CG-IMIF**: Tham khảo color fusion techniques (handcrafted features)
5. **Transfer Learning**: Yosinski et al. (2014)

### Online Resources

- PyTorch Docs: https://pytorch.org/docs/
- Torchvision Models: https://pytorch.org/vision/
- OpenCV Tutorials: https://docs.opencv.org/
- Scikit-learn Metrics: https://scikit-learn.org/stable/modules/model_evaluation/

---

## 🎓 Các Khái Niệm Cần Hiểu

### Confusion Matrix

```
        Predicted
        Class A  Class B  Class C
Actual
Class A   [TP]    [FP]    [FP]
Class B   [FN]    [TP]    [FP]
Class C   [FN]    [FN]    [TP]

True Positive (TP): Dự đoán đúng
False Positive (FP): Dự đoán sai lạc loại khác
False Negative (FN): Bỏ sót (không định danh)
True Negative (TN): Bộ dữ liệu multi-class không dùng
```

### F1-Score

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

Trong đó:
- **Precision** = TP / (TP + FP) - Trong những gì dự đoán, bao nhiêu đúng
- **Recall** = TP / (TP + FN) - Trong những gì thực tế, bao nhiêu được tìm thấy

**Macro F1**: Tính F1 cho mỗi lớp rồi lấy trung bình (công bằng hơn cho imbalanced data)

### Bounding Box Metrics

- **IoU (Intersection over Union)**: Diện tích giao / diện tích hợp
- **mAP**: Mean Average Precision @IoU threshold

```
                 Area of Overlap
IoU = ────────────────────────────────
      Area of Union (without double-counting overlap)
```

---

## 📝 Ghi Chú Thêm

### Optimization Tips

1. **Data Caching**: Crop cache giúp training ~3-5x nhanh hơn
2. **Mixed Precision**: AMP (automatic mixed precision) sử dụng float16 + float32 → 2x tốc độ
3. **DDP** (Distributed Data Parallel): Multi-GPU training (nếu có GPU)

### Model Variants

| Model | Speed | Accuracy | GPU Mem | Use Case |
|-------|-------|----------|---------|----------|
| ResNet18 | Fast | Good | Low | Quick experiments |
| ResNet50 | Moderate | Better | Med | Production |
| MobileNetV3 | Very Fast | Fair | Very Low | Mobile/Edge |
| EfficientNet | Balanced | Better+ | Med | Research |

### Future Improvements

- [ ] Vision Transformer backbone (ViT)
- [ ] Ensemble methods
- [ ] Quantization cho deployment mobile
- [ ] Self-supervised pretraining
- [ ] Active learning cho data acquisition

---

## 📞 Liên Hệ & Support

Dự án này là một ứng dụng học tập về Deep Learning for Medical Image Analysis.

**Tác giả**: [Tên DA]
**Phiên bản**: 2.0
**Cập nhật lần cuối**: Mar 2026

---

## 📄 License

[Thêm license nếu cần]

---

## Feedback & Questions

Nếu có câu hỏi về code, model, hoặc theory, vui lòng:
1. Xem README này trước
2. Kiểm tra code comments
3. Refer to references trên

**Happy Learning! 🚀**
