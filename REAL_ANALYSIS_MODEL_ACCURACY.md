## ⚠️ PHÂN TÍCH THỰC TẾ: Model Của Bạn - Accuracy Là "RÁC"?

### 📊 Tình Huống Hiện Tại

```
Model:  resnet18 (108-class classifier)
Accuracy: 96.23% (rất cao!)
```

### 🚨 NHẬN XÉT QUAN TRỌNG:

**Tool phân tích kết luận: `LIKELY_MISLEADING` ❌**

Điều này có nghĩa **Accuracy 96.23% là "rác"** - khôngpředstavuje hiệu suất thực sự của model!

---

## 🔍 Chi Tiết Vấn Đề

### Vấn Đề 1: Zero Recall Classes (Model Quên Mất Các Lớp)

```
❌ 3 lớp có recall = 0%:
   - Label 32: Recall 0%  - Model KHÔNG BHẠ nhận ra lớp này
   - Label 66: Recall 0%  - Model KHÔNG BHẠ nhận ra lớp này
   - Label 88: Recall 0%  - Model KHÔNG BHẠ nhận ra lớp này

Khi bạn gặp một mẫu thuốc loại 32, model không nhận ra,
mà dự đoán là lớp gì đó khác!
```

### Vấn Đề 2: Low Recall Classes (Nhận Ra Kém)

```
⚠️ 4 lớp có recall < 50%:
   - Label 32: 0%   (0/X mẫu được nhận ra)
   - Label 66: 0%   (0/X mẫu được nhận ra)
   - Label 88: 0%   (0/X mẫu được nhận ra)
   - Label 78: 40%  (chỉ nhận ra 2/5 hoặc 4/10 mẫu)

Tỷ lệ không tốt từ yêu cầu thực tế (cần ≥ 80%)
```

### Vấn Đề 3: Confusion Matrix Cho Thấy

```
Accuracy = 96.23%

Nhưng thật ra:
- 108 lớp
- 3 lớp model không tìm ra (recall 0%)
- 4 lớp model gặp vấn đề (recall < 50%)

Kết quả: Mô hình chỉ tốt cho ~100 lớp, các lớp khác "tệ"

Vì sao accuracy vẫn 96%?
→ Vì 100 lớp "tốt" chiếm phần lớn dữ liệu test,
  nên làm "che phủ" vấn đề của 3-4 lớp kém.
```

---

## 📈 Metrics Cụ Thể

```
Accuracy:              96.23%  ← Cao nhưng bị "bóp méo"
Macro F1:              90.88%  ← Thấp hơn 5.35% so với accuracy
Balanced Accuracy:     90.94%  ← Gần bằng macro F1
Gap (A - F1):          5.35%   ← OK nhưng có dấu hiệu
Gap (Acc - BalAcc):    5.29%   ← Không lớn lắm nhưng...

Total samples: 3,285
Classes: 108

Zero-recall classes: 3
Low-recall classes: 4 (recall < 50%)
```

---

## ❌ VẤN ĐỀ CHÍNH

### Diagram Vấn Đề:

```
┌─────────────────────────────────────────┐
│ Dataset: 3,285 mẫu, 108 lớp             │
├─────────────────────────────────────────┤
│ ✓ ~100 lớp: Model tốt (recall 80-100%)  │  ← "Che phủ" vấn đề
│ ⚠️ 4 lớp:    Model kém (recall < 50%)    │
│ ❌ 3 lớp:    Model quên (recall 0%)     │  ← VẤNĐỀ!
├─────────────────────────────────────────┤
│ Accuracy = 96% (vì ~100 lớp tốt)        │
│ Nhưng: "Quên" 3 lớp hoàn toàn          │
└─────────────────────────────────────────┘
```

### Ý Nghĩa Thực Tế:

```
Nếu bạn đưa hình ảnh thuốc loại 32 cho model:

❌ Model không nhận ra (hoặc nhầm thành lớp khác)
❌ Người dùng nhận thông tin SAI
❌ Ứng dụng không sử dụng được cho các loại thuốc 32, 66, 88

Accuracy 96% không phản ánh vấn đề này!
```

---

## 🎯 Hành Động Cải Thiện

### Bước 1: Xác Định Vấn Đề
```bash
# Xem chi tiết confusion matrix
python test.py --checkpoint checkpoints/best_model.pth

# Xem lớp nào có vấn đề
python check_model_metrics.py --metrics checkpoints/test_metrics.json
```

### Bước 2: Phân Tích Dữ Liệu

```python
# Kiểm tra:
# - Label 32, 66, 88 có bao nhiêu mẫu train/val?
# - Chúng có đặc điểm gì đặc biệt?
# - Dữ liệu có mất mát không?

import json
with open("checkpoints/test_metrics.json") as f:
    metrics = json.load(f)

per_class = metrics["per_class_metrics"]
problem_classes = [32, 66, 88, 78]

for item in per_class:
    if item["label_id"] in problem_classes:
        print(f"Label {item['label_id']}: "
              f"support={item['support']}, "
              f"recall={item['recall']:.1%}")
```

### Bước 3: Xử Lý

**Nếu lớp 32, 66, 88 có ít dữ liệu:**
```
1. Thu thập thêm dữ liệu cho các lớp này
2. Dùng data augmentation (rotation, blur, crop, etc.)
3. Tăng class weight trong loss function:
   
   class_weights = [1, 1, ..., 5, ..., 5, ..., 5]  # Label 32, 66, 88 = 5x
   criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

**Nếu lớp 32, 66, 88 dữ liệu nhiều nhưng model vẫn chưa nhận ra:**
```
1. Tăng training epochs
2. Thay loss function thành Focal Loss
3. Fine-tune hyperparameters:
   - Learning rate
   - Batch size
   - Weight decay
4. Dùng hard negative mining
```

---

## 📋 Checklist Tối Ưu Hóa

```
[ ] Thu thập dữ liệu cho label 32, 66, 88
[ ] Kiểm tra dữ liệu này có vấn đề không (corruption, mislabel, etc.)
[ ] Tăng class weight cho các lớp này
[ ] Retrain model
[ ] Kiểm tra lại metrics:
    
    ✓ Zero-recall classes = 0 (không còn class recall=0)
    ✓ Low-recall classes < 2 (tối đa 1-2 lớp recall<50%)
    ✓ Accuracy gap < 10% so với Macro F1
    ✓ Verdict = "LOOKS_CONSISTENT"
```

---

## 🔗 Hướng Dẫn Chi Tiết

| Bước | Tài Liệu | Mục Đích |
|------|---------|----------|
| 1. Hiểu vấn đề | [CONFUSION_MATRIX_VI.md](CONFUSION_MATRIX_VI.md) | Lý thuyết confusion matrix |
| 2. Phân tích dữ liệu | `validate_real_photo_dataset.py` | Check dữ liệu có vấn đề |
| 3. Cải thiện model | [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) | Kế hoạch optimize |
| 4. Kiểm tra kết quả | `python check_model_metrics.py` | Xác nhận cải thiện |

---

## 💡 Tóm Tắt

```
┌─────────────────────────────────────────────┐
│ TÌNH HUỐNG HIỆN TẠI                         │
├─────────────────────────────────────────────┤
│ Accuracy:           96.23% (HIGH nhưng...)  │
│ Thực tế:            3 lớp recall 0%    ❌  │
│                     4 lớp recall <50%  ⚠️  │
│                                             │
│ VERDICT:            ACCURACY LÀ "RÁC"  ❌  │
│                     Cần cải thiện            │
└─────────────────────────────────────────────┘

HÀNH ĐỘNG NGAY:
1. Xem dữ liệu của label 32, 66, 88
2. Thu thập / tăng data cường độ
3. Tăng class weight
4. Retrain model
5. Kiểm tra metrics lại
```

---

## 📞 Tiếp Theo

1. **Muốn hiểu confusion matrix chi tiết?**
   → Xem `confusion_matrix_guide.py` hoặc `CONFUSION_MATRIX_VI.md`

2. **Muốn optimize mô hình?**
   → Xem `IMPROVEMENT_ROADMAP.md`

3. **Muốn xem hình ảnh confusion matrix?**
   → Chạy: `python test.py` → Xem `checkpoints/confusion_matrix.png`

4. **Muốn tự phân tích metrics?**
   → Chạy: `python check_model_metrics.py --metrics checkpoints/test_metrics.json`
