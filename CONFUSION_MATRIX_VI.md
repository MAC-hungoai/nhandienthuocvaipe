# Confusion Matrix & Độ Chính Xác: Hướng Dẫn Toàn Diện

## 🎯 Câu Hỏi Chính

> **"Dùng confusion matrix để nhận biết độ chính xác có phải là rác không?"**

**Trả lời:** ĐÚNG! Confusion matrix phát hiện khi Accuracy cao nhưng model không sử dụng được thực tế.

---

## 📚 Phần 1: Tại Sao Accuracy Có Thể Là "Rác"?

### Vấn Đề đầu tiên: Class Imbalance (Dữ Liệu Không Cân Bằng)

**Tình Huống:**
```
Dataset: 100 mẫu thuốc
- Loại A: 99 mẫu (99%)
- Loại B: 1 mẫu (1%)

Model SỰ sai:
Dự đoán: "Tất cả đều là loại A"
```

**Kết Quả:**
- ✅ Accuracy = 99% (rất cao!)
- ❌ Nhưng: Model chưa bao giờ dự đoán B
- ❌ Recall cho B = 0/1 = 0% (không nhận ra được)
- ❌ Thực tế: Model **không sử dụng được** vì thiếu loại B

**Confusion Matrix phát hiện:**
```
         Dự đoán A    Dự đoán B
Thật A:    99            0
Thật B:     1            0     ← Toàn bộ dòng B = 0 = BUG!
```

### Vấn Đề thứ hai: Mode Collapse (Model Quên Mất Các Lớp)

**Tình Huống:**
```
Dataset: 4 loại thuốc
- Loại 1: 20 mẫu
- Loại 2: 15 mẫu
- Loại 3: 10 mẫu
- Loại 4: 5 mẫu
(Tổng: 50 mẫu, cân bằng tương đối)

Model tư duy bị gặp vấn đề:
Dự đoán: Hầu hết là loại 1 (mode collapse)
```

**Kết Quả:**
```
         Dự đoán 1  Dự đoán 2  Dự đoán 3  Dự đoán 4
Thật 1:    18          2          0          0
Thật 2:    13          2          0          0
Thật 3:     9          1          0          0
Thật 4:     5          0          0          0
         ----------- -------ALERT -----------
         Tổng: 45    5           0          0
         → Model quên mất loại 3 & 4 hoàn toàn!
```

Mặc dù accuracy có thể vẫn ~70%, nhưng **model không sử dụng được** vì không phân biệt được 2 loại.

### Vấn Đề thứ ba: Per-Class Performance Imbalance

**Tình Huống:**
```
3 loại thuốc, dữ liệu cân bằng, nhưng:
- Loại A: Dễ phân loại → Recall 95%
- Loại B: Trung bình → Recall 75%
- Loại C: Khó phân loại → Recall 40%   ← VẤN ĐỀ!

Accuracy = 70% (= trung bình 95%, 75%, 40%)
Nhưng: Nếu bạn gặp loại C, model chỉ nhận ra 40%
```

---

## 🎲 Phần 2: Confusion Matrix Là Gì?

### Định Nghĩa

**Confusion Matrix** là bảng thống kê cho từng lớp (class):
```
         Dự đoán Lớp 1   Dự đoán Lớp 2   Dự đoán Lớp 3
Thật 1:       TP₁             FN₁             FN₁
Thật 2:       FP₂             TP₂             FN₂
Thật 3:       FP₃             FP₃             TP₃

Ký hiệu:
- TP (True Positive): Đoán đúng
- FP (False Positive): Đoán sai (tưởng là lớp này nhưng không)
- FN (False Negative): Đoán sai (là lớp này nhưng không nhận ra)
- TN (True Negative): Không là lớp này, đoán đúng không phải (hiếm dùng trong multi-class)
```

### Ví Dụ Thực Tế

```
3 loại thuốc: A, B, C

Dữ liệu test thật:
A A A B B C C C C C

Dự đoán mô hình:
A A B B B C C C A A

Confusion Matrix:
         Dự đoán A   Dự đoán B   Dự đoán C
Thật A:      2           1           0      (Thực A = 3, đoán đúng 2)
Thật B:      0           2           0      (Thực B = 2, đoán đúng 2)
Thật C:      2           0           3      (Thực C = 5, đoán đúng 3)
```

---

## 📊 Phần 3: Metrics Từ Confusion Matrix

### 1. **Recall (Sensitivity)** - "Nhạy cảm"
```
Recall = TP / (TP + FN)
= "Khi gặp lớp này, model nhận ra được bao nhiêu %?"

Ví dụ:
- Thật C = 5 mẫu
- Đoán đúng C = 3 mẫu
- Recall_C = 3 / 5 = 60%
  → Khi gặp loại C, model chỉ nhận ra 60%
```

### 2. **Precision (Chính Xác Dự Đoán)**
```
Precision = TP / (TP + FP)
= "Khi model dự đoán lớp này, nó đúng bao nhiêu %?"

Ví dụ:
- Dự đoán C = 3 mẫu
- Đúng C = 3 mẫu
- Precision_C = 3 / 3 = 100%
  → Khi model dự đoán C, nó luôn đúng
```

### 3. **F1-Score** - "Cân Bằng Recall & Precision"
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
= Điểm cân bằng giữa nhạy cảm (recall) và chính xác (precision)

Ví dụ:
- Precision_C = 100%
- Recall_C = 60%
- F1_C = 2 × (1.0 × 0.6) / (1.0 + 0.6) = 75%
```

### 4. **Balanced Accuracy** - "Độ Chính Xác Cân Bằng"
```
Balanced Accuracy = (Recall₁ + Recall₂+ ... + Recall_n) / n
= Trung bình recall tất cả các lớp

Vì sao quan trọng?
- Nếu data không cân bằng, accuracy bị "bóp méo"
- Balanced accuracy không chịu ảnh hưởng của class imbalance
```

### 5. **Macro F1** - "F1 Không Trọng Số"
```
Macro F1 = (F1₁ + F1₂ + ... + F1_n) / n
= Trung bình F1 tất cả các lớp

Vô dụng F1-score bị "bóp méo" bởi class imbalance
```

---

## 🔍 Phần 4: Cách Đọc Confusion Matrix

### Bước 1: Xem Đường Chéo Chính
```
Confusion Matrix:
     Dự A   Dự B   Dự C
A:   [2    0     0]  ← Đường chéo chính = dự đoán đúng
B:   [0    2     0]
C:   [0    0     3]

✓ Đường chéo chính lớn → Model tốt
✗ Đường chéo chính nhỏ → Model kém
```

### Bước 2: Xem Theo Dòng (True Label) = Recall
```
Dòng A: [2, 0, 0] → Total = 3, TP = 2
Recall_A = 2/3 = 67%
→ Loại A nhận ra được 67%

Dòng B: [0, 2, 0] → Total = 2, TP = 2
Recall_B = 2/2 = 100%
→ Loại B nhận ra được 100%

Dòng C: [0, 0, 3] → Total = 3, TP = 3
Recall_C = 3/3 = 100%
→ Loại C nhận ra được 100%
```

### Bước 3: Xem Theo Cột (Predicted Label) = Precision
```
Cột A: [2, 0, 0] → Total = 2, TP = 2
Precision_A = 2/2 = 100%
→ Khi dự pred A, nó đúng 100%

Cột B: [0, 2, 0] → Total = 2, TP = 2
Precision_B = 2/2 = 100%

Cột C: [0, 0, 3] → Total = 3, TP = 3
Precision_C = 3/3 = 100%
```

### Bước 4: Phát Hiện Lỗi Nhầm
```
          Dự A   Dự B   Dự C
A: [     2      1      0   ]  ← A nhầm thành B 1 lần
B: [     0      5      2   ]  ← B nhầm thành C 2 lần
C: [     1      1      8   ]  ← C nhầm thành A, B mỗi 1 lần

Lỗi nhầm phổ biến: B → C (2 lần)
Action: Tăng dữ liệu phân biệt B vs C
```

---

## 📋 Phần 5: Danh Sách Kiểm Tra - Khi Nào Accuracy Là "Rác"?

### ✅ Accuracy TIN CẬY khi:
```
☑ ☑ ☑ Data cân bằng: Mỗi lớp 20-25% dữ liệu
☑ ☑ ☑ Recall cao: Tất cả lớp ≥ 80%
☑ ☑ ☑ Balanced Accuracy ≈ Accuracy (gap < 5%)
☑ ☑ ☑ Macro F1 ≈ Accuracy (gap < 5%)
☑ ☑ ☑ Không có class nào > 35% tổng dự đoán
```

### ❌ Accuracy CÓ THỂ LÀ "RÁC" khi:
```
☒ ☒ ☒ Data không cân bằng: Một lớp > 70% dữ liệu
☒ ☒ ☒ Recall kém: Một số lớp < 50%
☒ ☒ ☒ Gap lớn: |Accuracy - Balanced Accuracy| > 15%
☒ ☒ ☒ Gap lớn: |Accuracy - Macro F1| > 10%
☒ ☒ ☒ Mode collapse: Một lớp = > 35% tất cả dự đoán
```

---

## 🛠️ Phần 6: Công Cụ Sẵn Có Trong Project

### 1. Script Phân Tích Tự Động
```bash
python check_model_metrics.py --metrics checkpoints/test_metrics.json
```

**Input:** File `test_metrics.json` từ training
**Output:**
- LOOKS_CONSISTENT → Accuracy tin cậy ✓
- NEEDS_CONTEXT → Cần xem chi tiết ⚠️
- LIKELY_MISLEADING → Accuracy là rác ❌

### 2. Confusion Matrix Visualization
```python
python train.py  # hoặc test.py
# → Tạo confusion_matrix.png
```

### 3. Educational Script
```bash
python confusion_matrix_guide.py
```

**Output:**
- 3 ví dụ cụ thể về vấn đề
- 3 confusion matrix visualization
- Danh sách kiểm tra chi tiết

---

## 📖 Phần 7: Ví Dụ Thực Tế Từ Project

### Metric từ `checkpoints/test_metrics.json`

```json
{
  "accuracy": 0.92,
  "macro_f1": 0.85,
  "confusion_matrix": [[...], [...], ...],
  "per_class_metrics": [
    {
      "label_id": 0,
      "support": 150,
      "recall": 0.95,
      "precision": 0.93,
      "f1": 0.94
    },
    {
      "label_id": 1,
      "support": 50,
      "recall": 0.60,
      "precision": 0.75,
      "f1": 0.67
    }
  ]
}
```

### Phân Tích:
```
Accuracy = 92%
Macro F1 = 85%
Gap = 7%  ← Cảnh báo nhẹ

Per-class analysis:
- Label 0: recall 95%, support 150 (data nhiều + model tốt)
- Label 1: recall 60%, support 50  (data ít + model kém) ← CẢNH BÁO!

Verdict: Accuracy bị "bóp méo" bởi class imbalance + label 1 kém
Action: Cần cải thiện label 1
```

---

## 🎓 Phần 8: Hành Động Cải Thiện

### Nếu Accuracy Là "Rác":

#### 1. **Class Imbalance**
```
Problem: Loại A = 80%, Loại B = 20%
Solution:
- Use WeightedRandomSampler
- Increase learning rate for minority class
- Use SMOTE (Synthetic Minority Over-sampling)
- Collect more minority class data
```

#### 2. **Mode Collapse**
```
Problem: Model quên lớp C
Solution:
- Increase C weight in loss function
- Use Focal Loss instead of Cross Entropy
- Add hard-mining to focus on difficult samples
- Increase training epochs with early stopping on balanced accuracy
```

#### 3. **Low Recall Cho Một Lớp**
```
Problem: Loại C chỉ recall 40%
Solution:
- Analyze confusion matrix: C → ? (nhầm thành loại gì?)
- Collect more C data
- Add data augmentation
- Fine-tune hyperparameters
```

---

## 🔗 Tài Liệu Liên Quan

| Tài Liệu | Mục Đích |
|---------|---------|
| README.md | Overview project |
| README_VI.md | Hướng dẫn tiếng Việt |
| IMPROVEMENT_ROADMAP.md | Kế hoạch cải thiện chi tiết |
| check_model_metrics.py | Phân tích tự động |
| confusion_matrix_guide.py | Educational examples |

---

## 💡 Tóm Tắt Nhanh

```
┌─────────────────────────────────────────────────────────┐
│ Confusion Matrix giúp bạn:                              │
├─────────────────────────────────────────────────────────┤
│ 1. Phát hiện khi Accuracy cao nhưng model là rác        │
│ 2. Tìm ra lớp nào có vấn đề                             │
│ 3. Biết lỗi nhầm lẫn phổ biến giữa các lớp              │
│ 4. Quyết định hành động cải thiện                       │
│ 5. Theo dõi tiến độ giữa các lần training                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Quick Check: Accuracy có tin cậy không?                 │
├─────────────────────────────────────────────────────────┤
│ python check_model_metrics.py                           │
│ → Nếu "LOOKS_CONSISTENT" ✓                              │
│ → Nếu "LIKELY_MISLEADING" ❌ cần cải thiện             │
└─────────────────────────────────────────────────────────┘
```

---

## 📞 Liên Hệ & Feedback

Nếu cần chi tiết hơn, xem:
- `confusion_matrix_guide.py` - Chạy examples
- `docs/REAL_PHOTO_FINETUNE_VI.md` - Fine-tuning guide
- Code trong `train.py`, `test.py` - Chi tiết implementation
