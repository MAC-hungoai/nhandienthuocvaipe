# 📋 Confusion Matrix Cheatsheet - Kiểm Tra Nhanh

## 🎯 Câu Hỏi 1: Accuracy = 95%, Nó Tốt Không?

Trả lời bằng cách kiểm tra theo thứ tự dưới đây:

```
┌─────────────────────────────────────────────────────────┐
│ BƯỚC 1: Xem Gap Accuracy vs Macro F1                    │
├─────────────────────────────────────────────────────────┤
│ Gap = Accuracy - Macro F1                               │
│ ✓ Gap ≤ 5%   → OK, bước tiếp                             │
│ ⚠️ Gap 5-15% → Cảnh báo nhẹ, xem lớp kém              │
│ ❌ Gap > 15% → CẢNH BÁO, accuracy bị "bóp méo"        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ BƯỚC 2: Xem Recall Tất Cả Lớp                           │
├─────────────────────────────────────────────────────────┤
│ Min_Recall = recall thấp nhất của tất cả lớp             │
│ ✓ Min_Recall ≥ 80% → OK, bước tiếp                      │
│ ⚠️ 50% ≤ Min_Recall < 80% → Cảnh báo, lớp yếu        │
│ ❌ Min_Recall < 50% → CẢNH BÁO, model kém              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ BƯỚC 3: Xem Gap Accuracy vs Balanced Accuracy           │
├─────────────────────────────────────────────────────────┤
│ Balanced_Accuracy = Trung bình recall tất cả lớp        │
│ Gap = Accuracy - Balanced Accuracy                       │
│ ✓ Gap ≤ 5%   → OK, tất cả lớp cân bằng                  │
│ ⚠️ Gap 5-15% → Cảnh báo nhẹ, class imbalance            │
│ ❌ Gap > 15% → CẢNH BÁO, accuracy bị "bóp méo"        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ BƯỚC 4: Xem Confusion Matrix                            │
├─────────────────────────────────────────────────────────┤
│ ✓ Đường chéo chính tương đối bằng → OK                  │
│ ⚠️ Một dòng toàn 0 → Lớp bị mode collapse              │
│ ❌ Một cột chiếm >35% tổng → Mode collapse              │
└─────────────────────────────────────────────────────────┘
```

## 📊 Ví Dụ Thực Tế

### ✅ ACCURACY TIN CẬY:
```
Metrics:
  Accuracy     = 96%
  Macro F1     = 95%
  Gap A-F1     = 1%    ← OK (<5%)
  
Per-class:
  Class 0: recall 95%, support 100
  Class 1: recall 94%, support 95
  Class 2: recall 96%, support 105
  Min recall = 94%     ← OK (≥80%)
  
Balanced Accuracy = 95%
Gap Acc-BalAcc = 1%   ← OK (<5%)

✓ VERDICT: Accuracy 96% TIN CẬY
```

### ⚠️ ACCURACY CẢN CẢNH BÁO:
```
Metrics:
  Accuracy     = 92%
  Macro F1     = 82%
  Gap A-F1     = 10%   ← ⚠️ CẢNH BÁO
  
Per-class:
  Class 0: recall 95%, support 200  (nhiều lời)
  Class 1: recall 45%, support 30   (ít + kém) ← VẤN ĐỀ
  Min recall = 45%                  ← ❌ CẢNH BÁO
  
Balanced Accuracy = 70%
Gap Acc-BalAcc = 22%  ← ❌ CẢNH BÁO (>15%)

Confusion Matrix: 
  Lớp 1 nhầm thành lớp 0 thường xuyên

⚠️ VERDICT: Accuracy 92% BỊ "BÓP MÉO"
Action: Cải thiện lớp 1 (tăng data hoặc adjust loss)
```

### ❌ ACCURACY LÀ RÁC:
```
Metrics:
  Accuracy     = 95%
  Macro F1     = 60%
  Gap A-F1     = 35%   ← ❌ CẢNH BÁO NẶNG
  
Per-class:
  Class 0: recall 99%, support 500  (quá nhiều lớp A)
  Class 1: recall 20%, support 5    (ít + rất kém)
  Min recall = 20%                  ← ❌ CẢNH BÁO NẶNG
  
Balanced Accuracy = 60%
Gap Acc-BalAcc = 35%  ← ❌ CẢNH BÁO NẶNG (>20%)

Confusion Matrix:
  Lớp 1 toàn bộ nhầm thành lớp 0
  
❌ VERDICT: Accuracy 95% LÀ "RÁC"
Model chỉ hoạt động cho lớp 0, quên mất lớp 1 hoàn toàn
Action: Cần retrain từ đầu với loss weighting
```

## 🔍 Cách Đọc Metrics Từ `test_metrics.json`

```json
{
  "accuracy": 0.96,                 ← Tỉ lệ đoán đúng
  "macro_f1": 0.95,                 ← F1 trung bình (không trọng số)
  "confusion_matrix": [[...], ...], ← Ma trận nhầm lẫn
  "per_class_metrics": [
    {
      "label_id": 0,
      "support": 100,               ← Số mẫu thực tế
      "recall": 0.95,               ← Nhạy cảm: 95% gặp class 0 mà nhận ra
      "precision": 0.94,            ← Chính xác: 94% khi dự đoán 0 thì đúng
      "f1": 0.945                   ← Cân bằng recall & precision
    },
    ...
  ]
}
```

## 🚀 Lệnh Chạy

### 1. Chạy tool phân tích tự động:
```bash
python check_model_metrics.py --metrics checkpoints/test_metrics.json
```

**Output:** "LOOKS_CONSISTENT" hoặc "LIKELY_MISLEADING"

### 2. Chạy educational examples:
```bash
python confusion_matrix_guide.py
```

**Output:** 3 ví dụ + 3 hình ảnh confusion matrix

### 3. Xem confusion matrix sau khi train:
```bash
# Sau train hoặc test
open checkpoints/confusion_matrix.png
```

## 📝 Template: Báo Cáo Model Performance

```
📊 MODEL EVALUATION REPORT
==========================

Model: [Tên model]
Dataset: [Tênset]

✅ OVERALL METRICS:
   - Accuracy:           96.23%
   - Macro F1:           90.88%
   - Gap (A - F1):       5.34% (OK/WARNING/ALERT)

📈 PER-CLASS METRICS:
   - Min Recall:         88.2% (OK/WARNING/ALERT)
   - Max Recall:         97.5%
   - Balanced Accuracy:  91.5% (OK/WARNING/ALERT)

🎯 CLASS IMBALANCE:
   - Most common class:  35.2% (OK/WARNING/ALERT)
   - Least common class: 8.3%

📋 VERDICT:
   ✓ LOOKS_CONSISTENT - Accuracy tin cậy
   ⚠️ NEEDS_CONTEXT - Cần xem chi tiết
   ❌ LIKELY_MISLEADING - Accuracy bị "bóp méo"

ACTIONS:
   [ ] Cải thiện class X (recall chỉ 45%)
   [ ] Thêm data cho class Y (support < 10)
   [ ] Adjust loss weighting
```

## 🎓 Quy Tắc Nhanh Nhất

```
┌────────────────────────────────────────────────┐
│ Nếu bạn chỉ có 30 giây để check:               │
├────────────────────────────────────────────────┤
│                                                │
│ 1. Gap A - Macro F1 > 10%? → ⚠️ CẢNH BÁO    │
│                                                │
│ 2. Có lớp recall < 60%? → ⚠️ CẢNH BÁO        │
│                                                │
│ 3. Chạy check_model_metrics.py                │
│    → Nếu LIKELY_MISLEADING → ❌ CẢNH BÁO    │
│                                                │
│ Nếu cả 3 "NO" → ✓ Accuracy tin cậy!           │
└────────────────────────────────────────────────┘
```

## 📚 Liên Hệ

- Xem chi tiết: [CONFUSION_MATRIX_VI.md](CONFUSION_MATRIX_VI.md)
- Educational script: `confusion_matrix_guide.py`
- Auto-check tool: `python check_model_metrics.py`
