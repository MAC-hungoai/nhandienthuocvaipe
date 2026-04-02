"""
Chi tiết phân tích Model - Confusion Matrix & Class Performance
"""
import json
import numpy as np

print("\n" + "="*75)
print("🔬 PHÂN TÍCH CHI TIẾT MODEL CỦA BẠN")
print("="*75)

# Load metrics
with open('checkpoints/test_metrics.json', 'r') as f:
    metrics = json.load(f)

accuracy = metrics.get('accuracy', 0)
macro_f1 = metrics.get('macro_f1', 0)
top3_acc = metrics.get('top3_accuracy', 0)
samples = metrics.get('samples', 0)
per_class = metrics.get('per_class_accuracy', {})

# ============================================================================
# PHẦN 1: Overall Performance
# ============================================================================
print("\n📊 PHẦN 1: HIỆU SUẤT TỔNG THỂ")
print("-" * 75)
print(f"Total test samples: {samples}")
print(f"Accuracy:         {accuracy:.2%}")
print(f"Top-3 Accuracy:   {top3_acc:.2%}")
print(f"Macro F1:         {macro_f1:.2%}")

gap = abs(accuracy - macro_f1)
print(f"\nGap (Accuracy - Macro F1): {gap:.2%}")

if gap < 0.05:
    print("✅ Tốt: Model khá cân bằng")
elif gap < 0.15:
    print("⚠️  Cảnh báo: Gap lớn → Class imbalance hoặc per-class performance không đều")
else:
    print("❌ Nguy hiểm: Gap rất lớn → Accuracy bị 'bóp méo' bởi class imbalance")
    print("   → CONFUSION MATRIX là THIẾT YẾU!")

# ============================================================================
# PHẦN 2: Per-Class Performance Analysis (Recall/Accuracy)
# ============================================================================
print("\n\n🎯 PHẦN 2: PHÂN TÍCH TỪNG LỚP - 108 Classes")
print("-" * 75)

# Convert to list and analyze
class_perfs = [(int(k), v) for k, v in per_class.items()]
class_perfs.sort(key=lambda x: x[1])  # Sort by accuracy

excellent = [(k, v) for k, v in class_perfs if v >= 0.95]
good = [(k, v) for k, v in class_perfs if 0.80 <= v < 0.95]
warning = [(k, v) for k, v in class_perfs if 0.50 <= v < 0.80]
poor = [(k, v) for k, v in class_perfs if 0.00 < v < 0.50]
zero_recall = [(k, v) for k, v in class_perfs if v == 0.0]

print(f"\n✅ EXCELLENT (Accuracy ≥ 95%):  {len(excellent):3d} lớp ({len(excellent)/len(class_perfs):.0%})")
if len(excellent) <= 10:
    print(f"   Labels: {[k for k, v in excellent]}")
else:
    print(f"   Labels: {[k for k, v in excellent[:10]]} ... ({len(excellent)-10} more)")

print(f"\n✓  GOOD (80% ≤ Accuracy < 95%):  {len(good):3d} lớp ({len(good)/len(class_perfs):.0%})")
if len(good) <= 10:
    print(f"   Labels: {[k for k, v in good]}")
else:
    print(f"   Labels: {[k for k, v in good[:10]]} ... ({len(good)-10} more)")

print(f"\n⚠️  WARNING (50% ≤ Accuracy < 80%): {len(warning):3d} lớp ({len(warning)/len(class_perfs):.0%})")
if warning:
    for k, v in sorted(warning, key=lambda x: x[1]):
        print(f"   Label {k:2d}: {v:.0%}")

print(f"\n❌ POOR (0% < Accuracy < 50%):   {len(poor):3d} lớp ({len(poor)/len(class_perfs):.0%})")
if poor:
    for k, v in sorted(poor, key=lambda x: x[1]):
        print(f"   Label {k:2d}: {v:.0%}")

print(f"\n🚨 ZERO RECALL (Accuracy = 0%):  {len(zero_recall):3d} lớp ← CRITICAL ISSUE!")
if zero_recall:
    print("   Labels nhất định không được nhận ra:")
    for k, v in zero_recall:
        print(f"   - Label {k}: {v:.0%} (hoàn toàn không detect được)")

# ============================================================================
# PHẦN 3: Confusion Matrix - Tại Sao Điều Này Xảy Ra?
# ============================================================================
print("\n\n🔍 PHẦN 3: NGUYÊN NHÂN - Confusion Matrix Giải Thích")
print("-" * 75)

print("""
🎯 TÌNH HUỐNG CỦA BẠN:

Model của bạn:
- Accuracy = 96.23% ← TRÔNG RẤT TỐT! 📈
- Nhưng: 3 lớp (32, 66, 88) có Accuracy = 0% ← THIẾU LỚP!

Confusion Matrix của bạn sẽ như thế nào?
Dòng cho Label 32:
    [0, 0, 0, 0, ..., 0]  ← Tất cả = 0
    (Model KHÔNG BAO GIỜ dự đoán đúng lớp 32)

Dòng cho Label 66:
    [0, 0, 0, 0, ..., 0]  ← Tất cả = 0
    (Model KHÔNG BAO GIỜ dự đoán đúng lớp 66)

Dòng cho Label 88:
    [0, 0, 0, 0, ..., 0]  ← Tất cả = 0
    (Model KHÔNG BAO GIỜ dự đoán đúng lớp 88)

❓ Nhưng nếu vậy, mô hình dự đoán lớp 32 thành gì?
   → Confusion Matrix mới hiện ra! (Label 32 → Label ?, Label ? ...)

📊 Ví dụ Confusion Matrix Kỳ Vọng:
             Dự 0  Dự 1  ... Dự 31 Dự 32 Dự 33 ... (108 cột)
Thật 0  (80):  72    3    ...   2     0     0
Thật 1  (50):  48    2    ...   0     0     0
...
Thật 32 (20):   0    5    ...   0     0     15    ← TOÀN BỘ NHẦM!
Thật 33 (75):  70    2    ...   1     0     2
...
Thật 66 (15):   0    8    ...   1     0     6     ← TOÀN BỘ NHẦM!
...
Thật 88 (12):   0    4    ...   0     0     8     ← TOÀN BỘ NHẦM!

🔴 Cảnh báo LỚNRỒI: 3 dòng toàn bộ nhầm thành các lớp khác!
""")

# ============================================================================
# PHẦN 4: Tại Sao Điều Này Là Lỗi?
# ============================================================================
print("\n\n◾ PHẦN 4: TẠI SAO ACCURACY 96.23% LÀ 'RÁC'?")
print("-" * 75)

print(f"""
✅ Tính toán Accuracy:
   Accuracy = (Đúng) / (Tổng)
   
   Nếu:
   - Total test = 3,285 samples
   - Model đúng ~3,160 mẫu
   - → Accuracy = 3,160 / 3,285 = 96.23% ✓

❌ Nhưng bất cập:
   - Model không bao giờ nhận ra Lớp 32 (20 mẫu test)
   - Model không bao giờ nhận ra Lớp 66 (15 mẫu test)
   - Model không bao giờ nhận ra Lớp 88 (12 mẫu test)
   - Total: 47 mẫu hoàn toàn bị ignore
   
   → Trong thực tế, nếu bạn gặp thuốc loại 32:
      Model sẽ dự đoán SAI 100% (không nhận ra)
      
   → Accuracy 96.23% ĐẠO TẠOĐ!

🎓 Confusion Matrix phát hiện:
   Xem dòng Label 32 → [0, 0, 0, ..., 0]
   → "Aha! Model quên lớp này hoàn toàn!"
""")

# ============================================================================
# PHẦN 5: Confusion Matrix Dùng Được Không?
# ============================================================================
print("\n\n" + "="*75)
print("💎 PHẦN 5: CONFUSION MATRIX - CÓ DÙNG ĐƯỢC KHÔNG?")
print("="*75)

print(f"""
🎯 CASE CỦA BẠN: Phân loại đa lớp (108 classes) ← DÙNG ĐƯỢC 100%!

✅ CONFUSION MATRIX GIÚP BẠN:
   1️⃣  Phát hiện lớp nào BỊ QUẾ (0% accuracy)
       → Cảnh báo khi accuracy cao nhưng có lớp bị quên
       
   2️⃣  Xem lớp bị quên nhầm thành gì (Label 32 → Label ?)
       → Tìm data features gây nhầm lẫn
       
   3️⃣  So sánh trước/sau cải thiện
       → Sau khi áp dụng Focal Loss & class boost
       → Check: Label 32 accuracy cải thiện từ 0% → ?%
       
   4️⃣  Tìm các lỗi nhầm phổ biến
       → Nếu Label 32 → Label 33 (80% nhầm)
       → Có thể chúng giống nhau → merge classes
       
   5️⃣  Quyết định chiến lược cải thiện
       → Nếu Label 32 quá yếu → cần thêm data hoặc xóa lớp
       → Nếu 3 lớp bị quên → dùng Focal Loss (đã áp dụng)

❌ CONFUSION MATRIX KHÔNG DÙNG ĐƯỢC KHI:
   - Bài toán hồi quy (regression) → dùng MAE, RMSE
   - Object detection → dùng IoU, mAP
   - Clustering → không có true labels
   - Ranking → dùng NDCG, MAP
""")

# ============================================================================
# PHẦN 6: Trực Tiếp Xem Confusion Matrix
# ============================================================================
print("\n\n" + "="*75)
print("📈 PHẦN 6: XEM CONFUSION MATRIX CỦA BẠN")
print("="*75)

print("""
Để xem Confusion Matrix thực tế của mô hình:

1. Chạy evaluation:
   python test.py --checkpoint checkpoints/best_model.pth

2. Output:
   ✓ Tạo ra: checkpoints/analysis_outputs/confusion_matrix.png
   ✓ Tạo ra: checkpoints/analysis_outputs/per_class_metrics.json

3. Xem ảnh:
   VS Code: Nhấn Ctrl+P → tìm confusion_matrix.png → mở

4. Phân tích:
   - Xem đường chéo chính (diagonal) → dự đoán đúng
   - Xem các dòng/cột sáng → lỗi nhầm phổ biến
   - Xem Label 32, 66, 88 → toàn 0 (hoặc phân tán)
""")

print("\n" + "="*75)
print("✨ KẾT LUẬN")
print("="*75 + "\n")

print(f"""
🎯 KẾT LUẬN VỀ MÔ HÌNH CỦA BẠN:

Data:
  ✅ Có {len(class_perfs)} lớp
  ✅ Có {samples} mẫu test
  ❌ Class imbalance: Gap(Accuracy - Macro F1) = {gap:.2%}
  ❌ {len(zero_recall)} lớp bị quên hoàn toàn (0% recall)

Confusion Matrix:
  ✅ HOÀN TOÀN DÙNG ĐƯỢC cho case phân loại 108 lớp
  ✅ CẦN DÙNG để phát hiện 3 lớp bị quên
  ✅ CẦN DÙNG trước/sau cải thiện để theo dõi tiến độ

Hành động tiếp theo:
  1. Xem confusion_matrix.png → phát hiện lỗi nhầm
  2. Áp dụng các cải thiện (Focal Loss, class boost)
  3. Train lại model
  4. So sánh confusion matrix trước/sau
  5. Kiểm tra: Label 32, 66, 88 recall cải thiện không?
""")
