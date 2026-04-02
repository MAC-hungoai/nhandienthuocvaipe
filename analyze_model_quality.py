"""
Comprehensive Analysis: Model Performance & Data Quality
Kiểm tra: Dữ liệu và mô hình có vấn đề gì không?
"""
import json
import os
from pathlib import Path

print("\n" + "="*70)
print("📊 PHÂN TÍCH MODEL CỦA BẠN - Model Performance & Data Analysis")
print("="*70)

# ============================================================================
# PHẦN 1: Load Test Metrics
# ============================================================================
print("\n📋 PHẦN 1: TẢI METRICS TỪ CÁC FILE")
print("-" * 70)

metrics_files = {
    "test_metrics.json": "checkpoints/test_metrics.json",
    "evaluation_metrics.json": "checkpoints/analysis_outputs/evaluation_metrics.json",
    "history.json": "checkpoints/history.json",
}

data = {}
for name, path in metrics_files.items():
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data[name] = json.load(f)
            print(f"✅ {name:30} → {len(str(data[name]))} bytes")
        except Exception as e:
            print(f"❌ {name:30} → Lỗi: {e}")
    else:
        print(f"⚠️  {name:30} → Không tìm thấy")

if not data:
    print("\n❌ Không tìm thấy metrics files!")
    exit(1)

# ============================================================================
# PHẦN 2: Phân Tích Accuracy vs Balanced Accuracy (Class Imbalance)
# ============================================================================
print("\n\n🎯 PHẦN 2: PHÁT HIỆN CLASS IMBALANCE")
print("-" * 70)

if "test_metrics.json" in data:
    metrics = data["test_metrics.json"]
    
    accuracy = metrics.get("accuracy", 0)
    macro_f1 = metrics.get("macro_f1", 0)
    
    # Tính balanced accuracy từ per-class metrics
    if "per_class_metrics" in metrics:
        recalls = [m.get("recall", 0) for m in metrics["per_class_metrics"]]
        balanced_acc = sum(recalls) / len(recalls) if recalls else 0
        
        gap = abs(accuracy - balanced_acc)
        
        print(f"Accuracy:           {accuracy:.2%}")
        print(f"Balanced Accuracy:  {balanced_acc:.2%}")
        print(f"Gap:                {gap:.2%}")
        print(f"Macro F1:           {macro_f1:.2%}")
        print(f"F1 Gap:             {abs(accuracy - macro_f1):.2%}")
        
        print("\n" + "─" * 70)
        if gap < 0.05:
            print("✅ TỐTR: Class cân bằng, Accuracy tin cậy")
        elif gap < 0.15:
            print("⚠️  CẢNH BÁO: Class hơi không cân bằng")
            print("   → Balanced Accuracy thấp hơn Accuracy")
        else:
            print("❌ NGUY HIỂM: Class BẤT CÂN BẰNG rất lớn!")
            print("   → Accuracy có thể là 'rác'")

# ============================================================================
# PHẦN 3: Phân Tích Per-Class Performance (Recall)
# ============================================================================
print("\n\n📊 PHẦN 3: PHÂN TÍCH TỪNG LỚP - Recall, Precision, F1")
print("-" * 70)

if "test_metrics.json" in data:
    metrics = data["test_metrics.json"]
    
    if "per_class_metrics" in metrics:
        per_class = metrics["per_class_metrics"]
        
        # Phân loại lớp theo recall
        excellent = []  # recall >= 0.85
        good = []       # recall >= 0.70
        warning = []    # recall >= 0.50
        poor = []       # recall < 0.50
        
        for m in per_class:
            label = m.get("label_id", "?")
            recall = m.get("recall", 0)
            
            if recall >= 0.85:
                excellent.append((label, recall))
            elif recall >= 0.70:
                good.append((label, recall))
            elif recall >= 0.50:
                warning.append((label, recall))
            else:
                poor.append((label, recall))
        
        print(f"\n✅ EXCELLENT (Recall ≥ 85%):  {len(excellent)} lớp")
        print(f"   {[f'L{l}:{r:.0%}' for l, r in excellent[:10]]}")
        
        print(f"\n✓  GOOD (Recall ≥ 70%):        {len(good)} lớp")
        print(f"   {[f'L{l}:{r:.0%}' for l, r in good[:10]]}")
        
        print(f"\n⚠️  WARNING (Recall ≥ 50%):   {len(warning)} lớp")
        if warning:
            print(f"   {[f'L{l}:{r:.0%}' for l, r in warning[:10]]}")
        
        print(f"\n❌ POOR (Recall < 50%):       {len(poor)} lớp")
        if poor:
            print(f"   {[f'L{l}:{r:.0%}' for l, r in poor[:10]]}")
            print(f"   → CẦN CẢI THIỆN!")
        
        # Tìm lớp có recall = 0
        zero_recall = [m for m in per_class if m.get("recall", 0) == 0]
        if zero_recall:
            print(f"\n🚨 CẢNH BÁO NGHIÊM TRỌNG: {len(zero_recall)} lớp có Recall = 0%")
            print("   Các lớp này hoàn toàn không được nhận ra!")
            for m in zero_recall[:5]:
                print(f"   - Label {m['label_id']}: {m.get('support', 0)} mẫu")

# ============================================================================
# PHẦN 4: Phân Tích Data Distribution (Support)
# ============================================================================
print("\n\n📈 PHẦN 4: PHÂN TÍCH DATA DISTRIBUTION - Class Imbalance")
print("-" * 70)

if "test_metrics.json" in data:
    metrics = data["test_metrics.json"]
    
    if "per_class_metrics" in metrics:
        per_class = metrics["per_class_metrics"]
        
        supports = [m.get("support", 0) for m in per_class]
        total_samples = sum(supports)
        
        max_support = max(supports)
        min_support = min(supports)
        avg_support = total_samples / len(supports)
        
        imbalance_ratio = max_support / min_support if min_support > 0 else float('inf')
        
        print(f"Tổng mẫu test:     {total_samples}")
        print(f"Số lớp:            {len(supports)}")
        print(f"Support tối đa:    {max_support} ({max_support/total_samples:.1%})")
        print(f"Support tối thiểu: {min_support} ({min_support/total_samples:.1%})")
        print(f"Support trung bình:{avg_support:.0f}")
        print(f"Imbalance ratio:   {imbalance_ratio:.1f}x")
        
        print("\n" + "─" * 70)
        if imbalance_ratio <= 2:
            print("✅ TỐT: Data cân bằng (imbalance ratio ≤ 2x)")
        elif imbalance_ratio <= 5:
            print("⚠️  CẢNH BÁO: Data hơi không cân bằng (2x < ratio ≤ 5x)")
        elif imbalance_ratio <= 10:
            print("❌ BẤT CÂN BẰNG: Data không cân bằng (5x < ratio ≤ 10x)")
        else:
            print("🚨 RẤT BẤT CÂN BẰNG: Data rất không cân bằng (ratio > 10x)")

# ============================================================================
# PHẦN 5: Phân Tích Confusion Matrix
# ============================================================================
print("\n\n🔍 PHẦN 5: PHÂN TÍCH CONFUSION MATRIX")
print("-" * 70)

if "test_metrics.json" in data:
    metrics = data["test_metrics.json"]
    
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        
        print(f"Kích cỡ ma trận: {len(cm)} x {len(cm[0]) if cm else 0}")
        
        # Kiểm tra các dòng có tổng = 0 (lớp không bao giờ xuất hiện)
        zero_rows = []
        for i, row in enumerate(cm):
            if sum(row) == 0:
                zero_rows.append(i)
        
        if zero_rows:
            print(f"\n❌ {len(zero_rows)} dòng toàn 0 (không có mẫu test)")
            print(f"   Labels: {zero_rows[:10]}")
        
        # Kiểm tra mode collapse (một lớp chiếm > 50% dự đoán)
        pred_totals = [sum(cm[i]) for i in range(len(cm))]
        max_pred = max(pred_totals) if pred_totals else 0
        total_pred = sum(pred_totals)
        
        if max_pred / total_pred > 0.5:
            print(f"\n⚠️  MODE COLLAPSE: Một lớp = {max_pred/total_pred:.0%} dự đoán")
            print("   → Model quên mất các lớp khác")
        elif max_pred / total_pred > 0.35:
            print(f"\n⚠️  CẢNH BÁO: Một lớp = {max_pred/total_pred:.0%} dự đoán")
        else:
            print(f"\n✅ Dự đoán phân bố khá cân bằng (max = {max_pred/total_pred:.0%})")

# ============================================================================
# PHẦN 6: Training History (Convergence)
# ============================================================================
print("\n\n📉 PHẦN 6: LỊCH SỬ TRAINING - Convergence Analysis")
print("-" * 70)

if "history.json" in data:
    history = data["history.json"]
    
    epochs_ran = history.get("summary", {}).get("epochs_ran", 0)
    best_epoch = history.get("summary", {}).get("best_epoch", 0)
    best_val_loss = history.get("summary", {}).get("best_val_loss", 0)
    completed = history.get("summary", {}).get("completed", False)
    
    print(f"Epochs trained:    {epochs_ran}")
    print(f"Best epoch:        {best_epoch}")
    print(f"Best val loss:     {best_val_loss:.4f}")
    print(f"Training status:   {'✅ Hoàn tất' if completed else '⚠️  Dừng sớm'}")
    
    if not completed and epochs_ran < 30:
        print("\n⚠️  CẢNH BÁO: Training dừng sớm (< 30 epochs)")
        print("   → Có thể model chưa hội tụ")

# ============================================================================
# PHẦN 7: Kết Luận & Khuyến Nghị
# ============================================================================
print("\n\n" + "="*70)
print("🎓 KẾT LUẬN & KHUYẾN NGHỊ")
print("="*70)

verdict_score = 0
issues = []

if "test_metrics.json" in data:
    metrics = data["test_metrics.json"]
    accuracy = metrics.get("accuracy", 0)
    
    if "per_class_metrics" in metrics:
        per_class = metrics["per_class_metrics"]
        recalls = [m.get("recall", 0) for m in per_class]
        
        # Check 1: Accuracy reliability
        balanced_acc = sum(recalls) / len(recalls) if recalls else 0
        if abs(accuracy - balanced_acc) > 0.15:
            issues.append("❌ Accuracy BẤT CÂN BẰNG (gap > 15%)")
            issues.append("   → Confusion matrix là THIẾT YẾU để hiểu mô hình")
        else:
            verdict_score += 1
        
        # Check 2: Per-class performance
        poor_classes = [r for r in recalls if r < 0.5]
        zero_classes = [r for r in recalls if r == 0]
        
        if zero_classes:
            issues.append(f"❌ {len(zero_classes)} lớp có Recall = 0%")
            issues.append("   → Confusion matrix CẦN DÙNG để tìm nguyên nhân")
        
        if poor_classes:
            issues.append(f"⚠️  {len(poor_classes)} lớp có Recall < 50%")
            issues.append("   → Cần cải thiện model hoặc data")
            if len(poor_classes) <= 3:
                verdict_score += 0.5
        else:
            verdict_score += 1

if "history.json" in data:
    history = data["history.json"]
    if history.get("summary", {}).get("completed", False):
        verdict_score += 1
    else:
        issues.append("⚠️  Training dừng sớm")

print("\n" + "─" * 70)
print(f"Mô hình của bạn dùng Confusion Matrix được không?")
print(f"Verdict Score: {verdict_score}/3")

if not issues:
    print("\n✅ Model TỐT - Accuracy đáng tin cậy")
    print("   Có thể dùng Confusion Matrix để phân tích chi tiết")
    print("   hoặc áp dụng cho các bài toán tương tự")
else:
    print("\n🚨 Model CÓ VẤN ĐỀ - CẦN DÙNG Confusion Matrix!")
    for issue in issues:
        print(f"   {issue}")

# ============================================================================
# PHẦN 8: Áp Dụng Confusion Matrix
# ============================================================================
print("\n\n" + "="*70)
print("📚 HƯỚNG DẪN: CONFUSION MATRIX ÁP DỤNG TRONG CÁC TRƯỜNG HỢP")
print("="*70)

print("""
✅ CONFUSION MATRIX DÙNG ĐƯỢC KHI:
   1. Phân loại đa lớp (multi-class classification) ← BẠN ĐANG DÙNG
   2. Phát hiện class imbalance
   3. Tìm lớp nào yếu để cải thiện
   4. So sánh 2 model
   
❌ CONFUSION MATRIX KHÔNG DÙNG ĐƯỢC KHI:
   1. Bài toán hồi quy (regression)
   2. Phân loại nhị phân (binary) - dùng ROC/AUC thay vì CM
   3. Clustering (không có true labels)
   4. Unsupervised learning
   
🎯 CASE DÙNG CỦA BẠN:
   Phân loại thuốc 108 lớp → DÙNG CONFUSION MATRIX được 100%!
   
📊 CÁCH DÙNG:
   1. Xem confusion_matrix.png → phát hiện lỗi nhầm
   2. Xem per-class metrics → tìm lớp yếu
   3. Xem recall thấp → xác định nguyên nhân
   4. Sau cải thiện → so sánh new vs old confusion matrix
""")

print("\n" + "="*70)
print("✨ Phân tích hoàn tất!")
print("="*70 + "\n")
