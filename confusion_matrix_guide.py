"""
Hướng dẫn sử dụng Confusion Matrix để đánh giá độ chính xác mô hình
=================================================================

VẤNĐỀ: Độ chính xác cao (accuracy) có phải lúc nào cũng tốt không?
TRANSWEROMPONENT: KHÔNG! Confusion matrix giúp bạn phát hiện khi nào accuracy là "rác"

Tác giả: Guided Learning Example
Ngôn ngữ: Python
Yêu cầu: numpy, matplotlib, scikit-learn
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def example_1_tai_sao_accuracy_co_the_la_rac():
    """
    VÍ DỤ 1: Phát hiện khi accuracy cao nhưng model là rác
    
    Tưởng tượng bạn có:
    - 100 mẫu thuốc
    - 99 mẫu thuộc loại A
    - 1 mẫu thuộc loại B
    
    Model ngu ngợc nói "tất cả đều là loại A" → Accuracy = 99%
    
    ✓ Accuracy: 99% (quá cao!)
    ✗ Nhưng model KHÔNG bao giờ đoán được loại B
    ✗ Recall cho loại B: 0%
    ✗ Precision cho loại B: undefined (không có dự đoán nào)
    
    → Confusion matrix phát hiện vấn đề này!
    """
    print("=" * 80)
    print("VÍ DỤ 1: Tại sao Accuracy có thể là 'rác'?")
    print("=" * 80)

    # Dữ liệu thực tế
    y_true = [0] * 99 + [1]  # 99 loại A, 1 loại B
    y_pred = [0] * 100      # Model dự đoán tất cả là loại A

    # Tính metrics
    accuracy = sum(y_true[i] == y_pred[i] for i in range(len(y_true))) / len(y_true)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n📊 Độ chính xác (Accuracy): {accuracy:.1%}")
    print(f"\n🎯 Confusion Matrix:")
    print(f"     Dự đoán A  Dự đoán B")
    print(f"Thật A:  {cm[0, 0]:3d}      {cm[0, 1]:3d}")
    print(f"Thật B:  {cm[1, 0]:3d}      {cm[1, 1]:3d}")

    print("\n❌ VẤN ĐỀ:")
    print(f"   - Model chưa bao giờ dự đoán B (cột B = 0)")
    print(f"   - Recall cho B = 0/1 = 0% (gặp B mà không nhận ra)")
    print(f"   - Model KHÔNG sử dụng được vì thiếu loại B")

    # So sánh với F1
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n📈 Macro F1-score: {f1:.2%}")
    print(f"   → F1 thấp hơn accuracy, phát hiện vấn đề!")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "problem": "Class imbalance + model mode collapse"
    }


def example_2_phat_hien_lop_kho_nhan_biet():
    """
    VÍ DỤ 2: Phát hiện lớp nào trong mô hình có vấn đề
    
    Model phân loại 3 loại thuốc với dữ liệu không cân bằng:
    - Loại A: dễ phân loại (recall 95%)
    - Loại B: trung bình (recall 70%)
    - Loại C: khó phân loại (recall 40%)
    
    Confusion matrix cho thấy:
    ✓ Bạn biết loại nào cần cải thiện
    ✓ Biết lỗi nhầm lẫn phổ biến (C thường nhầm thành B)
    """
    print("\n" + "=" * 80)
    print("VÍ DỤ 2: Phát hiện các lớp có vấn đề")
    print("=" * 80)

    # Dữ liệu với 3 loại thuốc
    y_true = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 10 loại A
        1, 1, 1, 1, 1, 1, 1,            # 7 loại B
        2, 2, 2, 2, 2                   # 5 loại C
    ])

    # Dự đoán (model tốt với A, trung bình với B, tệ với C)
    y_pred = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  # A: 9/10 đúng (90%)
        1, 1, 1, 1, 1, 2, 2,            # B: 5/7 đúng (71%)
        2, 2, 1, 1, 1                   # C: 2/5 đúng (40%)
    ])

    cm = confusion_matrix(y_true, y_pred)
    accuracy = (y_true == y_pred).sum() / len(y_true)

    print(f"\n📊 Độ chính xác chung: {accuracy:.1%}")
    print(f"\n🎯 Confusion Matrix (3x3):")
    print(f"     Dự đoán A  Dự đoán B  Dự đoán C")
    for i in range(3):
        print(f"Thật {i}: {cm[i, 0]:3d}      {cm[i, 1]:3d}      {cm[i, 2]:3d}")

    print("\n📋 Chi tiết theo lớp:")
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred)
    for class_id in range(3):
        print(f"   Loại {class_id}:")
        print(f"      - Precision: {precisions[class_id]:.1%}")
        print(f"      - Recall:    {recalls[class_id]:.1%}")
        print(f"      - F1-score:  {f1s[class_id]:.1%}")
        print(f"      - Support:   {supports[class_id]}")

    print("\n🔍 NHẬN XÉT:")
    print(f"   - Loại 0: Dễ phân loại (recall 90%)")
    print(f"   - Loại 1: Trung bình (recall 71%)")
    print(f"   - Loại 2: KHÓ phân loại (recall 40%) ⚠️")
    print(f"   - Lỗi nhầm phổ biến: C nhầm thành B (3 trường hợp)")

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "precisions": precisions,
        "recalls": recalls,
        "problem": "Per-class performance imbalance"
    }


def example_3_phat_hien_class_mode_collapse():
    """
    VÍ DỤ 3: Mode Collapse - model chỉ dự đoán 1-2 loại
    
    Model học được mà quên mất các loại khác!
    Tất cả hoặc hầu hết dự đoán đều là loại A
    
    Accuracy vẫn có thể cao nếu A chiếm phần lớn dữ liệu,
    nhưng model không đủ khả năng phân biệt
    """
    print("\n" + "=" * 80)
    print("VÍ DỤ 3: Mode Collapse (model quên mất các loại)")
    print("=" * 80)

    # Dữ liệu 4 loại
    y_true = np.array([
        0, 0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2,
        3, 3
    ])

    # Model dự đoán: hầu hết là loại 0 (mode collapse)
    y_pred = np.array([
        0, 0, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, 0,
        0, 3
    ])

    cm = confusion_matrix(y_true, y_pred)
    accuracy = (y_true == y_pred).sum() / len(y_true)

    print(f"\n📊 Độ chính xác: {accuracy:.1%}")
    print(f"\n🎯 Confusion Matrix:")
    print()
    print(f"     Dự đoán 0  Dự đoán 1  Dự đoán 2  Dự đoán 3")
    for i in range(4):
        row_str = f"Thật {i}:"
        for j in range(4):
            row_str += f"  {cm[i, j]:3d}"
        print(row_str)

    # Phân tích
    print("\n⚠️ CẢNH BÁO:")
    print(f"   - Tổng dự đoán loại 0: {cm[:, 0].sum()} / {len(y_pred)} ({cm[:, 0].sum()/len(y_pred):.1%})")
    print(f"   - Tổng dự đoán loại 1: {cm[:, 1].sum()} / {len(y_pred)}")
    print(f"   - Tổng dự đoán loại 2: {cm[:, 2].sum()} / {len(y_pred)}")
    print(f"   - Tổng dự đoán loại 3: {cm[:, 3].sum()} / {len(y_pred)}")
    print(f"\n   → Model QUÊN mất loại 2 hoàn toàn!")
    print(f"   → Model mode collapse on class 0-1 only!")

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "problem": "Mode collapse - model ignores some classes"
    }


def visualize_confusion_matrix(cm: np.ndarray, title: str, figname: str):
    """Vẽ confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    # Thêm giá trị vào ô
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Dự đoán / Predicted")
    ax.set_ylabel("Thật / True")
    ax.set_title(title)
    
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(figname, dpi=100, bbox_inches="tight")
    print(f"\n📸 Đã lưu: {figname}")
    plt.close()


def dieu_kien_accuracy_la_rac():
    """
    📋 DANH SÁCH KIỂM TRA: Khi nào Accuracy là 'rác'?
    
    Accuracy chỉ tốt khi:
    ✓ Data cân bằng (mỗi lớp có ~số mẫu tương tự)
    ✓ Mỗi lớp có recall cao (~80%+ trở lên)
    ✓ Balanced accuracy ≈ Accuracy (không chênh lệch >10%)
    ✓ Macro F1 ≈ Accuracy (không chênh lệch >10%)
    
    Accuracy có thể là 'rác' khi:
    ❌ Data không cân bằng (một lớp chiếm >70% dữ liệu)
    ❌ Một số lớp có recall <50%
    ❌ Balanced accuracy thấp hơn accuracy >20%
    ❌ Một lớp dự đoán chiếm >35% tất cả các dự đoán (mode collapse)
    ❌ Macro F1 thấp hơn accuracy >10%
    """
    print("\n" + "=" * 80)
    print("📋 DANH SÁCH KIỂM TRA: Khi nào Accuracy là 'RÁC'?")
    print("=" * 80)

    criteria = [
        ("Data không cân bằng", "≥1 lớp chiếm >70% dữ liệu", "❌"),
        ("Recall kém", "≥1 lớp có recall <50%", "❌"),
        ("Gap lớn", "Balanced Acc vs Accuracy: >20% khoảng cách", "❌"),
        ("Mode collapse", "1 lớp = >35% tất cả dự đoán", "❌"),
        ("Macro F1 kém", "Macro F1 < Accuracy - 10%", "❌"),
    ]

    print("\n❌ CẢNH BÁO (Accuracy có thể là rác):")
    for prob, condition, icon in criteria[:4]:
        print(f"   {icon} {prob}: {condition}")

    print("\n✅ TÍNH NĂNG (Accuracy có thể tin cậy):")
    print(f"   ✓ Data cân bằng: mỗi lớp ~20-25% trong dataset 3-5 lớp")
    print(f"   ✓ Recall cao: tất cả lớp ≥80%")
    print(f"   ✓ Balanced Acc ≈ Accuracy (chênh lệch <5%)")
    print(f"   ✓ Macro F1 ≈ Accuracy (chênh lệch <5%)")


def load_va_phan_tich_metrics(metrics_path: Path | str):
    """
    Tải file test_metrics.json và phân tích
    """
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        print(f"❌ File không tìm thấy: {metrics_path}")
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(f"\n{'='*80}")
    print(f"📊 PHÂN TÍCH METRICS: {metrics_path.name}")
    print(f"{'='*80}")

    # Metrics cơ bản
    accuracy = metrics.get("accuracy", 0.0)
    macro_f1 = metrics.get("macro_f1", 0.0)
    
    print(f"\n🎯 KỲ VỌNG CƠ BẢN:")
    print(f"   Accuracy:    {accuracy:.2%}")
    print(f"   Macro F1:    {macro_f1:.2%}")
    print(f"   Gap (A-F1):  {(accuracy - macro_f1):.2%}")

    # Phân tích per-class
    per_class = metrics.get("per_class_metrics", [])
    if per_class:
        print(f"\n📋 CHI TIẾT THEO LỚP:")
        print(f"   {'Class':>6} {'Support':>8} {'Recall':>8} {'Precision':>10} {'F1':>8}")
        print(f"   {'-'*42}")
        
        recalls = []
        for item in per_class:
            label_id = item.get("label_id", "?")
            support = item.get("support", 0)
            recall = item.get("recall", 0.0)
            precision = item.get("precision", 0.0)
            f1 = item.get("f1", 0.0)
            
            recalls.append(recall)
            print(f"   {label_id:>6} {support:>8} {recall:>7.1%} {precision:>9.1%} {f1:>7.1%}")
        
        if recalls:
            min_recall = min(recalls)
            max_recall = max(recalls)
            balanced_acc = np.mean(recalls)
            
            print(f"\n   Recall phân tích:")
            print(f"      - Min: {min_recall:.1%}")
            print(f"      - Max: {max_recall:.1%}")
            print(f"      - Balanced Accuracy: {balanced_acc:.1%}")
            print(f"      - Gap (Acc vs BalAcc): {(accuracy - balanced_acc):.2%}")
            
            if min_recall < 0.5:
                print(f"\n⚠️  CẢNH BÁO: Một số lớp có recall <50%")
            if (accuracy - balanced_acc) > 0.15:
                print(f"\n⚠️  CẢNH BÁO: Khoảng cách lớn giữa Accuracy và Balanced Accuracy!")

    # Phân tích confusion matrix
    cm_data = metrics.get("confusion_matrix")
    if cm_data:
        cm = np.array(cm_data)
        total = cm.sum()
        pred_totals = cm.sum(axis=0)
        top_pred = pred_totals.argmax()
        top_pred_ratio = pred_totals[top_pred] / total
        
        print(f"\n🎲 PHÂN TÍCH CONFUSION MATRIX:")
        print(f"   Tổng mẫu: {total}")
        print(f"   Lớp được dự đoán nhiều nhất: Lớp {top_pred}")
        print(f"   Tỷ lệ: {top_pred_ratio:.1%} của tất cả dự đoán")
        
        if top_pred_ratio > 0.35:
            print(f"\n⚠️  CẢNH BÁO: Mode collapse - một lớp chiếm >{0.35:.0%} dự đoán!")

    return metrics


def tong_ket():
    """Tóm tắt học được"""
    print("\n\n" + "=" * 80)
    print("🎓 TÓM TẮT: Cách dùng Confusion Matrix để phát hiện Accuracy 'rác'")
    print("=" * 80)

    print("""
1️⃣  ACCURACY KHÔNG PHẢI LÚC NÀO CŨNG TỐTÓ TRỊ
   - Khi data không cân bằng (95% loại A, 5% loại B)
   - Khi model mode collapse (quên mất loại B)
   - Accuracy cao nhưng model không sử dụng được thực tế

2️⃣  CONFUSION MATRIX GIÚP PHÁT HIỆN
   ✓ Mỗi lớp có recall như thế nào
   ✓ Lỗi nhầm lẫn phổ biến là gì (B thường nhầm thành C)
   ✓ Lớp nào cần cải thiện
   ✓ Có phải model đang mode collapse không

3️⃣  METRICS BỔ SUNG CẦN KIỂM TRA
   - Balanced Accuracy = trung bình recall tất cả lớp
   - Macro F1 = trung bình F1 tất cả lớp
   - Nếu (Accuracy - Balanced Acc) > 15% → Accuracy bị "bóp méo"

4️⃣  DANH SÁCH KIỂM TRA NHANH
   [ ] Data cân bằng?
   [ ] Tất cả lớp recall ≥ 80%?
   [ ] Accuracy ≈ Balanced Accuracy (< 5% gap)?
   [ ] Accuracy ≈ Macro F1 (< 5% gap)?
   [ ] Không có class nào chiếm > 35% dự đoán?
   
   Trả lời TẠT CẢ "CÓ" → Accuracy tin cậy ✓
   Trả lời CÓ "KHÔNG" → Cần xem confusion matrix chi tiết ⚠️

5️⃣  KỸ NĂNG PHÂN TÍCH
   `python check_model_metrics.py --metrics checkpoints/test_metrics.json`
   
   → Tool này sẽ báo "LOOKS_CONSISTENT" hoặc "LIKELY_MISLEADING" ⚠️
   """)


if __name__ == "__main__":
    print("\n🎬 BẮTÁC PHƯƠNG PHÁP KIỂM TRA ACCURACY BẰNG CONFUSION MATRIX")
    print("="*80)

    # Chạy các ví dụ
    result1 = example_1_tai_sao_accuracy_co_the_la_rac()
    visualize_confusion_matrix(
        result1["confusion_matrix"],
        "VÍ DỤ 1: Accuracy cao nhưng model quên mất lớp B",
        "example1_mode_collapse.png"
    )

    result2 = example_2_phat_hien_lop_kho_nhan_biet()
    visualize_confusion_matrix(
        result2["confusion_matrix"],
        "VÍ DỤ 2: Lớp C khó nhận biết (recall 40%)",
        "example2_per_class_imbalance.png"
    )

    result3 = example_3_phat_hien_class_mode_collapse()
    visualize_confusion_matrix(
        result3["confusion_matrix"],
        "VÍ DỤ 3: Mode Collapse - quên lớp 2 hoàn toàn",
        "example3_severe_collapse.png"
    )

    # Đặt tiêu chí
    dieu_kien_accuracy_la_rac()

    # Thử phân tích file thực tế nếu tồn tại
    metrics_file = Path("checkpoints") / "test_metrics.json"
    if metrics_file.exists():
        load_va_phan_tich_metrics(metrics_file)
    else:
        print(f"\n💡 Gợi ý: Khi có {metrics_file}, chạy:")
        print(f"   python confusion_matrix_guide.py")

    # Tóm tắt
    tong_ket()

    print("\n✅ Hoàn thành! Kiểm tra các file PNG được tạo ra để thấy hình ảnh.")
    print("📚 Đọc thêm: README.md và IMPROVEMENT_ROADMAP.md")
