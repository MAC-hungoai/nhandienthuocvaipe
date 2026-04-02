#!/usr/bin/env python3
"""
Diagnostic Report Generator - Tạo báo cáo phân tích chi tiết từ test_metrics.json

Sử dụng:
    python confusion_matrix_diagnostic_report.py
    python confusion_matrix_diagnostic_report.py checkpoints/test_metrics.json
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_metrics(metrics_path: str | Path = "checkpoints/test_metrics.json") -> Dict:
    """Tải metrics từ file JSON"""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy: {metrics_path}")
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_banner(text: str, width: int = 80):
    """In banner đẹp"""
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_section(title: str):
    """In title section"""
    print(f"\n📋 {title}")
    print("-" * 70)


def analyze_metrics(metrics: Dict) -> Dict:
    """Phân tích metrics và trả lại diagnostic info"""
    
    accuracy = float(metrics.get("accuracy", 0))
    macro_f1 = float(metrics.get("macro_f1", 0))
    per_class = list(metrics.get("per_class_metrics", []))
    per_class_accuracy = dict(metrics.get("per_class_accuracy", {}))
    
    # Calculate gaps
    gap_acc_f1 = accuracy - macro_f1
    
    # Per-class analysis
    recalls = []
    precisions = []
    f1s = []
    supports = []
    zero_recall_classes = []
    low_recall_classes = []
    
    # Handle both formats: per_class_metrics (detailed) and per_class_accuracy (simple)
    if per_class:
        for item in per_class:
            label_id = int(item.get("label_id", -1))
            support = int(item.get("support", 0))
            recall = float(item.get("recall", 0))
            precision = float(item.get("precision", 0))
            f1 = float(item.get("f1", 0))
            
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            supports.append(support)
            
            if recall == 0 and support > 0:
                zero_recall_classes.append((label_id, support))
            if recall < 0.5 and support > 0:
                low_recall_classes.append((label_id, support, recall))
    elif per_class_accuracy:
        # Simple format: just accuracy per class
        for label_id_str, acc in per_class_accuracy.items():
            label_id = int(label_id_str)
            recall = float(acc)
            
            recalls.append(recall)
            precisions.append(recall)
            f1s.append(recall)
            
            if recall == 0:
                zero_recall_classes.append((label_id, None))
            if recall < 0.5:
                low_recall_classes.append((label_id, None, recall))
    
    recalls = np.array(recalls) if recalls else np.array([])
    precisions = np.array(precisions) if precisions else np.array([])
    f1s = np.array(f1s) if f1s else np.array([])
    supports = np.array(supports) if supports else np.array([])
    
    balanced_accuracy = float(recalls.mean()) if len(recalls) > 0 else 0.0
    median_recall = float(np.median(recalls)) if len(recalls) > 0 else 0.0
    min_recall = float(recalls.min()) if len(recalls) > 0 else 0.0
    max_recall = float(recalls.max()) if len(recalls) > 0 else 0.0
    
    gap_acc_balanced = accuracy - balanced_accuracy
    
    # Confusion matrix analysis
    cm_data = metrics.get("confusion_matrix")
    dominant_pred_ratio = 0
    if cm_data:
        cm = np.array(cm_data)
        total = cm.sum()
        if total > 0:
            pred_totals = cm.sum(axis=0)
            dominant_pred_ratio = float(pred_totals.max() / total)
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "gap_acc_f1": gap_acc_f1,
        "gap_acc_balanced": gap_acc_balanced,
        "recalls": recalls,
        "precisions": precisions,
        "f1s": f1s,
        "supports": supports,
        "median_recall": median_recall,
        "min_recall": min_recall,
        "max_recall": max_recall,
        "zero_recall_classes": zero_recall_classes,
        "low_recall_classes": low_recall_classes,
        "dominant_pred_ratio": dominant_pred_ratio,
        "num_classes": len(recalls) if len(recalls) > 0 else 0,
        "per_class_metrics": per_class,
    }


def get_verdict(analysis: Dict) -> tuple[str, List[str]]:
    """
    Xác định từ điển cảnh báo
    
    Returns:
        (verdict_text, list_of_warnings)
    """
    warnings = []
    
    accuracy = analysis["accuracy"]
    gap_acc_f1 = analysis["gap_acc_f1"]
    gap_acc_balanced = analysis["gap_acc_balanced"]
    min_recall = analysis["min_recall"]
    zero_recall_classes = analysis["zero_recall_classes"]
    low_recall_classes = analysis["low_recall_classes"]
    dominant_pred_ratio = analysis["dominant_pred_ratio"]
    
    # Check each criterion
    if gap_acc_f1 > 0.10:
        warnings.append(f"Gap Accuracy - Macro F1: {gap_acc_f1:.2%} (> 10%)")
    
    if gap_acc_balanced > 0.15:
        warnings.append(f"Gap Accuracy - Balanced Acc: {gap_acc_balanced:.2%} (> 15%)")
    
    if min_recall < 0.50:
        warnings.append(f"Min Recall: {min_recall:.1%} (< 50%)")
    
    if len(zero_recall_classes) > 0:
        warnings.append(f"{len(zero_recall_classes)} class(es) với recall = 0%")
    
    if len(low_recall_classes) > 2:
        warnings.append(f"{len(low_recall_classes)} class(es) với recall < 50%")
    
    if dominant_pred_ratio > 0.35:
        warnings.append(f"Mode collapse: 1 class = {dominant_pred_ratio:.1%} tất cả dự đoán")
    
    # Determine verdict
    if len(zero_recall_classes) > 2 or gap_acc_balanced > 0.20:
        verdict = "❌ LIKELY_MISLEADING"
        detail = "Accuracy bị 'bóp méo' - KHÔNG TIN CẬY"
    elif len(warnings) >= 2:
        verdict = "⚠️ NEEDS_CONTEXT"
        detail = "Accuracy có cảnh báo - CẦN KIỂM TRA CHI TIẾT"
    else:
        verdict = "✅ LOOKS_CONSISTENT"
        detail = "Accuracy tương đối tin cậy"
    
    return verdict, warnings, detail


def print_diagnostic_report(metrics_path: str | Path = "checkpoints/test_metrics.json"):
    """In báo cáo diagnostic"""
    
    # Load metrics
    try:
        metrics = load_metrics(metrics_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Analyze
    analysis = analyze_metrics(metrics)
    verdict, warnings, detail = get_verdict(analysis)
    
    # Print header
    print_banner(f"📊 CONFUSION MATRIX DIAGNOSTIC REPORT")
    print(f"File: {Path(metrics_path).name}")
    print(f"Ngày: {Path.cwd()}")
    
    # Overall metrics
    print_section("🎯 OVERALL METRICS")
    print(f"  Accuracy:                {analysis['accuracy']:>7.2%}")
    print(f"  Macro F1:                {analysis['macro_f1']:>7.2%}")
    print(f"  Balanced Accuracy:       {analysis['balanced_accuracy']:>7.2%}")
    print(f"  Median Class Recall:     {analysis['median_recall']:>7.2%}")
    
    # Gaps
    print_section("📈 GAPS & CONSISTENCY")
    print(f"  Gap (Accuracy - Macro F1):      {analysis['gap_acc_f1']:>7.2%}", end="")
    if analysis['gap_acc_f1'] > 0.10:
        print("  ⚠️ WARNING")
    elif analysis['gap_acc_f1'] > 0.05:
        print("  ⚠️ caution")
    else:
        print("  ✓ OK")
    
    print(f"  Gap (Acc - Balanced Acc):       {analysis['gap_acc_balanced']:>7.2%}", end="")
    if analysis['gap_acc_balanced'] > 0.15:
        print("  ⚠️ WARNING")
    elif analysis['gap_acc_balanced'] > 0.05:
        print("  ⚠️ caution")
    else:
        print("  ✓ OK")
    
    # Per-class statistics
    print_section("📋 PER-CLASS STATISTICS")
    print(f"  Number of classes:       {analysis['num_classes']}")
    print(f"  Min recall:              {analysis['min_recall']:>7.2%}", end="")
    if analysis['min_recall'] < 0.50:
        print("  ⚠️ WARNING")
    else:
        print("  ✓ OK")
    
    print(f"  Max recall:              {analysis['max_recall']:>7.2%}")
    if len(analysis['recalls']) > 0:
        avg_recall = float(np.mean(analysis['recalls']))
        std_recall = float(np.std(analysis['recalls']))
        print(f"  Avg recall:              {avg_recall:>7.2%}")
        print(f"  Std recall:              {std_recall:>7.2%}")
    else:
        print(f"  Avg recall:              N/A")
        print(f"  Std recall:              N/A")
    
    # Class issues
    print_section("🚨 CLASS ISSUES")
    
    if analysis['zero_recall_classes']:
        print(f"  ❌ Zero Recall ({len(analysis['zero_recall_classes'])} classes):")
        for label_id, support in analysis['zero_recall_classes'][:5]:
            print(f"     - Label {label_id:3d}: support={support}")
        if len(analysis['zero_recall_classes']) > 5:
            print(f"     ... và {len(analysis['zero_recall_classes']) - 5} lớp khác")
    else:
        print("  ✓ Không có lớp với recall = 0%")
    
    print()
    if analysis['low_recall_classes']:
        print(f"  ⚠️ Low Recall < 50% ({len(analysis['low_recall_classes'])} classes):")
        sorted_low = sorted(analysis['low_recall_classes'], key=lambda x: x[2])
        for label_id, support, recall in sorted_low[:5]:
            support_str = f"{support:3d}" if support is not None else "N/A"
            print(f"     - Label {label_id:3d}: support={support_str}, recall={recall:.1%}")
        if len(analysis['low_recall_classes']) > 5:
            print(f"     ... và {len(analysis['low_recall_classes']) - 5} lớp khác")
    else:
        print("  ✓ Tất cả lớp recall >= 50%")
    
    print()
    if analysis['dominant_pred_ratio'] > 0.35:
        print(f"  ⚠️ Mode Collapse:")
        print(f"     - 1 class chiếm {analysis['dominant_pred_ratio']:.1%} tất cả dự đoán")
    else:
        print("  ✓ Không có mode collapse")
    
    # Verdict
    print_section("📌 VERDICT")
    print(f"\n  {verdict}")
    print(f"\n  {detail}")
    
    # Warnings
    if warnings:
        print(f"\n  Cảnh báo cụ thể:")
        for i, warning in enumerate(warnings, 1):
            print(f"    {i}. {warning}")
    
    # Recommendations
    print_section("💡 RECOMMENDATIONS")
    
    if analysis['zero_recall_classes']:
        print(f"  1. ❌ FIX ZERO-RECALL CLASSES")
        print(f"     - Classes: {', '.join(str(c[0]) for c in analysis['zero_recall_classes'][:5])}")
        print(f"     - Hành động:")
        print(f"       a) Kiểm tra dữ liệu có lỗi không")
        print(f"       b) Tăng class weight cho các lớp này")
        print(f"       c) Thu thập thêm dữ liệu")
        print(f"       d) Dùng data augmentation")
    
    if len(analysis['low_recall_classes']) > 2:
        print(f"  2. ⚠️ IMPROVE LOW-RECALL CLASSES")
        print(f"     - {len(analysis['low_recall_classes'])} classes có recall < 50%")
        print(f"     - Hành động tương tự như trên")
    
    if analysis['gap_acc_f1'] > 0.10:
        print(f"  3. 📊 INVESTIGATE CLASS IMBALANCE")
        print(f"     - Gap {analysis['gap_acc_f1']:.2%} cho thấy class imbalance")
        print(f"     - Check dữ liệu: mỗi class có bao nhiêu mẫu?")
        print(f"     - Cân bằng weight hoặc dữ liệu")
    
    if analysis['dominant_pred_ratio'] > 0.35:
        print(f"  4. 🔄 PREVENT MODE COLLAPSE")
        print(f"     - Model đang 'quên' một số lớp (predicted > 35% một lớp)")
        print(f"     - Dùng loss weighting hoặc Focal Loss")
    
    print()
    print_banner("END OF REPORT")


if __name__ == "__main__":
    import sys
    
    metrics_file = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/test_metrics.json"
    print_diagnostic_report(metrics_file)
