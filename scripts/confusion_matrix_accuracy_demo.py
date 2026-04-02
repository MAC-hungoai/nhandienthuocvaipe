"""
Small sklearn demo showing why confusion matrix matters more than headline accuracy.

Usage:
  python scripts/confusion_matrix_accuracy_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def build_demo_cases() -> list[dict[str, object]]:
    return [
        {
            "name": "Case 1: High accuracy trap",
            "description": "Model predicts only the majority class.",
            "y_true": np.asarray([0] * 95 + [1] * 5, dtype=np.int64),
            "y_pred": np.asarray([0] * 100, dtype=np.int64),
        },
        {
            "name": "Case 2: Lower accuracy, better model",
            "description": "Accuracy drops a bit, but minority recall becomes useful.",
            "y_true": np.asarray([0] * 95 + [1] * 5, dtype=np.int64),
            "y_pred": np.asarray([0] * 89 + [1] * 6 + [0] * 1 + [1] * 4, dtype=np.int64),
        },
    ]


def summarize_case(case: dict[str, object]) -> dict[str, object]:
    y_true = np.asarray(case["y_true"], dtype=np.int64)
    y_pred = np.asarray(case["y_pred"], dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    summary = {
        "cm": cm,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }
    return summary


def print_case_report(case: dict[str, object], summary: dict[str, object]) -> None:
    cm = np.asarray(summary["cm"], dtype=np.int64)
    recall = np.asarray(summary["recall"], dtype=np.float32)

    print("=" * 80)
    print(case["name"])
    print(case["description"])
    print("Confusion matrix:")
    print(cm)
    print(f"Accuracy:          {summary['accuracy']:.2%}")
    print(f"Balanced accuracy: {summary['balanced_accuracy']:.2%}")
    print(f"Macro-F1:          {summary['macro_f1']:.2%}")
    print(f"Recall class 0:    {recall[0]:.2%}")
    print(f"Recall class 1:    {recall[1]:.2%}")

    if recall[1] == 0:
        print("Verdict: accuracy is misleading because the minority class is never recovered.")
    else:
        print("Verdict: this model is more useful because it actually detects the minority class.")


def save_plot(cases: list[dict[str, object]], summaries: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, len(cases), figsize=(12, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for axis, case, summary in zip(axes, cases, summaries):
        display = ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(summary["cm"], dtype=np.int64),
            display_labels=["Class 0", "Class 1"],
        )
        display.plot(ax=axis, colorbar=False, cmap="Blues", values_format="d")
        axis.set_title(
            f"{case['name']}\nacc={summary['accuracy']:.1%} | bal_acc={summary['balanced_accuracy']:.1%}",
            fontsize=11,
        )

    figure.suptitle("Confusion matrix can expose useless high accuracy", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    cases = build_demo_cases()
    summaries = [summarize_case(case) for case in cases]

    for case, summary in zip(cases, summaries):
        print_case_report(case, summary)

    output_path = Path("outputs") / "confusion_matrix_accuracy_demo.png"
    save_plot(cases, summaries, output_path)
    print("=" * 80)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
