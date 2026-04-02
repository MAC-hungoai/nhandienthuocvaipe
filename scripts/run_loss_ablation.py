"""
Run a small ablation comparing classifier loss functions on the same training setup.

Usage:
  python scripts/run_loss_ablation.py --output-root checkpoints/loss_ablation -- --epochs 20 --min-epochs 8
  python scripts/run_loss_ablation.py --dry-run -- --epochs 5 --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_LOSSES = ["cross_entropy", "weighted_ce", "focal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare train.py runs across multiple loss functions.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints") / "loss_ablation",
        help="Root directory containing one subdirectory per loss run.",
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        choices=DEFAULT_LOSSES,
        default=DEFAULT_LOSSES,
        help="Losses to compare.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to launch train.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing training.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep running other losses even if one run fails.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to train.py. Prefix them with '--'.",
    )
    return parser.parse_args()


def normalize_train_args(train_args: List[str]) -> List[str]:
    normalized = list(train_args)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]

    forbidden_flags = {"--output-dir", "--loss-type"}
    overlaps = [arg for arg in normalized if arg in forbidden_flags]
    if overlaps:
        overlap_text = ", ".join(sorted(set(overlaps)))
        raise ValueError(f"Do not pass {overlap_text} through train_args; the ablation runner manages those flags.")
    return normalized


def load_json(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def build_command(python_executable: Path, train_py: Path, output_dir: Path, loss_type: str, train_args: List[str]) -> List[str]:
    return [
        str(python_executable),
        str(train_py),
        "--output-dir",
        str(output_dir),
        "--loss-type",
        loss_type,
        *train_args,
    ]


def summarize_run(loss_type: str, output_dir: Path) -> Dict[str, object]:
    test_metrics = load_json(output_dir / "test_metrics.json") or {}
    accuracy_diagnostics = load_json(output_dir / "accuracy_diagnostics.json") or {}
    history = load_json(output_dir / "history.json") or {}
    history_summary = history.get("summary", {}) if isinstance(history, dict) else {}

    return {
        "loss_type": loss_type,
        "output_dir": str(output_dir),
        "accuracy": test_metrics.get("accuracy"),
        "top3_accuracy": test_metrics.get("top3_accuracy"),
        "macro_f1": test_metrics.get("macro_f1"),
        "balanced_accuracy": accuracy_diagnostics.get("balanced_accuracy"),
        "verdict": accuracy_diagnostics.get("verdict"),
        "zero_recall_classes": len(accuracy_diagnostics.get("zero_recall_classes", [])),
        "epochs_ran": history_summary.get("epochs_ran"),
        "best_epoch": history_summary.get("best_epoch"),
        "best_val_balanced_accuracy": history_summary.get("best_val_balanced_accuracy"),
    }


def format_percent(value: object) -> str:
    if value is None:
        return "--"
    return f"{float(value):.2%}"


def save_summary_files(output_root: Path, rows: List[Dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "ablation_summary.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    csv_path = output_root / "ablation_summary.csv"
    fieldnames = [
        "loss_type",
        "accuracy",
        "top3_accuracy",
        "macro_f1",
        "balanced_accuracy",
        "verdict",
        "zero_recall_classes",
        "epochs_ran",
        "best_epoch",
        "best_val_balanced_accuracy",
        "output_dir",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_root / "ablation_summary.md"
    markdown_lines = [
        "# Loss Ablation Summary",
        "",
        "| loss_type | accuracy | top3 | macro_f1 | balanced_accuracy | verdict | zero_recall | epochs_ran | best_epoch |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            f"{row['loss_type']} | "
            f"{format_percent(row['accuracy'])} | "
            f"{format_percent(row['top3_accuracy'])} | "
            f"{format_percent(row['macro_f1'])} | "
            f"{format_percent(row['balanced_accuracy'])} | "
            f"{row.get('verdict', '--')} | "
            f"{row.get('zero_recall_classes', '--')} | "
            f"{row.get('epochs_ran', '--')} | "
            f"{row.get('best_epoch', '--')} |"
        )
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown_lines) + "\n")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    train_py = repo_root / "train.py"
    train_args = normalize_train_args(list(args.train_args))
    rows: List[Dict[str, object]] = []

    print(f"Output root: {args.output_root}")
    print(f"Losses: {', '.join(args.losses)}")
    if train_args:
        print(f"Forwarded train args: {' '.join(train_args)}")
    print()

    for loss_type in args.losses:
        output_dir = args.output_root / loss_type
        command = build_command(args.python, train_py, output_dir, loss_type, train_args)
        print(f"[{loss_type}] {' '.join(command)}")

        if args.dry_run:
            continue

        try:
            subprocess.run(command, cwd=repo_root, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[{loss_type}] failed with exit code {exc.returncode}")
            if not args.continue_on_error:
                raise
            rows.append(
                {
                    "loss_type": loss_type,
                    "output_dir": str(output_dir),
                    "accuracy": None,
                    "top3_accuracy": None,
                    "macro_f1": None,
                    "balanced_accuracy": None,
                    "verdict": "FAILED",
                    "zero_recall_classes": None,
                    "epochs_ran": None,
                    "best_epoch": None,
                    "best_val_balanced_accuracy": None,
                }
            )
            continue

        row = summarize_run(loss_type, output_dir)
        rows.append(row)
        print(
            f"[{loss_type}] acc={format_percent(row['accuracy'])} | "
            f"macro_f1={format_percent(row['macro_f1'])} | "
            f"balanced_acc={format_percent(row['balanced_accuracy'])} | "
            f"zero_recall={row['zero_recall_classes']} | verdict={row['verdict']}"
        )

    if args.dry_run:
        print("\nDry run only. No training jobs were executed.")
        return

    save_summary_files(args.output_root, rows)
    print()
    print(f"Saved summary: {args.output_root / 'ablation_summary.json'}")
    print(f"Saved summary: {args.output_root / 'ablation_summary.csv'}")
    print(f"Saved summary: {args.output_root / 'ablation_summary.md'}")


if __name__ == "__main__":
    main()
