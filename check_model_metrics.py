"""
Check model metrics từ checkpoint
"""
import torch
from pathlib import Path
import json

best_model_path = Path("checkpoints/best_model.pth")
checkpoint = torch.load(best_model_path, map_location="cpu")

print("✅ Checkpoint Metadata:")
for key in ['model_name', 'model_variant', 'best_epoch', 'best_val_loss', 'metrics']:
    if key in checkpoint:
        print(f"   {key}: {checkpoint[key]}")

# Check test metrics
test_metrics_path = Path("checkpoints/test_metrics.json")
if test_metrics_path.exists():
    with open(test_metrics_path) as f:
        test_metrics = json.load(f)
    print(f"\n✅ Test Metrics:")
    for key in ['accuracy', 'top_3_accuracy', 'loss']:
        if key in test_metrics:
            print(f"   {key}: {test_metrics[key]}")

# Check history
history_path = Path("checkpoints/history.json")
if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)
    print(f"\n✅ Training History:")
    if 'val_acc' in history:
        print(f"   Best val_acc: {max(history['val_acc']):.4f}")
    if 'train_acc' in history:
        print(f"   Final train_acc: {history['train_acc'][-1] if history['train_acc'] else 0:.4f}")

# Check dataset
dataset_summary_path = Path("checkpoints/dataset_summary.json")
if dataset_summary_path.exists():
    with open(dataset_summary_path) as f:
        dataset_info = json.load(f)
    print(f"\n✅ Dataset Info:")
    if 'num_classes' in dataset_info:
        print(f"   num_classes: {dataset_info['num_classes']}")
    if 'train_samples' in dataset_info:
        print(f"   train_samples: {dataset_info['train_samples']}")
    if 'val_samples' in dataset_info:
        print(f"   val_samples: {dataset_info['val_samples']}")
    if 'test_samples' in dataset_info:
        print(f"   test_samples: {dataset_info['test_samples']}")
