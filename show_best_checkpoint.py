import torch
import json

# Load best checkpoint từ training lần 1 (9 epochs)
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')

if 'extra_metrics' in checkpoint:
    metrics = checkpoint['extra_metrics']
    print(f"\n{'='*60}")
    print(f"📊 PREVIOUS BEST MODEL (Epoch 9 Training)")
    print(f"{'='*60}\n")
    print(f"Best Epoch: {checkpoint.get('best_epoch', 'N/A')}")
    print(f"Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    print(f"Val Accuracy: {metrics.get('val_acc', 0):.2%}")
    print(f"Val Top-3 Accuracy: {metrics.get('val_top3_acc', 0):.2%}")
    print(f"Selection Metric: {metrics.get('selection_metric', 'N/A')}")
    print(f"\n{'='*60}\n")
else:
    print("❌ No metrics in checkpoint")
