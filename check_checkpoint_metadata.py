"""
Check checkpoint metadata to see which model was actually trained
"""
import torch
from pathlib import Path
import json

best_model_path = Path("checkpoints/best_model.pth")
checkpoint = torch.load(best_model_path, map_location="cpu")

print("✅ Checkpoint metadata:")
for key in ['model_name', 'model_variant', 'best_epoch', 'best_val_loss']:
    if key in checkpoint:
        print(f"   {key}: {checkpoint[key]}")

print(f"\n✅ State dict keys (ResNet-like):")
state_dict = checkpoint['state_dict']
print(f"   Has 'image_backbone.' prefix: {any(k.startswith('image_backbone.') for k in state_dict.keys())}")
print(f"   Has 'color_head' modules: {any('color_head' in k for k in state_dict.keys())}")
print(f"   Has 'classifier' modules: {any('classifier' in k for k in state_dict.keys())}")
print(f"   Has 'fc' modules: {any('fc' in k for k in state_dict.keys())}")

print(f"\n✅ Sample keys:")
for key in list(state_dict.keys())[:10]:
    print(f"   {key}")
