"""
Script test model inference để debug tại sao kết quả thấp
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import json
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

# Model architecture
class CGIMIFColorFusionClassifier(nn.Module):
    def __init__(self, num_classes: int, color_feature_dim: int = 24, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.image_backbone = models.resnet18(weights=weights)
        in_features = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity()
        
        self.color_head = nn.Sequential(
            nn.LayerNorm(color_feature_dim),
            nn.Linear(color_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, images: torch.Tensor, color_features: torch.Tensor) -> torch.Tensor:
        image_features = self.image_backbone(images)
        color_embed = self.color_head(color_features)
        fused_features = torch.cat([image_features, color_embed], dim=1)
        return self.classifier(fused_features)


class SimpleResNet18Classifier(nn.Module):
    """Simple ResNet18 + FC layers - này là model được train thực tế."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features  # 512
        
        # Replace fc with sequential layers
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes),
        )
        self.uses_color_stream = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

best_model_path = Path("checkpoints/best_model.pth")
dataset_summary_path = Path("checkpoints/dataset_summary.json")

with open(dataset_summary_path) as f:
    dataset_info = json.load(f)
num_classes = dataset_info.get("num_classes", 108)

print(f"📦 Số lớp: {num_classes}")

# Load checkpoint FIRST
print(f"\n📥 Loading checkpoint từ {best_model_path}...")
checkpoint = torch.load(best_model_path, map_location=device)

# Check checkpoint keys
print(f"📋 Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'numpy/list'}")

if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print(f"   → Tìm thấy 'state_dict'")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"   → Tìm thấy 'model_state_dict'")
    else:
        state_dict = checkpoint
        print(f"   → Dùng toàn bộ checkpoint")
else:
    state_dict = checkpoint
    print(f"   → Checkpoint là numpy array/list")

print(f"📊 State dict keys (first 20): {list(state_dict.keys())[:20]}")
print(f"📊 State dict size: {len(state_dict)} keys")

# Build model - use SimpleResNet18Classifier (the one actually trained)
print(f"\n✅ Detecting model type from checkpoint...")
has_color_fusion = any('image_backbone.' in k for k in state_dict.keys())
print(f"   Has 'image_backbone.' prefix: {has_color_fusion}")

if has_color_fusion:
    model = CGIMIFColorFusionClassifier(num_classes=num_classes, color_feature_dim=24, pretrained=True)
    print(f"   → Using CGIMIFColorFusionClassifier")
else:
    model = SimpleResNet18Classifier(num_classes=num_classes, pretrained=True)
    print(f"   → Using SimpleResNet18Classifier (actual trained model)")

# FIX: Handle key mismatch - checkpoint keys don't have prefixes for simple model
new_state_dict = {}
for key, value in state_dict.items():
    if not has_color_fusion and not key.startswith('backbone.'):
        # Các keys từ ResNet backbone
        if key in ['conv1.weight', 'bn1.weight', 'bn1.bias'] or key.startswith(('layer', 'fc', 'bn1', 'avgpool')):
            new_key = f'backbone.{key}'
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    else:
        new_state_dict[key] = value

print(f"✅ Fixed key mapping:")
print(f"   Old keys sample: {list(state_dict.keys())[:5]}")
print(f"   New keys sample: {list(new_state_dict.keys())[:5]}")

# Load state dict
result = model.load_state_dict(new_state_dict, strict=False)
print(f"\n✅ Load state dict result:")
print(f"   Missing keys: {result.missing_keys[:5] if result.missing_keys else 'None'}...")
print(f"   Unexpected keys: {result.unexpected_keys[:5] if result.unexpected_keys else 'None'}...")

model.to(device)
model.eval()
print(f"✅ Model loaded in EVAL mode")

# Test with dummy input
print(f"\n🧪 TEST INFERENCE:")
batch_size = 1
image_tensor = torch.randn(batch_size, 3, 160, 160).to(device)

print(f"   Image shape: {image_tensor.shape}")

with torch.no_grad():
    if has_color_fusion:
        color_tensor = torch.randn(batch_size, 24).to(device)
        print(f"   Color shape: {color_tensor.shape}")
        logits = model(image_tensor, color_tensor)
    else:
        logits = model(image_tensor)

print(f"   Logits shape: {logits.shape}")
print(f"   Logits range: min={logits.min():.4f}, max={logits.max():.4f}")

probs = F.softmax(logits, dim=1)[0]
print(f"   Probs shape: {probs.shape}")
print(f"   Probs range: min={probs.min():.4f}, max={probs.max():.4f}")
print(f"   Probs sum: {probs.sum():.6f}")

top_k = torch.topk(probs, k=5)
print(f"   Top 5 classes: {top_k.indices.cpu().numpy()}")
print(f"   Top 5 probs: {[f'{p:.6f}' for p in top_k.values.cpu().numpy()]}")

# Test with real image if exists
print(f"\n📷 TEST WITH REAL IMAGE:")
test_images_dir = Path("archive (1)/public_test/pill")
if test_images_dir.exists():
    image_files = list(test_images_dir.glob("*.jpg"))[:1]
    if image_files:
        img_path = image_files[0]
        print(f"   Loading: {img_path}")
        
        # Load and preprocess
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image_rgb, (160, 160))
        
        # Get color histogram
        hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        hist = hist / (hist.sum() + 1e-8)
        color_hist = hist.astype(np.float32)
        
        print(f"   Color hist: shape={color_hist.shape}, range=[{color_hist.min():.4f}, {color_hist.max():.4f}]")
        
        # Normalize image
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        
        # To tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
        color_tensor = torch.from_numpy(color_hist).unsqueeze(0).to(device)
        
        print(f"   Image tensor: shape={image_tensor.shape}, dtype={image_tensor.dtype}")
        print(f"   Color tensor: shape={color_tensor.shape}, dtype={color_tensor.dtype}")
        
        # Inference
        with torch.no_grad():
            logits = model(image_tensor, color_tensor)
        
        probs = F.softmax(logits, dim=1)[0]
        top_k = torch.topk(probs, k=5)
        
        print(f"\n   ✨ PREDICTIONS:")
        for idx, (class_id, prob) in enumerate(zip(top_k.indices.cpu().numpy(), top_k.values.cpu().numpy())):
            print(f"      {idx+1}. Class {int(class_id):03d}: {prob*100:.2f}%")
else:
    print(f"   ❌ Test images not found at {test_images_dir}")

print(f"\n✅ Test complete!")
