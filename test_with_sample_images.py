"""
Test model với ảnh từ training set để verify preprocessing
"""
import torch
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import json
from pathlib import Path
from PIL import Image
import torch.nn as nn

# Model
class SimpleResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        
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

model = SimpleResNet18Classifier(num_classes=num_classes, pretrained=True)

checkpoint = torch.load(best_model_path, map_location=device)
state_dict = checkpoint['state_dict']

# Fix key mapping
new_state_dict = {}
for key, value in state_dict.items():
    if not key.startswith('backbone.'):
        if key in ['conv1.weight', 'bn1.weight', 'bn1.bias'] or key.startswith(('layer', 'fc', 'bn1', 'avgpool')):
            new_key = f'backbone.{key}'
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    else:
        new_state_dict[key] = value

model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

print(f"✅ Model loaded\n")

# Helper function to test image
def test_image(img_path):
    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"   ❌ Failed to load {img_path.name}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(image_rgb, (160, 160))
    
    # Normalize
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = (image_normalized - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    
    # To tensor (IMPORTANT: cast to float32)
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).to(device).float()
    
    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
    
    probs = F.softmax(logits, dim=1)[0]
    top_k = torch.topk(probs, k=5)
    
    print(f"   {img_path.name}:")
    for idx, (class_id, prob) in enumerate(zip(top_k.indices.cpu().numpy(), top_k.values.cpu().numpy())):
        print(f"      {idx+1}. Class {int(class_id):03d}: {prob*100:.2f}%")

# Test with training set
train_pill_dir = Path("archive (1)/public_train/pill/image")
if train_pill_dir.exists():
    print(f"📷 Testing with training set images:\n")
    image_files = list(train_pill_dir.glob("*.jpg"))[:5]
    for img_path in image_files:
        test_image(img_path)
    print()
else:
    print(f"❌ Training images not found at {train_pill_dir}\n")

# Test with test set
test_pill_dir = Path("archive (1)/public_test/pill/image")
if test_pill_dir.exists():
    print(f"📷 Testing with test set images:\n")
    image_files = list(test_pill_dir.glob("*.jpg"))[:5]
    for img_path in image_files:
        test_image(img_path)
    print()
else:
    print(f"❌ Test images not found at {test_pill_dir}\n")
