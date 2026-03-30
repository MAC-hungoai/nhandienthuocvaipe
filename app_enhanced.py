"""
✨ ENHANCED UI VERSION - GIAO DIỆN MODERN & PROFESSIONAL
"""
import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
from typing import Dict, Tuple, Optional
import torchvision.models as models
import torch.nn.functional as F

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SimpleResNet18Classifier(nn.Module):
    """Simple ResNet18 + FC layers"""
    
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
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


class CGIMIFColorFusionClassifier(nn.Module):
    """Model phân loại viên thuốc dùng Color Fusion"""
    
    def __init__(self, num_classes: int, color_feature_dim: int = 24, pretrained: bool = True) -> None:
        super().__init__()
        
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.image_backbone = models.resnet18(weights=weights)
        in_features = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity()
        self.uses_color_stream = True
        
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

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="💊 VAIPE - Phân Tích Viên Thuốc",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# ============================================================================
# ✨ ENHANCED CUSTOM CSS - MODERN DESIGN
# ============================================================================

st.markdown("""
<style>
:root {
    --primary: #00D9FF;
    --secondary: #FF1493;
    --accent: #9D4EDD;
    --dark-bg: #0a0e27;
    --darker-bg: #050815;
    --card-bg: #1a1f3a;
    --card-hover: #252d4a;
    --text-main: #FFFFFF;
    --text-sub: #B0B8C1;
    --success: #00FF88;
    --warning: #FFB800;
    --danger: #FF3366;
    --light-shadow: rgba(0, 217, 255, 0.15);
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* MAIN BACKGROUND - GRADIENT UNIVERSE                                    */
/* ═══════════════════════════════════════════════════════════════════════ */
body, .main, .stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e27 0%, #1a0033 50%, #0f1933 100%) !important;
    color: #FFFFFF !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* SIDEBAR - GLASS MORPHISM                                               */
/* ═══════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: rgba(26, 31, 58, 0.8) !important;
    backdrop-filter: blur(10px) !important;
    border-right: 2px solid rgba(0, 217, 255, 0.3) !important;
    box-shadow: inset -1px 0 rgba(0, 217, 255, 0.1) !important;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* TYPOGRAPHY - BOLD & MODERN                                             */
/* ═══════════════════════════════════════════════════════════════════════ */
h1 {
    background: linear-gradient(90deg, #00D9FF 0%, #FF1493 50%, #9D4EDD 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 56px !important;
    font-weight: 900 !important;
    letter-spacing: 1px !important;
    margin: 30px 0 15px 0 !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}

h2 {
    color: #FFFFFF !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    margin: 25px 0 15px 0 !important;
    border-left: 4px solid #00D9FF !important;
    padding-left: 12px !important;
}

h3 {
    color: #00D9FF !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    margin: 18px 0 10px 0 !important;
}

p, label, span, .stMarkdown, .stMarkdown * {
    color: #FFFFFF !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* BUTTONS - NEON GLOW + PREMIUM EFFECTS                                  */
/* ═══════════════════════════════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #00D9FF 0%, #00F5FF 50%, #00D9FF 100%) !important;
    color: #000000 !important;
    border: 2px solid rgba(0, 245, 255, 0.5) !important;
    border-radius: 12px !important;
    font-weight: 900 !important;
    font-size: 16px !important;
    padding: 16px 32px !important;
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    box-shadow: 
        0 0 20px rgba(0, 217, 255, 0.4),
        0 4px 20px rgba(0, 217, 255, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
    transition: left 0.5s !important;
}

.stButton > button:hover {
    transform: translateY(-4px) scale(1.02) !important;
    box-shadow: 
        0 0 40px rgba(0, 217, 255, 0.6),
        0 8px 40px rgba(0, 217, 255, 0.5),
        0 0 60px rgba(0, 217, 255, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    background: linear-gradient(135deg, #00F5FF 0%, #00D9FF 50%, #00F5FF 100%) !important;
    border-color: rgba(0, 245, 255, 0.8) !important;
}

.stButton > button:hover::before {
    left: 100% !important;
}

.stButton > button:active {
    transform: translateY(-2px) scale(0.98) !important;
    box-shadow: 
        0 0 30px rgba(0, 217, 255, 0.5),
        0 4px 20px rgba(0, 217, 255, 0.3),
        inset 0 2px 5px rgba(0, 0, 0, 0.2) !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* CARDS & CONTAINERS - GLASS EFFECT                                      */
/* ═══════════════════════════════════════════════════════════════════════ */
.stInfo, .stWarning, .stError, .stSuccess {
    background: rgba(26, 31, 58, 0.7) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 217, 255, 0.2) !important;
    padding: 16px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

.stInfo {
    border-left: 4px solid #00D9FF !important;
}

.stWarning {
    border-left: 4px solid #FFB800 !important;
}

.stError {
    border-left: 4px solid #FF3366 !important;
}

.stSuccess {
    border-left: 4px solid #00FF88 !important;
}

.stInfo *, .stWarning *, .stError *, .stSuccess * {
    color: #FFFFFF !important;
    background: transparent !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* TABS - MODERN                                                          */
/* ═══════════════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(26, 31, 58, 0.5) !important;
    border-bottom: 2px solid rgba(0, 217, 255, 0.2) !important;
    gap: 24px !important;
    padding: 12px 0 !important;
}

.stTabs [data-baseweb="tab"] {
    color: #B0B8C1 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 8px 0 !important;
    transition: all 0.3s !important;
}

.stTabs [aria-selected="true"] {
    color: #00D9FF !important;
    border-bottom: 3px solid #00D9FF !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* FILE UPLOADER - STYLISH DROP ZONE                                      */
/* ═══════════════════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(26, 31, 58, 0.9), rgba(26, 31, 58, 0.7)) !important;
    border: 2px dashed #00D9FF !important;
    border-radius: 12px !important;
    padding: 32px 24px !important;
    transition: all 0.3s !important;
}

[data-testid="stFileUploader"]:hover {
    background: linear-gradient(135deg, rgba(26, 31, 58, 0.95), rgba(26, 31, 58, 0.85)) !important;
    border-color: #00F5FF !important;
    box-shadow: inset 0 0 30px rgba(0, 217, 255, 0.2), 0 0 20px rgba(0, 217, 255, 0.1) !important;
}

[data-testid="stFileUploader"] * {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

[data-testid="stFileUploader"] p {
    font-size: 18px !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    line-height: 1.6 !important;
}

[data-testid="stFileUploader"] span {
    color: #00D9FF !important;
    font-weight: 800 !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* INPUTS & FORMS                                                         */
/* ═══════════════════════════════════════════════════════════════════════ */
input, textarea, .stSelectbox, .stMultiSelect, .stSlider {
    background: rgba(26, 31, 58, 0.8) !important;
    color: #FFFFFF !important;
    border: 2px solid rgba(0, 217, 255, 0.2) !important;
    border-radius: 8px !important;
    padding: 10px !important;
    transition: all 0.3s !important;
}

input:focus, textarea:focus {
    border-color: #00D9FF !important;
    box-shadow: 0 0 20px rgba(0, 217, 255, 0.3) !important;
    outline: none !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* DATAFRAME & TABLES                                                     */
/* ═══════════════════════════════════════════════════════════════════════ */
.stDataFrame {
    background: rgba(26, 31, 58, 0.7) !important;
    border: 1px solid rgba(0, 217, 255, 0.2) !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
}

.stDataFrame th {
    background: linear-gradient(90deg, rgba(0, 217, 255, 0.2), rgba(157, 78, 221, 0.2)) !important;
    color: #00D9FF !important;
    font-weight: 800 !important;
    border-bottom: 2px solid rgba(0, 217, 255, 0.3) !important;
}

.stDataFrame td {
    color: #FFFFFF !important;
    border-bottom: 1px solid rgba(0, 217, 255, 0.1) !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* PLOTLY CHARTS - DARK THEME                                             */
/* ═══════════════════════════════════════════════════════════════════════ */
.plotly-graph-div {
    background: rgba(26, 31, 58, 0.5) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 217, 255, 0.15) !important;
    padding: 16px !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* SPECIAL ELEMENTS                                                       */
/* ═══════════════════════════════════════════════════════════════════════ */
.prediction-card {
    background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(157, 78, 221, 0.1)) !important;
    border: 2px solid rgba(0, 217, 255, 0.3) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(10px) !important;
}

.metric-value {
    color: #00D9FF !important;
    font-size: 42px !important;
    font-weight: 900 !important;
    letter-spacing: 1px !important;
}

.metric-label {
    color: #B0B8C1 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* SCROLL BAR - CUSTOM                                                    */
/* ═══════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar {
    width: 10px !important;
}

::-webkit-scrollbar-track {
    background: rgba(26, 31, 58, 0.4) !important;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00D9FF, #9D4EDD) !important;
    border-radius: 10px !important;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #00F5FF, #B040FF) !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* ANIMATIONS                                                             */
/* ═══════════════════════════════════════════════════════════════════════ */
@keyframes fade-in {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes glow-pulse {
    0%, 100% {
        box-shadow: 0 4px 20px rgba(0, 217, 255, 0.4);
    }
    50% {
        box-shadow: 0 4px 30px rgba(0, 217, 255, 0.6);
    }
}

.stApp {
    animation: fade-in 0.5s ease-out !important;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/* RESPONSIVE                                                             */
/* ═══════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    h1 { font-size: 40px !important; }
    h2 { font-size: 24px !important; }
    p, label { font-size: 15px !important; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS (cached)
# ============================================================================

@st.cache_resource
def load_models():
    """Load pretrained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_model_path = Path("checkpoints/best_model.pth")
    dataset_summary_path = Path("checkpoints/dataset_summary.json")
    
    if not best_model_path.exists():
        st.error("❌ Model file not found!")
        st.stop()
    
    with open(dataset_summary_path) as f:
        dataset_info = json.load(f)
    
    num_classes = dataset_info.get("num_classes", 108)
    class_dist = dataset_info.get("class_distribution", {})
    
    checkpoint = torch.load(best_model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    if any('image_backbone.' in k for k in state_dict.keys()):
        model = CGIMIFColorFusionClassifier(num_classes=num_classes, color_feature_dim=24, pretrained=True)
        has_color_fusion = True
    else:
        model = SimpleResNet18Classifier(num_classes=num_classes, pretrained=True)
        has_color_fusion = False
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if not has_color_fusion and not key.startswith('backbone.'):
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
    
    class_names = {i: f"Thuốc_{i:03d}" for i in range(num_classes)}
    
    try:
        for class_id_str in class_dist.keys():
            class_id = int(class_id_str)
            class_names[class_id] = f"Viên Thuốc {class_id}"
    except:
        pass
    
    return {
        "model": model,
        "device": device,
        "num_classes": num_classes,
        "class_names": class_names,
        "class_dist": class_dist,
        "has_color_fusion": has_color_fusion,
    }

def process_image(image_array: np.ndarray) -> np.ndarray:
    """Prepare image for model"""
    from torchvision import transforms
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    pil_image = Image.fromarray(image_array.astype(np.uint8))
    tensor = transform(pil_image)
    
    return tensor.unsqueeze(0)

def extract_color_histogram(image: np.ndarray, bins: int = 8) -> np.ndarray:
    """Extract HSV histogram"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    hist = hist / (hist.sum() + 1e-8)
    return hist.astype(np.float32)

def predict_pill(image: np.ndarray, model_info: Dict) -> Dict:
    """Predict pill class"""
    try:
        model = model_info["model"]
        device = model_info["device"]
        class_names = model_info["class_names"]
        has_color_fusion = model_info.get("has_color_fusion", False)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = process_image(image_rgb).to(device)
        
        with torch.no_grad():
            if has_color_fusion:
                color_hist = extract_color_histogram(image_rgb)
                color_tensor = torch.from_numpy(color_hist).unsqueeze(0).to(device).float()
                logits = model(image_tensor.float(), color_tensor)
            else:
                logits = model(image_tensor.float())
        
        probs = F.softmax(logits, dim=1)[0]
        top_k = torch.topk(probs, k=min(5, len(probs)))
        top_classes = top_k.indices.cpu().numpy()
        top_probs = top_k.values.cpu().numpy()
        
        pred_class = int(top_classes[0])
        confidence = float(top_probs[0])
        
        return {
            "class_id": pred_class,
            "class_name": class_names.get(pred_class, f"Thuốc_{pred_class}"),
            "confidence": confidence,
            "probabilities": {str(int(c)): float(p) for c, p in zip(top_classes, top_probs)},
            "top_5": [(int(c), class_names.get(int(c), f"Thuốc_{c}"), float(p)) 
                      for c, p in zip(top_classes[:5], top_probs[:5])],
        }
    except Exception as e:
        st.error(f"❌ Lỗi dự đoán: {str(e)}")
        return None

# ============================================================================
# HEADER - PROFESSIONAL BRANDING
# ============================================================================

st.markdown("""
<div style='text-align: center; padding: 40px 20px; margin-bottom: 30px;'>
    <h1 style='font-size: 60px; margin-bottom: 5px;'>💊 VAIPE</h1>
    <p style='font-size: 18px; color: #00D9FF; font-weight: 600; letter-spacing: 3px; margin-bottom: 10px;'>
        INTELLIGENT PHARMACEUTICAL RECOGNITION
    </p>
    <p style='font-size: 14px; color: #B0B8C1;'>
        Công nghệ AI nhận diện viên thuốc chính xác, nhanh chóng & đáng tin cậy
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["📸 Chụp & Phân Tích", "📊 Phân Tích Hàng Loạt", "❓ Trợ Giúp"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: SINGLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<h2>🔍 Phân Tích Ảnh Đơn Lẻ</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<h3>⬆️ Tải Ảnh Viên Thuốc</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Chọn ảnh JPG/PNG",
            type=["jpg", "jpeg", "png", "bmp"],
            key="single_upload"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="ảnh đã tải", use_column_width=True)
            
            if st.button("🚀 Phân Tích Ngay", key="analyze_single", use_container_width=True):
                with st.spinner("⏳ Đang phân tích..."):
                    model_info = load_models()
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    result = predict_pill(image_cv, model_info)
                    
                    if result:
                        st.session_state.last_prediction = result
                        st.success("✅ Phân tích thành công!")
    
    with col2:
        if st.session_state.last_prediction:
            result = st.session_state.last_prediction
            
            st.markdown("<h3>📊 Kết Quả Phân Tích</h3>", unsafe_allow_html=True)
            
            # Main prediction card
            st.markdown(f"""
            <div class='prediction-card'>
                <div style='text-align: center;'>
                    <p style='font-size: 14px; color: #B0B8C1; margin-bottom: 5px;'>🎯 NGUYÊN TỬ CHÍNH</p>
                    <p style='font-size: 28px; font-weight: 900; color: #00D9FF; margin-bottom: 15px;'>
                        {result['class_name']}
                    </p>
                    <p style='font-size: 14px; color: #B0B8C1;'>Độ tin cậy</p>
                    <p style='font-size: 36px; font-weight: 900; background: linear-gradient(90deg, #00D9FF, #FF1493); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;'>
                        {result['confidence']*100:.1f}%
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top-5
            st.markdown("<h3>🥇 Top-5 Dự Đoán</h3>", unsafe_allow_html=True)
            
            top5_data = []
            for idx, (class_id, name, prob) in enumerate(result['top_5'], 1):
                top5_data.append({
                    "🏆 Xếp Hạng": f"#{idx}",
                    "💊 Loại Thuốc": name,
                    "📈 Xác Suất": f"{prob*100:.2f}%"
                })
            
            df = pd.DataFrame(top5_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Charts
            st.markdown("<h3>📈 Biểu Đồ Phân Tích</h3>", unsafe_allow_html=True)
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Gauge chart
                fig_gauge = go.Figure(data=[go.Indicator(
                    mode="gauge+number+delta",
                    value=result['confidence']*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence Score", 'font': {'color': '#FFFFFF'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickfont': {'color': '#B0B8C1'}},
                        'bar': {'color': "#00D9FF"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(255, 51, 102, 0.2)"},
                            {'range': [50, 80], 'color': "rgba(255, 184, 0, 0.2)"},
                            {'range': [80, 100], 'color': "rgba(0, 255, 136, 0.2)"},
                        ],
                    }
                )])
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(26, 31, 58, 0.5)',
                    font={'color': '#FFFFFF', 'family': 'Arial'},
                    margin={'l': 0, 'r': 0, 't': 30, 'b': 0}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with chart_col2:
                # Bar chart
                top_classes = [name for _, name, _ in result['top_5']]
                top_probs = [prob*100 for _, _, prob in result['top_5']]
                
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=top_probs,
                        y=top_classes,
                        orientation='h',
                        marker=dict(
                            color=top_probs,
                            colorscale='Viridis',
                        )
                    )
                ])
                fig_bar.update_layout(
                    title="Top-5 Probabilities",
                    xaxis_title="Probability (%)",
                    yaxis_title=None,
                    paper_bgcolor='rgba(26, 31, 58, 0.5)',
                    plot_bgcolor='rgba(26, 31, 58, 0.3)',
                    font={'color': '#FFFFFF'},
                    margin={'l': 120, 'r': 20, 't': 40, 'b': 40}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<h2>⚡ Phân Tích Hàng Loạt (Batch)</h2>", unsafe_allow_html=True)
    st.info("💡 Tải nhiều ảnh cùng lúc để phân tích nhanh chóng")
    
    uploaded_files = st.file_uploader(
        "Chọn nhiều ảnh",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files and st.button("🚀 Phân Tích Tất Cả", use_container_width=True):
        model_info = load_models()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_list = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"⏳ Đang xử lý: {idx+1}/{len(uploaded_files)}")
            
            image = Image.open(uploaded_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            result = predict_pill(image_cv, model_info)
            
            if result:
                results_list.append({
                    "📁 Tên File": uploaded_file.name,
                    "💊 Kết Quả": result['class_name'],
                    "📊 Tự Tin": f"{result['confidence']*100:.2f}%"
                })
            
            progress_bar.progress((idx+1)/len(uploaded_files))
        
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Hoàn thành! Đã phân tích {len(results_list)} ảnh")
        
        # Show results
        results_df = pd.DataFrame(results_list)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Export
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Tải CSV",
            csv,
            "results.csv",
            "text/csv",
            use_container_width=True
        )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: HELP
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<h2>❓ Hướng Dẫn & FAQ</h2>", unsafe_allow_html=True)
    
    with st.expander("🤔 Làm sao để có kết quả tốt nhất?"):
        st.markdown("""
        ✅ **Mẹo chụp ảnh tốt:**
        - Chụp từ trên xuống viên thuốc
        - Ánh sáng tự nhiên, không chói không tối
        - Ảnh rõ nét, không mờ
        - Viên thuốc chiếm ~70% khung hình
        """)
    
    with st.expander("📊 Độ tin cậy bao nhiêu là OK?"):
        st.markdown("""
        - 🔴 < 30%: Không đủ tin cậy
        - 🟡 30-60%: Tạm được, nên xem top-5
        - 🟢 60-80%: Tốt
        - 🟢🟢 > 80%: Rất tốt
        """)
    
    with st.expander("🔧 Vấn đề & Giải Pháp"):
        st.markdown("""
        **❌ Kết quả thấp → Chụp lại ảnh:**
        - Ảnh phải rõ ràng, không mờ
        - Ánh sáng đủ
        
        **❌ Model chức năng bất thường:**
        - Làm tươi lại trang (Ctrl+R)
        - Xóa cache Streamlit
        """)
    
    st.markdown("<h3>📱 Công Nghệ Sử Dụng</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **Model:** ResNet18 + PyTorch
   - **Framework:** Streamlit
    - **GPU:** CUDA enabled
    - **Classes:** 108 loại thuốc
    """)
