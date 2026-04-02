"""
◆═══════════════════════════════════════════════════════════════════════════◆
║                   💊 HỆ THỐNG PHÂN TÍCH VIÊN THUỐC                      ║
║                   VAIPE - Thông Minh & Chính Xác                       ║
◆═══════════════════════════════════════════════════════════════════════════◆

Ứng dụng web phân tích ảnh viên thuốc tự động:
- 📸 Chụp hoặc tải lên ảnh viên thuốc
- 🤖 AI nhận diện loại thuốc + thành phần
- 📊 Xem chi tiết + biểu đồ phân tích
- 💾 Lưu kết quả dưới dạng PDF/JSON

Thiết kế: 🎨 Đẹp mắt | ⚡ Nhanh | 🎯 Chính xác
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import tempfile
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from html import escape
import pandas as pd
from typing import Dict, Tuple, Optional
import torchvision.models as models
from demo_infer import (
    DEFAULT_DEMO_CLASSIFIER_CHECKPOINT,
    DEFAULT_DEMO_DETECTOR_CHECKPOINT,
    build_app_response,
)
from detection_test import predict_single_image
from detection_utils import DEFAULT_DETECTION_SCORE_THRESHOLD, load_detection_checkpoint
from knowledge_graph import (
    DEFAULT_KG_ANCHOR_WEIGHT,
    DEFAULT_KG_CANDIDATES,
    DEFAULT_KG_CONTEXT_WEIGHT,
    DEFAULT_KG_MAX_ANCHOR_PROBABILITY,
    DEFAULT_KG_MAX_DETECTOR_SCORE,
    DEFAULT_KG_MIN_CANDIDATE_PROBABILITY,
    DEFAULT_KG_SELECTIVE_OVERRIDE,
    DEFAULT_KG_VISUAL_WEIGHT,
    DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR,
    DEFAULT_KNOWLEDGE_GRAPH_PATH,
    build_label_display_names,
    classify_crop_candidates,
    load_classifier_checkpoint,
    load_or_build_knowledge_graph,
)

# ============================================================================
# MODEL ARCHITECTURE - ResNet18 Simple (Fallback - được dùng trong training)
# ============================================================================

class SimpleResNet18Classifier(nn.Module):
    """Simple ResNet18 + FC layers - này là model được train thực tế."""
    
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
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


class CGIMIFColorFusionClassifier(nn.Module):
    """Model phân loại viên thuốc dùng Color Fusion."""
    
    def __init__(
        self,
        num_classes: int,
        color_feature_dim: int = 24,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        
        # IMAGE STREAM (ResNet18)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.image_backbone = models.resnet18(weights=weights)
        in_features = self.image_backbone.fc.in_features  # 512
        self.image_backbone.fc = nn.Identity()
        self.uses_color_stream = True
        
        # COLOR STREAM
        self.color_head = nn.Sequential(
            nn.LayerNorm(color_feature_dim),
            nn.Linear(color_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        
        # CLASSIFICATION HEAD
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features + 64, 256),  # 512 + 64 = 576
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, images: torch.Tensor, color_features: torch.Tensor) -> torch.Tensor:
        image_features = self.image_backbone(images)  # [B, 512]
        color_embed = self.color_head(color_features)  # [B, 24] → [B, 64]
        fused_features = torch.cat([image_features, color_embed], dim=1)  # [B, 576]
        return self.classifier(fused_features)  # [B, num_classes]

# ============================================================================
# CẤU HÌNH TRANG
# ============================================================================

st.set_page_config(
    page_title="💊 Phân Tích Viên Thuốc",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

DEFAULT_STREAMLIT_OUTPUT_DIR = Path("outputs") / "streamlit_runs"
MODEL_BUNDLE_CACHE_VERSION = "detector-import-fix-v1"
RUNTIME_SETUP_HINT = "Xem docs/SETUP.md và checkpoints/README.md để chuẩn bị đúng dependency và artifact."

# ============================================================================
# CUSTOM CSS - GIAO DIỆN HIỆN ĐẠI + DỄ ĐỌC
# ============================================================================

st.markdown("""
<style>
:root {
    --primary: #00D9FF;            /* Cyan sáng */
    --secondary: #FF1493;          /* Pink đỏ */
    --dark-bg: #0f1419;            /* Dark Navy */
    --card-bg: #1a2332;            /* Card tối */
    --text-main: #FFFFFF;          /* Trắng */
    --text-sub: #B0B8C1;           /* Xám nhạt */
    --success: #2ECC71;            /* Xanh */
    --warning: #F39C12;            /* Cam */
    --danger: #E74C3C;             /* Đỏ */
}

/* Nền chính */
body, .main, .stApp {
    background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%) !important;
    color: #FFFFFF !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1419 0%, #1a2332 100%) !important;
    border-right: 3px solid var(--primary) !important;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

.st-emotion-cache-1d391yk {
    background: linear-gradient(180deg, #0f1419 0%, #1a2332 100%) !important;
    border-right: 3px solid var(--primary) !important;
}

.st-emotion-cache-1d391yk * {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Sidebar headers */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-size: 22px !important;
    font-weight: 800 !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* Sidebar sliders */
[data-testid="stSidebar"] [data-baseweb="slider"] {
    background: rgba(255, 255, 255, 0.1) !important;
}

/* Sidebar inputs */
[data-testid="stSidebar"] input {
    background: var(--card-bg) !important;
    color: #FFFFFF !important;
    border: 2px solid var(--primary) !important;
}

/* Sidebar checkbox */
[data-testid="stSidebar"] .stCheckbox {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] .stCheckbox label {
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* Tiêu đề - LỚNN + BOLD */
h1 {
    color: #FFFFFF !important;
    font-size: 56px !important;
    font-weight: 900 !important;
    margin: 30px 0 10px 0 !important;
    letter-spacing: 2px !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5) !important;
}

h2 {
    color: #FFFFFF !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    margin: 20px 0 10px 0 !important;
    border-bottom: 3px solid var(--primary);
    padding-bottom: 10px;
}

h3 {
    color: #FFFFFF !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin: 15px 0 8px 0 !important;
}

p, label, span, .stMarkdown, .stMarkdown * {
    color: #FFFFFF !important;
    font-size: 18px !important;
    line-height: 1.8 !important;
}

/* Nút bấm */
.stButton > button {
    background: linear-gradient(90deg, var(--primary), #00F5FF) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
    font-size: 18px !important;
    padding: 14px 28px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(0, 217, 255, 0.5) !important;
}

/* Card */
.stInfo {
    background: linear-gradient(135deg, #1a2332, #16202A) !important;
    border-left: 4px solid var(--primary) !important;
    border-radius: 6px !important;
    padding: 15px !important;
    color: #FFFFFF !important;
}

.stInfo p, .stInfo span, .stInfo h3, .stInfo h4, .stInfo * {
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* Sidebar Info */
[data-testid="stSidebar"] .stInfo {
    background: linear-gradient(135deg, #1a2332, #16202A) !important;
    border: 2px solid var(--primary) !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

[data-testid="stSidebar"] .stInfo * {
    color: #FFFFFF !important;
}

.stWarning {
    background: var(--card-bg) !important;
    border-left: 4px solid var(--warning) !important;
    color: #FFFFFF !important;
}

.stError {
    background: var(--card-bg) !important;
    border-left: 4px solid var(--danger) !important;
    color: #FFFFFF !important;
}

.stSuccess {
    background: var(--card-bg) !important;
    border-left: 4px solid var(--success) !important;
    color: #FFFFFF !important;
}

/* Tab */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid var(--primary) !important;
    gap: 20px !important;
}

.stTabs [data-baseweb="tab"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 18px !important;
}

.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom: 3px solid var(--primary) !important;
}

/* File uploader */
.st-emotion-cache-u16kxy {
    border: 2px dashed var(--primary) !important;
    border-radius: 8px !important;
    background: rgba(0, 217, 255, 0.08) !important;
}

.st-emotion-cache-u16kxy * {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* File uploader labels */
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #1a2332, #16202A) !important;
    border: 2px dashed var(--primary) !important;
    color: #FFFFFF !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Upload text */
.st-emotion-cache-u16kxy p, 
[data-testid="stFileUploaderDropzone"] p {
    color: #FFFFFF !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* Browse button */
.st-emotion-cache-u16kxy button,
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(90deg, var(--primary), #00F5FF) !important;
    color: #000000 !important;
    font-weight: 800 !important;
}

/* Bảng dữ liệu */
.stDataFrame {
    background: var(--card-bg) !important;
    color: #FFFFFF !important;
}

.stDataFrame th, .stDataFrame td {
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

/* Metric */
.metric-value {
    color: var(--primary) !important;
    font-size: 36px !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ui-font-family: "Manrope", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}

html, body, [class*="st-"], button, input, textarea, select, label {
    font-family: var(--ui-font-family) !important;
}

header[data-testid="stHeader"] {
    position: relative !important;
    background: rgba(247, 251, 255, 0.82) !important;
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(215, 231, 246, 0.92);
}

header[data-testid="stHeader"]::before {
    content: "" !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 240px !important;
    height: 100% !important;
    background: rgba(247, 251, 255, 0.98) !important;
    border-bottom: 1px solid rgba(215, 231, 246, 0.92) !important;
    pointer-events: none !important;
    z-index: 99990 !important;
}

[data-testid="stToolbar"] {
    right: 1rem !important;
}

.block-container {
    max-width: 1380px !important;
    padding-top: 0.45rem !important;
    padding-bottom: 2.6rem !important;
}

.hero-panel {
    position: relative;
    overflow: hidden;
    padding: 24px 28px !important;
    border-radius: 28px !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.hero-panel::before {
    content: "";
    position: absolute;
    left: -58px;
    bottom: -96px;
    width: 210px;
    height: 210px;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.18), rgba(255, 255, 255, 0) 72%);
}

.hero-panel::after {
    content: "";
    position: absolute;
    right: -54px;
    top: -76px;
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.32), rgba(255, 255, 255, 0) 70%);
}

.hero-title {
    position: relative;
    z-index: 1;
    max-width: 700px;
    font-size: 38px !important;
    letter-spacing: -0.04em;
}

.hero-subtitle {
    position: relative;
    z-index: 1;
    max-width: 640px;
    font-size: 15px !important;
}

.hero-chip-row {
    position: relative;
    z-index: 1;
    margin-top: 22px !important;
}

.hero-chip {
    padding: 9px 14px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
}

.quick-stat {
    position: relative;
    overflow: hidden;
    min-height: 132px;
    padding: 18px 18px 16px !important;
    border-radius: 22px !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.quick-stat::before {
    content: "";
    position: absolute;
    top: 0;
    left: 22px;
    width: 52px;
    height: 4px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--ui-primary), #abd4ff);
}

.section-intro {
    padding: 16px 18px !important;
    border-radius: 22px !important;
    margin-bottom: 14px !important;
}

.result-shell {
    position: relative;
    overflow: hidden;
    padding: 22px 22px !important;
    border-left: 4px solid var(--ui-primary) !important;
}

.result-shell::after {
    content: "";
    position: absolute;
    right: -34px;
    top: -34px;
    width: 146px;
    height: 146px;
    background: radial-gradient(circle, rgba(74, 144, 226, 0.18), rgba(74, 144, 226, 0) 70%);
}

.confidence-rail {
    margin-top: 18px;
    height: 10px;
    border-radius: 999px;
    background: #edf4fc;
    border: 1px solid #d7e7f7;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--ui-primary), #8fc6ff);
    box-shadow: 0 6px 16px rgba(74, 144, 226, 0.22);
}

div[data-testid="stImage"] img {
    border-radius: 26px;
    border: 1px solid #dceafa;
    box-shadow: 0 20px 38px rgba(37, 94, 154, 0.14);
}

div[data-testid="stPlotlyChart"] {
    background: linear-gradient(180deg, #ffffff, #fbfdff);
    border: 1px solid #dceafa;
    border-radius: 24px;
    padding: 10px 12px 4px;
    box-shadow: 0 16px 32px rgba(39, 86, 140, 0.08);
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 180px;
    border: 1.5px dashed #aacdf5 !important;
    background: linear-gradient(180deg, #fbfdff, #f2f8ff) !important;
    border-radius: 26px !important;
    padding: 24px 20px !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #6eaeea !important;
    box-shadow: 0 18px 32px rgba(74, 144, 226, 0.12) !important;
    transform: translateY(-1px) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px !important;
    border: none !important;
}

.stTabs [data-baseweb="tab"] {
    border: 1px solid #d8e8f8 !important;
    border-bottom: none !important;
    background: rgba(255, 255, 255, 0.78) !important;
    box-shadow: 0 8px 20px rgba(35, 84, 142, 0.05);
    transition: background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
}

.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    box-shadow: inset 0 -3px 0 var(--ui-primary), 0 14px 28px rgba(35, 84, 142, 0.08) !important;
}

div[data-testid="stMetric"] {
    min-height: 114px;
    padding: 18px 18px 16px !important;
    border-radius: 20px !important;
}

.metric-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em;
}

.metric-badge.positive {
    background: rgba(34, 197, 94, 0.12);
    color: #16803d !important;
}

.metric-badge.negative {
    background: rgba(239, 68, 68, 0.12);
    color: #b42318 !important;
}

.metric-badge.neutral {
    background: rgba(74, 144, 226, 0.12);
    color: var(--ui-primary-strong) !important;
}

.progress-card,
.insight-card,
.sidebar-card-min {
    background: linear-gradient(180deg, #ffffff, #fbfdff);
    border: 1px solid #dceafa;
    border-radius: 22px;
    box-shadow: 0 14px 28px rgba(39, 86, 140, 0.08);
}

.progress-card,
.insight-card {
    padding: 20px 22px;
}

.progress-card-title,
.insight-title {
    margin: 0;
    color: var(--ui-text) !important;
    font-size: 16px !important;
    font-weight: 800 !important;
}

.progress-card-value {
    margin-top: 8px;
    color: var(--ui-text) !important;
    font-size: 34px !important;
    font-weight: 900 !important;
    line-height: 1.05 !important;
}

.progress-card-note,
.insight-body {
    margin-top: 10px;
    color: var(--ui-muted) !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
}

.progress-meta {
    margin-top: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}

.progress-rail {
    margin-top: 14px;
    height: 10px;
    border-radius: 999px;
    background: #edf4fc;
    border: 1px solid #d7e7f7;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--ui-primary), #8fc6ff);
}

.sidebar-card-min {
    padding: 16px 16px 14px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.22);
    box-shadow: none;
}

.sidebar-card-min * {
    color: #ffffff !important;
}

.sidebar-card-min .sidebar-title {
    font-size: 13px !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.9;
}

.sidebar-card-min .sidebar-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 10px;
    margin-top: 12px;
}

.sidebar-card-min .sidebar-label {
    font-size: 13px !important;
    opacity: 0.86;
}

.sidebar-card-min .sidebar-value {
    font-size: 18px !important;
    font-weight: 900 !important;
}

.sidebar-badge {
    display: inline-flex;
    margin-top: 8px;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 800 !important;
}

.stDataFrame {
    border-radius: 22px !important;
    overflow: hidden;
}

[data-testid="collapsedControl"] {
    position: fixed !important;
    top: 76px !important;
    left: 18px !important;
    z-index: 100000 !important;
}

[data-testid="collapsedControl"] button {
    outline: none !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container {
    max-width: 1280px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

.hero-panel {
    position: relative;
    overflow: hidden;
    padding: 28px 30px;
    margin: 8px 0 26px 0;
    border-radius: 24px;
    border: 1px solid rgba(0, 217, 255, 0.24);
    background:
        radial-gradient(circle at top right, rgba(0, 217, 255, 0.16), transparent 35%),
        radial-gradient(circle at left center, rgba(255, 20, 147, 0.15), transparent 32%),
        linear-gradient(145deg, rgba(16, 24, 39, 0.98), rgba(23, 34, 52, 0.92));
    box-shadow: 0 24px 50px rgba(0, 0, 0, 0.28);
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #9EE7FF !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.hero-title {
    margin: 16px 0 8px 0;
    color: #FFFFFF;
    font-size: 42px;
    line-height: 1.08;
    font-weight: 900;
    letter-spacing: -0.03em;
}

.hero-subtitle {
    max-width: 780px;
    margin: 0;
    color: #D6DEE8 !important;
    font-size: 17px !important;
    line-height: 1.75 !important;
}

.hero-chip-row,
.tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 18px;
}

.hero-chip,
.tag-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #F8FAFC !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}

.quick-stat {
    height: 100%;
    min-height: 132px;
    padding: 18px 18px 16px 18px;
    border-radius: 20px;
    background: linear-gradient(160deg, rgba(22, 30, 46, 0.96), rgba(16, 24, 39, 0.96));
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.quick-stat-label {
    color: #AAB7C8 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.quick-stat-value {
    margin-top: 10px;
    color: #FFFFFF !important;
    font-size: 30px !important;
    font-weight: 900 !important;
    line-height: 1.1 !important;
}

.quick-stat-caption {
    margin-top: 8px;
    color: #91A0B3 !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

.section-intro,
.workspace-card,
.empty-state {
    padding: 20px 22px;
    border-radius: 22px;
    background: linear-gradient(160deg, rgba(22, 30, 46, 0.96), rgba(16, 24, 39, 0.96));
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 18px;
}

.section-title {
    margin: 0 0 6px 0;
    color: #FFFFFF !important;
    font-size: 24px !important;
    font-weight: 900 !important;
}

.section-subtitle {
    margin: 0;
    color: #AAB7C8 !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}

.result-shell {
    padding: 22px;
    border-radius: 22px;
    border: 1px solid rgba(0, 217, 255, 0.18);
    background:
        radial-gradient(circle at top right, rgba(0, 217, 255, 0.12), transparent 32%),
        linear-gradient(155deg, rgba(18, 27, 42, 0.98), rgba(16, 24, 39, 0.95));
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
}

.result-shell.strong {
    border-color: rgba(46, 204, 113, 0.44);
}

.result-shell.medium {
    border-color: rgba(243, 156, 18, 0.42);
}

.result-shell.low {
    border-color: rgba(231, 76, 60, 0.42);
}

.result-eyebrow {
    color: #9EE7FF !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.result-title {
    margin: 8px 0 10px 0;
    color: #FFFFFF !important;
    font-size: 30px !important;
    font-weight: 900 !important;
    line-height: 1.2 !important;
}

.result-summary {
    margin: 0;
    color: #D6DEE8 !important;
    font-size: 15px !important;
    line-height: 1.75 !important;
}

.result-highlight {
    color: #9EE7FF !important;
    font-weight: 800 !important;
}

.empty-state {
    text-align: left;
    min-height: 220px;
}

.empty-state-title {
    margin: 0 0 10px 0;
    color: #FFFFFF !important;
    font-size: 28px !important;
    font-weight: 900 !important;
}

.empty-state-text {
    margin: 0;
    max-width: 720px;
    color: #B9C6D6 !important;
    font-size: 15px !important;
    line-height: 1.75 !important;
}

.tip-card {
    height: 100%;
    padding: 18px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.tip-card h4 {
    margin: 0 0 8px 0 !important;
    font-size: 18px !important;
    border: 0 !important;
    padding: 0 !important;
}

.tip-card p {
    margin: 0 !important;
    color: #AAB7C8 !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
}

.soft-divider {
    height: 1px;
    margin: 18px 0 22px 0;
    background: linear-gradient(90deg, transparent, rgba(0, 217, 255, 0.5), transparent);
}

div[data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(22, 30, 46, 0.96), rgba(16, 24, 39, 0.96));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 14px 16px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div {
    color: #F8FAFC !important;
}

div[data-testid="stMetricLabel"] {
    color: #AAB7C8 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding-top: 1.15rem;
}

.stDownloadButton > button {
    width: 100%;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #00D4FF, #7CFFB2) !important;
    color: #04131C !important;
    border: none !important;
    font-weight: 900 !important;
    box-shadow: 0 12px 24px rgba(0, 212, 255, 0.20);
}

.stDownloadButton > button * {
    color: #04131C !important;
    font-weight: 900 !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 16px 28px rgba(0, 212, 255, 0.24) !important;
}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {
    background: rgba(15, 23, 42, 0.92) !important;
    color: #EAF7FF !important;
    border: 1px solid rgba(0, 217, 255, 0.28) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.22);
}

[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="collapsedControl"] button:hover {
    background: rgba(23, 37, 58, 0.98) !important;
}

[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg {
    fill: #EAF7FF !important;
    color: #EAF7FF !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root {
    --ui-bg: #f5f9ff;
    --ui-panel: #ffffff;
    --ui-primary: #5b9ef0;
    --ui-primary-strong: #2969c9;
    --ui-accent: #e8f2ff;
    --ui-border: #d8e8f8;
    --ui-text: #16324f;
    --ui-muted: #5f7b98;
}

body, .main, .stApp {
    background: linear-gradient(180deg, #f2f8ff 0%, #f8fbff 100%) !important;
    color: var(--ui-text) !important;
}

p, label, span, .stMarkdown, .stMarkdown * {
    color: var(--ui-text) !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}

h1, h2, h3 {
    color: var(--ui-text) !important;
    text-shadow: none !important;
}

[data-testid="stSidebar"] {
    width: 292px !important;
    min-width: 292px !important;
    max-width: 292px !important;
    background: linear-gradient(180deg, #6aaef3 0%, #95d0ff 100%) !important;
    border-right: none !important;
}

[data-testid="stSidebar"] > div:first-child {
    width: 292px !important;
    min-width: 292px !important;
    max-width: 292px !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.hero-panel {
    border: none !important;
    padding: 18px 22px !important;
    margin: 4px 0 14px 0 !important;
    border-radius: 24px !important;
    background: linear-gradient(135deg, #5d9eea 0%, #86bef7 54%, #b8ddff 100%) !important;
    box-shadow: 0 16px 28px rgba(74, 144, 226, 0.14) !important;
}

.hero-title,
.hero-subtitle,
.hero-eyebrow {
    color: #ffffff !important;
}

.hero-title {
    max-width: 760px !important;
    margin: 8px 0 6px 0 !important;
    font-size: 38px !important;
    line-height: 1.06 !important;
    letter-spacing: -0.045em !important;
}

.hero-subtitle {
    max-width: 560px !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

.hero-chip-row {
    margin-top: 12px !important;
    gap: 10px !important;
}

.hero-chip {
    padding: 9px 13px !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    background: rgba(255, 255, 255, 0.14) !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    color: #ffffff !important;
}

.hero-chip.primary {
    background: #ffffff !important;
    border-color: #ffffff !important;
    color: var(--ui-primary-strong) !important;
    box-shadow: 0 12px 24px rgba(30, 78, 150, 0.18) !important;
}

.hero-chip.secondary {
    background: rgba(255, 255, 255, 0.16) !important;
}

.quick-stat,
.section-intro,
.workspace-card,
.empty-state,
.result-shell,
.tip-card,
div[data-testid="stMetric"],
.stDataFrame,
[data-testid="stExpander"],
[data-testid="stFileUploaderDropzone"] {
    background: var(--ui-panel) !important;
    border: 1px solid var(--ui-border) !important;
    box-shadow: 0 12px 30px rgba(20, 78, 150, 0.08) !important;
}

.quick-stat-label,
.result-eyebrow {
    color: var(--ui-primary) !important;
}

.quick-stat-value,
.section-title,
.result-title,
.empty-state-title,
div[data-testid="stMetricValue"] {
    color: var(--ui-text) !important;
}

.quick-stat-caption,
.section-subtitle,
.result-summary,
.empty-state-text,
.tip-card p,
div[data-testid="stMetricLabel"] {
    color: var(--ui-muted) !important;
}

.tag-chip {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    color: var(--ui-primary) !important;
}

.upload-shell {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 18px;
    margin-bottom: 10px;
    border-radius: 22px;
    background: linear-gradient(180deg, #ffffff, #f8fbff);
    border: 1px solid var(--ui-border);
    box-shadow: 0 12px 24px rgba(20, 78, 150, 0.08);
}

.upload-icon {
    flex: 0 0 52px;
    width: 52px;
    height: 52px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 16px;
    background: linear-gradient(135deg, var(--ui-primary), #8bc6ff);
    color: #ffffff !important;
    font-size: 24px !important;
    font-weight: 900 !important;
    box-shadow: 0 12px 22px rgba(74, 144, 226, 0.2);
}

.upload-copy {
    flex: 1 1 auto;
}

.upload-title {
    margin: 0;
    color: var(--ui-text) !important;
    font-size: 18px !important;
    font-weight: 900 !important;
    line-height: 1.2 !important;
}

.upload-subtitle {
    margin: 4px 0 0 0;
    color: var(--ui-muted) !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

.upload-note-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.upload-note {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 999px;
    background: #edf5ff;
    border: 1px solid #d4e5f8;
    color: var(--ui-primary-strong) !important;
    font-size: 12px !important;
    font-weight: 800 !important;
}

.quick-stat.featured {
    border-color: #c7def7 !important;
    background: linear-gradient(180deg, #f2f8ff, #ffffff) !important;
    box-shadow: 0 14px 30px rgba(74, 144, 226, 0.12) !important;
}

.quick-stat.featured .quick-stat-label {
    color: var(--ui-primary-strong) !important;
}

.quick-stat.featured .quick-stat-value {
    font-size: 36px !important;
}

.result-highlight {
    color: var(--ui-primary-strong) !important;
}

.result-status-pill {
    display: inline-flex;
    align-items: center;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 12px !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em;
}

.result-status-pill.strong {
    background: rgba(22, 163, 74, 0.12);
    color: #177a3b !important;
}

.result-status-pill.medium {
    background: rgba(245, 158, 11, 0.14);
    color: #a16207 !important;
}

.result-status-pill.low {
    background: rgba(239, 68, 68, 0.12);
    color: #b42318 !important;
}

.result-kpi-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 16px;
}

.result-kpi {
    padding: 14px 14px 12px;
    border-radius: 18px;
    background: #f7fbff;
    border: 1px solid #dbe9f7;
}

.result-kpi-label {
    color: var(--ui-muted) !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.result-kpi-value {
    margin-top: 6px;
    color: var(--ui-text) !important;
    font-size: 24px !important;
    font-weight: 900 !important;
    line-height: 1.05 !important;
}

.result-kpi-value.status {
    font-size: 17px !important;
}

.compare-shell {
    padding: 16px 18px;
    margin: 12px 0 14px;
    border-radius: 22px;
    background: linear-gradient(180deg, #ffffff, #f9fbff);
    border: 1px solid var(--ui-border);
    box-shadow: 0 12px 28px rgba(20, 78, 150, 0.08);
}

.compare-title {
    margin: 0;
    color: var(--ui-text) !important;
    font-size: 18px !important;
    font-weight: 900 !important;
}

.compare-subtitle {
    margin: 4px 0 0 0;
    color: var(--ui-muted) !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

.detail-table-shell,
.actions-shell {
    padding: 16px 18px;
    border-radius: 22px;
    background: linear-gradient(180deg, #ffffff, #fbfdff);
    border: 1px solid var(--ui-border);
    box-shadow: 0 12px 28px rgba(20, 78, 150, 0.08);
}

.detail-table-shell {
    margin-top: 8px;
}

.detail-table-title,
.actions-title {
    margin: 0 0 12px 0;
    color: var(--ui-text) !important;
    font-size: 18px !important;
    font-weight: 900 !important;
}

.detail-table {
    width: 100%;
    border-collapse: collapse;
}

.detail-table th {
    padding: 11px 12px;
    text-align: left;
    color: var(--ui-muted) !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid #deebf8;
}

.detail-table td {
    padding: 13px 12px;
    border-bottom: 1px solid #eef4fb;
    color: var(--ui-text) !important;
    font-size: 14px !important;
    vertical-align: top;
}

.detail-table tbody tr:nth-child(even) {
    background: #f9fbff;
}

.detail-table tbody tr:hover {
    background: #eef6ff;
}

.detail-main {
    font-weight: 800 !important;
    color: var(--ui-text) !important;
}

.detail-sub {
    margin-top: 4px;
    color: var(--ui-muted) !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}

.detail-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px !important;
    font-weight: 800 !important;
    border: 1px solid transparent;
}

.detail-badge.score-high {
    background: rgba(22, 163, 74, 0.12);
    border-color: rgba(22, 163, 74, 0.14);
    color: #177a3b !important;
}

.detail-badge.score-medium {
    background: rgba(245, 158, 11, 0.14);
    border-color: rgba(245, 158, 11, 0.16);
    color: #a16207 !important;
}

.detail-badge.score-low {
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.14);
    color: #b42318 !important;
}

.detail-badge.source {
    background: #edf5ff;
    border-color: #d4e5f8;
    color: var(--ui-primary-strong) !important;
}

.actions-shell {
    margin-top: 10px;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--ui-border) !important;
    gap: 12px !important;
    padding-top: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    min-height: 48px !important;
    background: rgba(232, 242, 255, 0.36) !important;
    color: var(--ui-muted) !important;
    border: 1px solid transparent !important;
    border-radius: 16px 16px 0 0 !important;
    padding: 12px 18px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--ui-primary-strong) !important;
    background: rgba(219, 234, 254, 0.7) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--ui-primary-strong) !important;
    border: 1px solid #c6def8 !important;
    border-bottom: 3px solid var(--ui-primary) !important;
    background: #ffffff !important;
    box-shadow: 0 12px 24px rgba(74, 144, 226, 0.08) !important;
    font-weight: 800 !important;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #4a90e2, #74b8ff) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    box-shadow: 0 10px 22px rgba(74, 144, 226, 0.18) !important;
    font-weight: 800 !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
}

.stDownloadButton > button *,
.stButton > button * {
    color: #ffffff !important;
}

.stFileUploader {
    margin-bottom: 14px !important;
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 176px !important;
    padding: 22px 20px !important;
    border: 2px dashed #a6c9ee !important;
    background: linear-gradient(180deg, #fbfdff, #f3f9ff) !important;
    box-shadow: 0 14px 24px rgba(74, 144, 226, 0.08) !important;
}

[data-testid="stFileUploaderDropzone"] > div {
    gap: 12px !important;
}

[data-testid="stFileUploaderDropzone"] p {
    color: var(--ui-text) !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    text-align: center !important;
}

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: var(--ui-muted) !important;
}

[data-testid="stFileUploaderDropzone"] button {
    border-radius: 14px !important;
    padding: 0.72rem 1.2rem !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 20px rgba(74, 144, 226, 0.18) !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #3f82d1, #63abf6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 14px 24px rgba(74, 144, 226, 0.22) !important;
}

[data-baseweb="switch"] > div {
    background: rgba(255, 255, 255, 0.3) !important;
}

[data-baseweb="switch"] input:checked + div {
    background: rgba(255, 255, 255, 0.95) !important;
}

.sidebar-stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.sidebar-card-min {
    padding: 16px 16px 14px !important;
    border-radius: 20px !important;
}

.sidebar-status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
    margin-bottom: 12px;
}

.sidebar-inline-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #ffffff !important;
    font-size: 12px !important;
    font-weight: 800 !important;
}

.sidebar-inline-badge.ready {
    background: rgba(22, 163, 74, 0.18);
    border-color: rgba(255, 255, 255, 0.24);
}

.soft-divider {
    height: 1px;
    margin: 10px 0 14px 0;
    background: linear-gradient(90deg, transparent, rgba(74, 144, 226, 0.32), transparent);
}

.sidebar-value.compact {
    max-width: 150px;
    font-size: 14px !important;
    font-weight: 800 !important;
    line-height: 1.35 !important;
    text-align: right;
}

div[data-testid="stMetricValue"] {
    font-size: 34px !important;
    font-weight: 900 !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {
    background: transparent !important;
    color: var(--ui-primary-strong) !important;
    border: none !important;
    width: 28px !important;
    min-width: 28px !important;
    height: 28px !important;
    min-height: 28px !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

[data-testid="stToolbar"] button:not([data-testid="stExpandSidebarButton"]),
[data-testid="stToolbar"] a {
    background: rgba(234, 244, 255, 0.96) !important;
    color: var(--ui-primary-strong) !important;
    border: 1px solid #b8d6f7 !important;
    min-height: 44px !important;
    height: 44px !important;
    width: auto !important;
    min-width: max-content !important;
    padding: 0 16px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 18px rgba(74, 144, 226, 0.14) !important;
    white-space: nowrap !important;
    font-size: 15px !important;
    line-height: 1 !important;
    font-weight: 700 !important;
    overflow: visible !important;
}

[data-testid="stToolbar"] button:not([data-testid="stExpandSidebarButton"]) *,
[data-testid="stToolbar"] a * {
    white-space: nowrap !important;
}

[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg {
    fill: var(--ui-primary-strong) !important;
    color: var(--ui-primary-strong) !important;
    stroke: var(--ui-primary-strong) !important;
    width: 18px !important;
    height: 18px !important;
}

[data-testid="stSidebarCollapseButton"] svg *,
[data-testid="collapsedControl"] svg * {
    fill: var(--ui-primary-strong) !important;
    color: var(--ui-primary-strong) !important;
    stroke: var(--ui-primary-strong) !important;
}

[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="collapsedControl"] button:hover {
    background: #ffffff !important;
}

[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="collapsedControl"] button:hover {
    background: transparent !important;
    border-color: transparent !important;
    box-shadow: none !important;
}

[data-testid="collapsedControl"] {
    background: transparent !important;
    padding-left: 10px !important;
    pointer-events: auto !important;
    width: 60px !important;
    min-width: 60px !important;
    max-width: 60px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    white-space: nowrap !important;
    font-size: 0 !important;
    color: transparent !important;
}

[data-testid="stSidebarCollapseButton"] {
    position: relative !important;
    z-index: 100000 !important;
    width: 60px !important;
    min-width: 60px !important;
    max-width: 60px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    white-space: nowrap !important;
    font-size: 0 !important;
    color: transparent !important;
}

[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapseButton"] * {
    color: transparent !important;
}

[data-testid="collapsedControl"] > :not(button),
[data-testid="stSidebarCollapseButton"] > :not(button),
[data-testid="collapsedControl"] span,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="collapsedControl"] [class*="material-symbol"],
[data-testid="stSidebarCollapseButton"] [class*="material-symbol"] {
    display: none !important;
}

/* Force the sidebar toggle to render as a clean arrow button even if
   Streamlit falls back to icon ligature text like "keyboard_double_arrow_*". */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {
    position: relative !important;
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    text-shadow: none !important;
    overflow: hidden !important;
    text-indent: -9999px !important;
}

[data-testid="stSidebarCollapseButton"] button > *,
[data-testid="collapsedControl"] button > * {
    display: none !important;
}

[data-testid="stSidebarCollapseButton"] button::before,
[data-testid="collapsedControl"] button::before {
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    font-size: 22px !important;
    line-height: 1 !important;
    font-weight: 800 !important;
    color: var(--ui-primary-strong) !important;
    text-shadow: none !important;
    text-indent: 0 !important;
}

[data-testid="stSidebarCollapseButton"] button::before {
    content: "\\2039";
}

[data-testid="collapsedControl"] button::before {
    content: "\\203A";
}

header[data-testid="stHeader"] [data-testid="collapsedControl"] {
    display: none !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
}

header[data-testid="stHeader"] [data-testid="collapsedControl"] * {
    display: none !important;
}

#custom-sidebar-toggle {
    display: none !important;
}

button[data-testid="stExpandSidebarButton"],
header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"] {
    position: fixed !important;
    top: 78px !important;
    left: 8px !important;
    z-index: 100003 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 28px !important;
    min-width: 28px !important;
    height: 34px !important;
    min-height: 34px !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
    pointer-events: auto !important;
}

button[data-testid="stExpandSidebarButton"],
header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"] button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 28px !important;
    min-width: 28px !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 !important;
    border: none !important;
    border-radius: 10px !important;
    background: transparent !important;
    box-shadow: none !important;
    color: transparent !important;
    font-size: 0 !important;
    line-height: 0 !important;
    text-indent: -9999px !important;
    overflow: hidden !important;
}

button[data-testid="stExpandSidebarButton"]::before,
header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"] button::before {
    display: block !important;
    font: 800 20px/1 "Segoe UI Symbol", "Arial Unicode MS", sans-serif !important;
    color: #2969c9 !important;
    text-indent: 0 !important;
}

button[data-testid="stExpandSidebarButton"]::before {
    content: "\\203A" !important;
}

header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"] button::before {
    content: "\\2039" !important;
}

button[data-testid="stExpandSidebarButton"] *,
header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"] button * {
    display: none !important;
}

body[data-sidebar-state="collapsed"] [data-testid="stSidebar"],
body[data-sidebar-state="collapsed"] [data-testid="stSidebar"] > div:first-child {
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    overflow: hidden !important;
    border-right: none !important;
}

body[data-sidebar-state="expanded"] [data-testid="stSidebar"],
body[data-sidebar-state="expanded"] [data-testid="stSidebar"] > div:first-child {
    width: 292px !important;
    min-width: 292px !important;
    max-width: 292px !important;
}

body[data-sidebar-state] #custom-sidebar-toggle {
    position: fixed !important;
    top: 78px !important;
    left: 8px !important;
    width: 28px !important;
    min-width: 28px !important;
    height: 34px !important;
    min-height: 34px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    border-radius: 10px !important;
    background: transparent !important;
    box-shadow: none !important;
    color: transparent !important;
    text-indent: -9999px !important;
    overflow: hidden !important;
    cursor: pointer !important;
    pointer-events: auto !important;
    z-index: 100002 !important;
}

body[data-sidebar-state] #custom-sidebar-toggle::before {
    display: block !important;
    font: 800 20px/1 "Segoe UI Symbol", "Arial Unicode MS", sans-serif !important;
    color: #2969c9 !important;
    text-indent: 0 !important;
}

body[data-sidebar-state="collapsed"] #custom-sidebar-toggle::before {
    content: "\\203A" !important;
}

body[data-sidebar-state="expanded"] #custom-sidebar-toggle::before {
    content: "\\2039" !important;
}

#custom-sidebar-toggle,
button[data-testid="stExpandSidebarButton"],
header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"],
header[data-testid="stHeader"] [data-testid="collapsedControl"] {
    display: none !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

#custom-sidebar-toggle *,
button[data-testid="stExpandSidebarButton"] *,
header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"] *,
header[data-testid="stHeader"] [data-testid="collapsedControl"] * {
    display: none !important;
}

header[data-testid="stHeader"] {
    min-height: 0 !important;
    height: 0 !important;
    background: transparent !important;
    border-bottom: none !important;
    backdrop-filter: none !important;
    overflow: hidden !important;
}

header[data-testid="stHeader"]::before,
[data-testid="stToolbar"] {
    display: none !important;
}

body[data-sidebar-state="collapsed"] [data-testid="stSidebar"],
body[data-sidebar-state="collapsed"] [data-testid="stSidebar"] > div:first-child,
body[data-sidebar-state="expanded"] [data-testid="stSidebar"],
body[data-sidebar-state="expanded"] [data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    width: 332px !important;
    min-width: 332px !important;
    max-width: 332px !important;
    transform: translateX(0) !important;
    margin-left: 0 !important;
    left: 0 !important;
    visibility: visible !important;
    display: block !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    box-sizing: border-box !important;
    height: 100vh !important;
}

[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"] > div:first-child > div:first-child {
    padding-left: 14px !important;
    padding-right: 14px !important;
    box-sizing: border-box !important;
}

[data-testid="stAppViewContainer"],
.main,
.stApp,
section.main {
    overflow-y: auto !important;
}

.block-container {
    padding-top: 0.12rem !important;
    padding-bottom: 2.35rem !important;
    max-width: 1320px !important;
}

[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding-top: 0.72rem !important;
    margin-top: 0 !important;
}

.hero-panel {
    padding: 16px 22px !important;
    margin: 2px 0 14px 0 !important;
    border-radius: 22px !important;
    min-height: 214px !important;
}

.hero-panel::after {
    display: none !important;
    background: none !important;
}

.hero-title {
    max-width: 620px !important;
    font-size: 33px !important;
    line-height: 1.08 !important;
}

.hero-subtitle {
    max-width: 560px !important;
    font-size: 14px !important;
}

.hero-chip-row {
    margin-top: 10px !important;
}

.hero-panel::before {
    left: -36px !important;
    bottom: -86px !important;
    width: 180px !important;
    height: 180px !important;
}

.hero-panel::after {
    display: none !important;
    background: none !important;
}

.quick-stat {
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    height: 152px !important;
    min-height: 152px !important;
    padding: 16px 18px 16px !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease !important;
}

.quick-stat:hover {
    transform: translateY(-2px) !important;
    border-color: #bfd8f8 !important;
    box-shadow: 0 16px 28px rgba(20, 78, 150, 0.12) !important;
}

.quick-stat.featured {
    background: linear-gradient(180deg, #f8fbff 0%, #edf5ff 100%) !important;
    border-color: #b9d7fa !important;
    box-shadow: 0 14px 26px rgba(74, 144, 226, 0.12) !important;
}

.quick-stat-label {
    min-height: 18px !important;
}

.quick-stat-value {
    margin-top: 18px !important;
    font-size: 31px !important;
    line-height: 1 !important;
}

.quick-stat-caption {
    margin-top: auto !important;
    min-height: 40px !important;
    display: flex !important;
    align-items: flex-end !important;
}

.quick-stat.featured .quick-stat-caption {
    color: var(--ui-primary-strong) !important;
}

.quick-stat.featured .quick-stat-value {
    font-size: 33px !important;
}

[data-testid="stTabs"] {
    margin-top: 8px !important;
}

.stTabs [data-baseweb="tab-list"] {
    width: 100% !important;
    gap: 10px !important;
    padding: 0 0 0 2px !important;
    margin: 0 !important;
    align-items: flex-end !important;
}

.stTabs [data-baseweb="tab"] {
    min-height: 50px !important;
    margin-bottom: -1px !important;
}

.stTabs [aria-selected="true"] {
    box-shadow: 0 10px 18px rgba(74, 144, 226, 0.08) !important;
}

.upload-shell {
    padding: 16px 18px !important;
    gap: 14px !important;
    background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%) !important;
    border-left: 4px solid var(--ui-primary) !important;
}

.upload-icon {
    width: 56px !important;
    height: 56px !important;
    font-size: 22px !important;
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 164px !important;
    padding: 18px 18px !important;
}

.section-intro {
    margin-bottom: 10px !important;
}

.result-shell,
.empty-state {
    padding: 20px 22px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root {
    --ui-shell-max: 1280px;
    --ui-sidebar-width: 324px;
    --ui-radius-card: 20px;
    --ui-shadow-soft: 0 16px 34px rgba(37, 94, 154, 0.10);
    --ui-sidebar-ink: #0f2338;
    --ui-sidebar-ink-soft: #3f5a74;
}

.block-container {
    max-width: var(--ui-shell-max) !important;
    padding-top: 0.18rem !important;
}

body[data-sidebar-state="collapsed"] [data-testid="stSidebar"],
body[data-sidebar-state="collapsed"] [data-testid="stSidebar"] > div:first-child,
body[data-sidebar-state="expanded"] [data-testid="stSidebar"],
body[data-sidebar-state="expanded"] [data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    width: var(--ui-sidebar-width) !important;
    min-width: var(--ui-sidebar-width) !important;
    max-width: var(--ui-sidebar-width) !important;
}

[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"] > div:first-child > div:first-child {
    padding-left: 16px !important;
    padding-right: 16px !important;
    padding-bottom: 30px !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    scrollbar-width: thin !important;
}

[data-testid="stSidebar"] * {
    color: var(--ui-sidebar-ink) !important;
    text-shadow: none !important;
}

[data-testid="stSidebar"] h3 {
    margin: 0 0 10px 0 !important;
    font-size: 15px !important;
    line-height: 1.3 !important;
    letter-spacing: -0.01em !important;
    color: var(--ui-sidebar-ink) !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption * {
    font-size: 13px !important;
    line-height: 1.55 !important;
    opacity: 0.92 !important;
    color: var(--ui-sidebar-ink-soft) !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-baseweb="slider"] span,
[data-testid="stSidebar"] .stToggle label,
[data-testid="stSidebar"] .stCheckbox label {
    color: var(--ui-sidebar-ink) !important;
}

.sidebar-stack {
    gap: 10px !important;
}

.sidebar-card-min {
    padding: 14px 14px 12px !important;
    border-radius: 18px !important;
    background: rgba(255, 255, 255, 0.16) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.20) !important;
}

.sidebar-title {
    font-size: 12px !important;
    font-weight: 800 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--ui-sidebar-ink) !important;
}

.sidebar-row {
    gap: 12px !important;
    align-items: flex-start !important;
}

.sidebar-label {
    font-size: 12px !important;
    opacity: 0.86 !important;
    color: var(--ui-sidebar-ink-soft) !important;
}

.sidebar-value,
.sidebar-value.compact {
    font-size: 14px !important;
    line-height: 1.35 !important;
    color: var(--ui-sidebar-ink) !important;
}

.sidebar-inline-badge,
.sidebar-badge {
    padding: 6px 9px !important;
    font-size: 11.5px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
}

.sidebar-badge {
    background: rgba(22, 64, 126, 0.16) !important;
    border: 1px solid rgba(22, 64, 126, 0.14) !important;
}

.sidebar-inline-badge:not(.ready) {
    background: rgba(18, 59, 117, 0.18) !important;
    border-color: rgba(18, 59, 117, 0.16) !important;
}

[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    box-shadow: 0 0 0 6px rgba(15, 35, 56, 0.12) !important;
}

[data-testid="stSidebar"] svg {
    color: var(--ui-sidebar-ink-soft) !important;
}

.hero-panel {
    padding: 20px 24px !important;
    min-height: 198px !important;
    border-radius: 24px !important;
    box-shadow: 0 18px 34px rgba(37, 94, 154, 0.12) !important;
}

.hero-title {
    max-width: 690px !important;
    font-size: 34px !important;
    line-height: 1.06 !important;
}

.hero-subtitle {
    max-width: 620px !important;
    font-size: 15px !important;
    line-height: 1.58 !important;
}

.hero-chip-row {
    margin-top: 12px !important;
    gap: 8px !important;
}

.hero-chip {
    padding: 8px 12px !important;
    font-size: 12px !important;
    box-shadow: none !important;
}

.hero-chip.primary {
    box-shadow: 0 10px 18px rgba(30, 78, 150, 0.12) !important;
}

.quick-stat,
.section-intro,
.upload-shell,
.result-shell,
.empty-state,
.tip-card,
.detail-table-shell,
.progress-card,
.insight-card,
div[data-testid="stMetric"],
div[data-testid="stPlotlyChart"],
.stDataFrame {
    box-shadow: var(--ui-shadow-soft) !important;
    border-radius: var(--ui-radius-card) !important;
}

.quick-stat {
    height: 148px !important;
    min-height: 148px !important;
}

.quick-stat.featured .quick-stat-value {
    font-size: 32px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 16px 16px 0 0 !important;
    padding: 11px 20px !important;
}

.upload-shell,
.result-shell {
    border-left-width: 3px !important;
}

.detail-table-shell {
    overflow: hidden !important;
}

.detail-table th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: #f8fbff;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def inject_streamlit_header_cleanup() -> None:
    components.html(
        """
        <script>
        (() => {
          const doc = window.parent.document;

          const hideSidebarToggles = () => {
            [
              '[data-testid="stExpandSidebarButton"]',
              '[data-testid="stSidebarCollapseButton"]',
              '[data-testid="collapsedControl"]',
            ].forEach((selector) => {
              doc.querySelectorAll(selector).forEach((element) => {
                element.style.display = "none";
                element.style.opacity = "0";
                element.style.pointerEvents = "none";
                element.style.width = "0";
                element.style.minWidth = "0";
                element.style.maxWidth = "0";
                element.style.margin = "0";
                element.style.padding = "0";
                element.style.overflow = "hidden";
              });
            });
          };

          const ensureExpanded = () => {
            const expandButton = doc.querySelector('[data-testid="stExpandSidebarButton"]');
            if (expandButton) {
              expandButton.click();
            }
            hideSidebarToggles();
          };

          let attempts = 0;
          const tick = () => {
            attempts += 1;
            ensureExpanded();
            if (attempts < 12) {
              window.setTimeout(tick, 180);
            }
          };

          tick();
        })();
        </script>
        """,
        height=0,
        width=0,
    )

def format_label_display(label_id: int, label_display_names: Dict[int, str]) -> str:
    label_name = str(label_display_names.get(int(label_id), "")).strip()
    return f"{int(label_id)} | {label_name}" if label_name else f"Viên Thuốc {int(label_id)}"


def normalize_uploaded_image(image_array: np.ndarray) -> np.ndarray:
    if image_array.ndim == 2:
        return cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    if image_array.shape[-1] == 4:
        return cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    return image_array


@st.cache_resource
def load_models_legacy():
    """Load pretrained model thật từ checkpoints."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check files
    best_model_path = Path("checkpoints/best_model.pth")
    dataset_summary_path = Path("checkpoints/dataset_summary.json")
    
    if not best_model_path.exists():
        st.error("❌ Model file not found!")
        st.stop()
    
    # Load dataset summary
    with open(dataset_summary_path) as f:
        dataset_info = json.load(f)
    
    num_classes = dataset_info.get("num_classes", 108)
    class_dist = dataset_info.get("class_distribution", {})
    
    # Load checkpoint để detect model variant
    checkpoint = torch.load(best_model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Chọn model architecture dựa trên checkpoint keys
    if any('image_backbone.' in k for k in state_dict.keys()):
        # Color fusion model
        model = CGIMIFColorFusionClassifier(
            num_classes=num_classes,
            color_feature_dim=24,
            pretrained=True
        )
        has_color_fusion = True
    else:
        # Simple ResNet18 (đây là model được train thực tế)
        model = SimpleResNet18Classifier(
            num_classes=num_classes,
            pretrained=True
        )
        has_color_fusion = False
    
    # Load weights
    try:
        # FIX: Handle key mismatch for simple ResNet model
        # Keys không có prefix nên add vào nếu cần
        new_state_dict = {}
        for key, value in state_dict.items():
            # Nếu model hiện tại dùng backbone., thêm prefix
            if not has_color_fusion and not key.startswith('backbone.'):
                # Các keys từ ResNet backbone
                if key in ['conv1.weight', 'bn1.weight', 'bn1.bias'] or key.startswith(('layer', 'fc', 'bn1', 'avgpool')):
                    new_key = f'backbone.{key}'
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            else:
                new_state_dict[key] = value
        
        # Load state dict
        model.load_state_dict(new_state_dict, strict=False)
        
        model.to(device)
        model.eval()
        
        # Create class id to name mapping
        class_names = {i: f"Thuốc_{i:03d}" for i in range(num_classes)}
        
        # Update với actual class IDs từ dataset
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
    
    except Exception as e:
        st.error(f"❌ Lỗi load model: {str(e)}")
        st.stop()


@st.cache_resource
def _load_models_cached(cache_version: str):
    """Load detector + classifier bundle đúng cho ảnh nhiều viên thuốc."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = cache_version

    if not DEFAULT_DEMO_CLASSIFIER_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Thiếu classifier checkpoint mặc định: "
            f"{DEFAULT_DEMO_CLASSIFIER_CHECKPOINT}. {RUNTIME_SETUP_HINT}"
        )

    classifier_model, classifier_checkpoint, _, classifier_idx_to_class = load_classifier_checkpoint(
        DEFAULT_DEMO_CLASSIFIER_CHECKPOINT,
        device,
    )

    knowledge_graph = None
    confusion_metrics_path = DEFAULT_DEMO_CLASSIFIER_CHECKPOINT.parent / "test_metrics.json"
    if not confusion_metrics_path.exists():
        confusion_metrics_path = None

    if DEFAULT_KNOWLEDGE_GRAPH_PATH.exists() or DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR.exists():
        try:
            knowledge_graph = load_or_build_knowledge_graph(
                artifact_path=DEFAULT_KNOWLEDGE_GRAPH_PATH,
                cache_dir=DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR,
                image_size=int(classifier_checkpoint["image_size"]),
                color_bins=int(classifier_checkpoint.get("color_bins", 8)),
                confusion_metrics_path=confusion_metrics_path,
                rebuild=False,
            )
        except Exception as graph_error:
            st.info(
                "ℹ️ Không load được knowledge graph, app sẽ chạy detector + classifier không reranking. "
                f"Chi tiết: {graph_error}"
            )

    try:
        detector_model, detector_checkpoint, _, detector_index_to_label = load_detection_checkpoint(
            DEFAULT_DEMO_DETECTOR_CHECKPOINT,
            device,
        )
        return {
            "mode": "multi_pill_detection",
            "device": device,
            "score_threshold": DEFAULT_DETECTION_SCORE_THRESHOLD,
            "detector_model": detector_model,
            "detector_checkpoint": detector_checkpoint,
            "detector_index_to_label": detector_index_to_label,
            "classifier_model": classifier_model,
            "classifier_checkpoint": classifier_checkpoint,
            "classifier_idx_to_class": classifier_idx_to_class,
            "label_display_names": build_label_display_names(),
            "knowledge_graph": knowledge_graph,
        }

    except Exception as detection_error:
        st.warning(
            "⚠️ Không load được detector nhiều viên, app sẽ fallback sang classifier một-crop. "
            f"Chi tiết: {detection_error}. {RUNTIME_SETUP_HINT}"
        )
        return {
            "mode": "single_crop_classifier",
            "device": device,
            "classifier_model": classifier_model,
            "classifier_checkpoint": classifier_checkpoint,
            "classifier_idx_to_class": classifier_idx_to_class,
            "label_display_names": build_label_display_names(),
            "knowledge_graph": knowledge_graph,
        }


def load_models():
    """Wrapper để có thể chủ động đổi cache key khi model bundle thay đổi."""
    return _load_models_cached(MODEL_BUNDLE_CACHE_VERSION)


def process_image(image_array: np.ndarray) -> np.ndarray:
    """
    Chuẩn bị ảnh cho model.
    - Resize về 160x160
    - Normalize
    """
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
    
    return tensor.unsqueeze(0)  # Add batch dimension


def extract_color_histogram(image: np.ndarray, bins: int = 8) -> np.ndarray:
    """Trích xuất HSV histogram features."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    hist = hist / (hist.sum() + 1e-8)
    return hist.astype(np.float32)


def predict_pill_legacy(image: np.ndarray, model_info: Dict, checkpoint: Dict = None) -> Dict:
    """
    Dự đoán loại viên thuốc bằng model THẬT.
    """
    from torch.nn import functional as F
    
    try:
        model = model_info["model"]
        device = model_info["device"]
        class_names = model_info["class_names"]
        has_color_fusion = model_info.get("has_color_fusion", False)
        
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare image tensor
        image_tensor = process_image(image_rgb).to(device)
        
        # Forward pass - branch dựa trên model type
        with torch.no_grad():
            if has_color_fusion:
                # Color fusion model: cần cả image + color features
                color_hist = extract_color_histogram(image_rgb)
                color_tensor = torch.from_numpy(color_hist).unsqueeze(0).to(device).float()
                logits = model(image_tensor.float(), color_tensor)
            else:
                # Simple ResNet18: chỉ cần image
                logits = model(image_tensor.float())
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)[0]
        
        # Get top predictions
        top_k = torch.topk(probs, k=min(5, len(probs)))
        top_classes = top_k.indices.cpu().numpy()
        top_probs = top_k.values.cpu().numpy()
        
        pred_class = int(top_classes[0])
        confidence = float(top_probs[0])
        
        result = {
            "class_id": pred_class,
            "class_name": class_names.get(pred_class, f"Thuốc_{pred_class}"),
            "confidence": confidence,
            "probabilities": {str(int(c)): float(p) for c, p in zip(top_classes, top_probs)},
            "top_5": [(int(c), class_names.get(int(c), f"Thuốc_{c}"), float(p)) 
                      for c, p in zip(top_classes[:5], top_probs[:5])],
        }
        
        return result
        
    except Exception as e:
        st.error(f"❌ Lỗi dự đoán: {str(e)}")
        # Return mock result on error
        num_classes = model_info.get("num_classes", 108)
        logits = torch.randn(1, num_classes)
        probs = F.softmax(logits, dim=1)[0]
        top_k = torch.topk(probs, k=5)
        top_classes = top_k.indices.cpu().numpy()
        top_probs = top_k.values.cpu().numpy()
        class_names = model_info.get("class_names", {i: f"Thuốc_{i}" for i in range(num_classes)})
        
        return {
            "class_id": int(top_classes[0]),
            "class_name": class_names.get(top_classes[0], f"Thuốc_{top_classes[0]}"),
            "confidence": float(top_probs[0]),
            "probabilities": {str(c): float(p) for c, p in zip(top_classes, top_probs)},
            "top_5": [(int(c), class_names.get(c, f"Thuốc_{c}"), float(p)) 
                      for c, p in zip(top_classes[:5], top_probs[:5])],
        }


def create_detection_score_chart(detections):
    detection_rows = [
        {
            "Viên": f"#{index + 1}",
            "Nhãn": item["display_label"],
            "Độ tin cậy": float(item["score"]),
        }
        for index, item in enumerate(detections)
    ]
    fig = px.bar(
        pd.DataFrame(detection_rows),
        x="Độ tin cậy",
        y="Viên",
        color="Độ tin cậy",
        text="Nhãn",
        orientation="h",
        color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
    )
    fig.update_layout(
        template="plotly_dark",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_showscale=False,
        xaxis=dict(range=[0, 1]),
    )
    fig.update_traces(textposition="outside")
    return fig


def predict_crop_classifier(image_rgb: np.ndarray, model_info: Dict) -> Dict:
    candidates = classify_crop_candidates(
        model=model_info["classifier_model"],
        checkpoint=model_info["classifier_checkpoint"],
        idx_to_class=model_info["classifier_idx_to_class"],
        device=model_info["device"],
        crop_rgb=image_rgb,
        top_k=5,
    )
    if not candidates:
        raise RuntimeError("Classifier không trả về candidate nào.")

    label_display_names = model_info["label_display_names"]
    top_candidate = candidates[0]
    return {
        "mode": "single_crop_classifier",
        "class_id": int(top_candidate["label_id"]),
        "class_name": format_label_display(int(top_candidate["label_id"]), label_display_names),
        "confidence": float(top_candidate["probability"]),
        "probabilities": {
            str(int(item["label_id"])): float(item["probability"])
            for item in candidates
        },
        "top_5": [
            (
                int(item["label_id"]),
                format_label_display(int(item["label_id"]), label_display_names),
                float(item["probability"]),
            )
            for item in candidates
        ],
    }


def predict_pill(image: np.ndarray, model_info: Dict, checkpoint: Dict = None) -> Dict:
    from torch.nn import functional as F

    try:
        image_rgb = normalize_uploaded_image(np.asarray(image, dtype=np.uint8))

        if model_info.get("mode") != "multi_pill_detection":
            return predict_crop_classifier(image_rgb, model_info)

        DEFAULT_STREAMLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = DEFAULT_STREAMLIT_OUTPUT_DIR / run_id

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_image_path = Path(handle.name)

        try:
            Image.fromarray(image_rgb).save(temp_image_path)
            payload = predict_single_image(
                model=model_info["detector_model"],
                checkpoint=model_info["detector_checkpoint"],
                index_to_label=model_info["detector_index_to_label"],
                device=model_info["device"],
                image_path=temp_image_path,
                output_dir=run_dir,
                score_threshold=float(model_info["score_threshold"]),
                classifier_model=model_info["classifier_model"],
                classifier_checkpoint=model_info["classifier_checkpoint"],
                classifier_idx_to_class=model_info["classifier_idx_to_class"],
                knowledge_graph=model_info["knowledge_graph"],
                kg_top_k=DEFAULT_KG_CANDIDATES,
                kg_visual_weight=DEFAULT_KG_VISUAL_WEIGHT,
                kg_context_weight=DEFAULT_KG_CONTEXT_WEIGHT,
                kg_anchor_weight=DEFAULT_KG_ANCHOR_WEIGHT,
                kg_selective_override=DEFAULT_KG_SELECTIVE_OVERRIDE,
                kg_max_detector_score=DEFAULT_KG_MAX_DETECTOR_SCORE,
                kg_min_candidate_probability=DEFAULT_KG_MIN_CANDIDATE_PROBABILITY,
                kg_max_anchor_probability=DEFAULT_KG_MAX_ANCHOR_PROBABILITY,
            )
        finally:
            temp_image_path.unlink(missing_ok=True)

        response = build_app_response(payload, label_display_names=model_info["label_display_names"])
        detections = list(response.get("detections", []))
        if not detections:
            fallback = predict_crop_classifier(image_rgb, model_info)
            fallback["warning"] = "Detector chưa tìm thấy viên thuốc rõ ràng, đang fallback sang classifier cho toàn ảnh."
            return fallback

        ranked_detections = sorted(detections, key=lambda item: float(item["score"]), reverse=True)
        primary_detection = ranked_detections[0]
        preview_path = response.get("artifacts", {}).get("knowledge_graph_preview") or response.get("artifacts", {}).get("detector_preview")
        return {
            "mode": "multi_pill_detection",
            "class_id": int(primary_detection["label_id"]),
            "class_name": str(primary_detection["display_label"]),
            "confidence": float(primary_detection["score"]),
            "num_detections": int(response.get("num_detections", len(detections))),
            "raw_num_detections": int(response.get("raw_num_detections", len(detections))),
            "suppressed_detections": int(response.get("suppressed_detections", 0)),
            "detections": ranked_detections,
            "top_labels": [str(item["display_label"]) for item in ranked_detections[:5]],
            "probabilities": {
                str(int(item["label_id"])): float(item["score"])
                for item in ranked_detections[:5]
            },
            "top_5": [
                (
                    int(item["label_id"]),
                    str(item["display_label"]),
                    float(item["score"]),
                )
                for item in ranked_detections[:5]
            ],
            "preview_path": preview_path,
            "artifacts": dict(response.get("artifacts", {})),
            "raw_response": response,
        }

    except Exception as e:
        st.error(f"❌ Lỗi dự đoán: {str(e)}")
        num_classes = int(model_info.get("num_classes", 108))
        logits = torch.randn(1, num_classes)
        probs = F.softmax(logits, dim=1)[0]
        top_k = torch.topk(probs, k=5)
        top_classes = top_k.indices.cpu().numpy()
        top_probs = top_k.values.cpu().numpy()
        return {
            "mode": "error_fallback",
            "class_id": int(top_classes[0]),
            "class_name": f"Viên Thuốc {int(top_classes[0])}",
            "confidence": float(top_probs[0]),
            "probabilities": {str(int(c)): float(p) for c, p in zip(top_classes, top_probs)},
            "top_5": [(int(c), f"Viên Thuốc {int(c)}", float(p)) for c, p in zip(top_classes[:5], top_probs[:5])],
        }


def create_confidence_gauge(confidence: float):
    """Tạo biểu đồ gauge cho confidence."""
    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Mức tin cậy"},
        delta={'reference': 80, 'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00D4FF"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.3)"},
                {'range': [50, 80], 'color': "rgba(245, 158, 11, 0.3)"},
                {'range': [80, 100], 'color': "rgba(16, 185, 129, 0.3)"},
            ],
            'threshold': {
                'line': {'color': "#FF006E", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    )])
    
    fig.update_layout(
        font={"color": "#e2e8f0"},
        paper_bgcolor="rgba(26, 32, 44, 0.8)",
        plot_bgcolor="rgba(26, 32, 44, 0.8)",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    
    return fig


def create_top_k_chart(top_k_data: list):
    """Tạo biểu đồ top-K predictions."""
    classes = [item[1] for item in top_k_data[:5]]
    probs = [item[2] * 100 for item in top_k_data[:5]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=classes,
            orientation='h',
            marker=dict(
                color=probs,
                colorscale=[[0, '#ef4444'], [0.5, '#f59e0b'], [1, '#10b981']],
            ),
            text=[f"{p:.1f}%" for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Top 5 nhãn mạnh nhất",
        xaxis_title="Xác suất (%)",
        yaxis_title="Nhãn",
        font={"color": "#e2e8f0"},
        paper_bgcolor="rgba(26, 32, 44, 0.8)",
        plot_bgcolor="rgba(26, 32, 44, 0.8)",
        hovermode='y unified',
    )
    
    return fig


def create_class_distribution_chart(top_k_data: list):
    """Tạo biểu đồ phân phối xác suất."""
    df = pd.DataFrame([
        {"Nhãn": item[1], "Xác suất": float(item[2]) * 100}
        for item in top_k_data[:5]
    ])
    
    fig = px.pie(
        df,
        values="Xác suất",
        names="Nhãn",
        title="Phân bổ xác suất top 5",
        hole=0.48,
        color_discrete_sequence=["#00D4FF", "#14B8A6", "#F59E0B", "#F97316", "#EC4899"],
    )
    
    fig.update_layout(
        font={"color": "#e2e8f0"},
        paper_bgcolor="rgba(26, 32, 44, 0.8)",
        plot_bgcolor="rgba(26, 32, 44, 0.8)",
    )
    
    return fig


def create_detection_score_chart_v2(detections: list):
    rows = [
        {
            "Viên": f"Viên {index + 1}",
            "Nhãn": item["display_label"],
            "Độ chính xác": float(item["score"]) * 100,
        }
        for index, item in enumerate(detections)
    ]
    if not rows:
        return go.Figure()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("Độ chính xác", ascending=False).reset_index(drop=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=frame["Độ chính xác"],
                y=frame["Viên"],
                orientation="h",
                marker=dict(
                    color=frame["Độ chính xác"],
                    colorscale=[[0, "#dbeafe"], [0.55, "#7cb4f8"], [1, "#1565d8"]],
                    line=dict(color="#d0e2f5", width=1),
                ),
                text=[f"{value:.1f}%" for value in frame["Độ chính xác"]],
                textposition="auto",
                customdata=np.array(frame["Nhãn"]).reshape(-1, 1),
                hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>%{x:.1f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Độ chính xác từng viên",
        height=290,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#16324f"},
        margin=dict(l=20, r=20, t=48, b=20),
        xaxis=dict(range=[0, 100], title=None, showgrid=True, gridcolor="#e7f0fa", ticksuffix="%"),
        yaxis=dict(title=None, autorange="reversed"),
        showlegend=False,
    )
    fig.update_traces(cliponaxis=False, textfont=dict(color="#16324f", size=12))
    return fig


def create_confidence_gauge_v2(
    confidence: float,
    title: str = "Độ chính xác dự đoán",
    *,
    reference: Optional[float] = None,
):
    value = max(0.0, min(float(confidence) * 100.0, 100.0))
    reference_value = None if reference is None else max(0.0, min(float(reference) * 100.0, 100.0))
    if reference_value is not None and normalize_delta_points(value - reference_value) == 0.0:
        reference_value = value
    indicator_config = {
        "mode": "gauge+number" + ("+delta" if reference is not None else ""),
        "value": value,
        "domain": {"x": [0.04, 0.96], "y": [0.0, 0.82]},
        "number": {
            "suffix": "%",
            "font": {"size": 50, "color": "#17365d"},
        },
        "title": {"text": ""},
        "gauge": {
            "bgcolor": "rgba(255,255,255,0)",
            "borderwidth": 0,
            "axis": {
                "range": [0, 100],
                "tickmode": "linear",
                "tick0": 0,
                "dtick": 20,
                "tickwidth": 1.1,
                "tickcolor": "#6b87ad",
                "tickfont": {"size": 16, "color": "#537196"},
            },
            "bar": {
                "color": "#1f6dde",
                "thickness": 0.38,
                "line": {"color": "#1a57b4", "width": 1.4},
            },
            "steps": [
                {"range": [0, 45], "color": "#edf5ff"},
                {"range": [45, 75], "color": "#d9e9ff"},
                {"range": [75, 90], "color": "#bdd8fb"},
                {"range": [90, 100], "color": "#8bb9f5"},
            ],
            "threshold": {
                "line": {"color": "#0f4fb4", "width": 4},
                "thickness": 0.82,
                "value": 90,
            },
        },
    }
    if reference is not None:
        indicator_config["delta"] = {
            "reference": reference_value,
            "suffix": "%",
            "valueformat": ".1f",
            "relative": False,
            "position": "bottom",
            "font": {"size": 17},
            "increasing": {"color": "#16815b"},
            "decreasing": {"color": "#ef4444"},
        }

    fig = go.Figure(data=[go.Indicator(**indicator_config)])
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font={"color": "#16324f"},
        margin=dict(l=18, r=18, t=88, b=18),
        height=320,
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 1.06,
                "showarrow": False,
                "text": f"<b>{escape(title)}</b>",
                "font": {"size": 24, "color": "#17365d"},
                "align": "center",
            }
        ],
    )
    return fig


def create_top_k_chart_v2(top_k_data: list):
    classes = [item[1] for item in top_k_data[:5]]
    probs = [item[2] * 100 for item in top_k_data[:5]]
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=classes,
            orientation="h",
            marker=dict(
                color=probs,
                colorscale=[[0, "#dbeafe"], [0.5, "#60a5fa"], [1, "#1565d8"]],
            ),
            text=[f"{prob:.1f}%" for prob in probs],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Top 5 nhãn mạnh nhất",
        xaxis_title="Độ chính xác (%)",
        yaxis_title="Nhãn",
        font={"color": "#16324f"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="y unified",
        height=320,
    )
    return fig


def create_class_distribution_chart_v2(top_k_data: list):
    frame = pd.DataFrame([
        {"Nhãn": item[1], "Độ chính xác": float(item[2]) * 100}
        for item in top_k_data[:5]
    ])
    fig = px.pie(
        frame,
        values="Độ chính xác",
        names="Nhãn",
        title="Tỷ trọng top 5",
        hole=0.52,
        color_discrete_sequence=["#1565d8", "#2f85f5", "#60a5fa", "#93c5fd", "#bfdbfe"],
    )
    fig.update_layout(
        font={"color": "#16324f"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
    )
    return fig


@st.cache_data
def load_training_dashboard() -> Dict[str, object]:
    def _safe_read_json(path: Path) -> Optional[Dict]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    detector_dir = DEFAULT_DEMO_DETECTOR_CHECKPOINT.parent
    classifier_dir = DEFAULT_DEMO_CLASSIFIER_CHECKPOINT.parent

    return {
        "detector_history": _safe_read_json(detector_dir / "history.json"),
        "detector_test": _safe_read_json(detector_dir / "test_metrics.json"),
        "classifier_history": _safe_read_json(classifier_dir / "history.json"),
        "classifier_test": _safe_read_json(classifier_dir / "test_metrics.json"),
        "previous_detector_test": _safe_read_json(
            Path("checkpoints") / "detection_mnv3_hardmining_ft_lr5e5_e3" / "test_metrics.json"
        ),
        "previous_classifier_test": _safe_read_json(
            Path("checkpoints") / "color_fusion_v1" / "test_metrics.json"
        ),
    }


def create_history_chart_v2(
    series_map: Dict[str, list],
    *,
    title: str,
    yaxis_title: str,
    percent: bool = False,
):
    palette = ["#1565d8", "#60a5fa", "#0f766e", "#93c5fd"]
    fig = go.Figure()
    for index, (label, values) in enumerate(series_map.items()):
        if not values:
            continue
        y_values = [(float(item) * 100) if percent else float(item) for item in values]
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(y_values) + 1)),
                y=y_values,
                mode="lines+markers",
                name=label,
                line=dict(color=palette[index % len(palette)], width=3),
                marker=dict(size=7),
            )
        )

    fig.update_layout(
        title=title,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#16324f"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=20, r=20, t=56, b=20),
        height=340,
    )
    fig.update_xaxes(title="Epoch", gridcolor="#e4eef8")
    fig.update_yaxes(title=yaxis_title, gridcolor="#e4eef8")
    return fig


def has_history_series(series_map: Dict[str, list]) -> bool:
    return any(bool(values) for values in series_map.values())


def render_model_performance_dashboard() -> None:
    dashboard = load_training_dashboard()
    detector_test = dashboard.get("detector_test") or {}
    classifier_test = dashboard.get("classifier_test") or {}
    detector_history = dashboard.get("detector_history") or {}
    classifier_history = dashboard.get("classifier_history") or {}
    previous_detector_test = dashboard.get("previous_detector_test") or {}
    previous_classifier_test = dashboard.get("previous_classifier_test") or {}

    detector_reference = previous_detector_test.get("f1")
    classifier_reference = previous_classifier_test.get("accuracy")
    detector_recall_reference = previous_detector_test.get("recall")
    classifier_top3_reference = previous_classifier_test.get("top3_accuracy")

    detector_delta = compute_delta_points(detector_test.get("f1"), detector_reference)
    classifier_delta = compute_delta_points(classifier_test.get("accuracy"), classifier_reference)
    recall_delta = compute_delta_points(detector_test.get("recall"), detector_recall_reference)
    top3_delta = compute_delta_points(classifier_test.get("top3_accuracy"), classifier_top3_reference)
    detector_delta_display = normalize_delta_points(detector_delta)
    classifier_delta_display = normalize_delta_points(classifier_delta)

    insight_parts = []
    if detector_delta_display not in (None, 0.0):
        insight_parts.append(f"F1 phát hiện {format_delta_points(detector_delta_display)}")
    if classifier_delta_display not in (None, 0.0):
        insight_parts.append(f"Top-1 {format_delta_points(classifier_delta_display)}")

    if detector_delta_display is not None and classifier_delta_display is not None:
        if detector_delta_display > 0 and classifier_delta_display < 0:
            insight_text = (
                f"Checkpoint hiện tại tốt hơn bản trước ở {insight_parts[0]}, "
                f"nhưng giảm ở {insight_parts[1]}. Khuyến nghị: kiểm tra lại bộ crop classifier."
            )
        else:
            insight_text = " | ".join(insight_parts)
    elif insight_parts:
        insight_text = " | ".join(insight_parts)
    else:
        insight_text = "Đang hiển thị số liệu hiện tại của checkpoint đang dùng."

    st.markdown(
        """
        <div class="section-intro">
            <div class="section-title">Hiệu năng mô hình</div>
            <p class="section-subtitle">Số liệu thật từ checkpoint đang dùng, ưu tiên so sánh nhanh và insight ngắn.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">Insight nhanh</div>
            <div class="insight-body">{escape(insight_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    support_cols = st.columns(2)
    with support_cols[0]:
        render_progress_metric_card(
            title="Recall phát hiện",
            value=detector_test.get("recall"),
            note="Khả năng bắt đủ viên trong ảnh",
            delta_points=recall_delta,
        )
    with support_cols[1]:
        render_progress_metric_card(
            title="Độ chính xác Top-3",
            value=classifier_test.get("top3_accuracy"),
            note="Mức ổn định của top-k",
            delta_points=top3_delta,
        )

    gauge_cols = st.columns(2)
    with gauge_cols[0]:
        st.plotly_chart(
            create_confidence_gauge_v2(
                float(detector_test.get("f1", 0.0)),
                "F1 phát hiện",
                reference=detector_reference,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
        detector_note = "Đang dùng checkpoint hiện tại."
        if detector_delta_display is not None:
            detector_note = f"So với checkpoint trước: {format_delta_points(detector_delta_display, zero_text='không đổi')}"
        st.caption(detector_note)
    with gauge_cols[1]:
        st.plotly_chart(
            create_confidence_gauge_v2(
                float(classifier_test.get("accuracy", 0.0)),
                "Độ chính xác Top-1",
                reference=classifier_reference,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
        classifier_note = "Đang dùng checkpoint hiện tại."
        if classifier_delta_display is not None:
            classifier_note = f"So với checkpoint trước: {format_delta_points(classifier_delta_display, zero_text='không đổi')}"
        st.caption(classifier_note)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        classifier_loss_series = {
            "Loss huấn luyện": classifier_history.get("train_loss", []),
            "Loss xác thực": classifier_history.get("val_loss", []),
        }
        if has_history_series(classifier_loss_series):
            st.plotly_chart(
                create_history_chart_v2(
                    classifier_loss_series,
                    title="Loss theo epoch",
                    yaxis_title="Loss",
                    percent=False,
                ),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.info("Checkpoint classifier hiện tại chưa có `history.json`, nên chưa hiển thị được loss theo epoch.")
    with chart_cols[1]:
        classifier_acc_series = {
            "Độ chính xác huấn luyện": classifier_history.get("train_acc", []),
            "Độ chính xác xác thực": classifier_history.get("val_acc", []),
        }
        if has_history_series(classifier_acc_series):
            st.plotly_chart(
                create_history_chart_v2(
                    classifier_acc_series,
                    title="Độ chính xác theo epoch",
                    yaxis_title="Độ chính xác (%)",
                    percent=True,
                ),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.info("Checkpoint classifier hiện tại mới có benchmark test sạch; accuracy theo epoch sẽ có sau khi hoàn tất một run train đầy đủ.")

    comparison_rows = [
        {
            "Checkpoint": "Hiện tại",
            "F1 phát hiện": format_metric_value(detector_test.get("f1")),
            "Recall": format_metric_value(detector_test.get("recall")),
            "Top-1": format_metric_value(classifier_test.get("accuracy")),
            "Top-3": format_metric_value(classifier_test.get("top3_accuracy")),
        }
    ]
    if previous_detector_test or previous_classifier_test:
        comparison_rows.append(
            {
                "Checkpoint": "Trước đó",
                "F1 phát hiện": format_metric_value(previous_detector_test.get("f1")),
                "Recall": format_metric_value(previous_detector_test.get("recall")),
                "Top-1": format_metric_value(previous_classifier_test.get("accuracy")),
                "Top-3": format_metric_value(previous_classifier_test.get("top3_accuracy")),
            }
        )

    st.markdown("### So sánh checkpoint")
    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

    st.plotly_chart(
        create_history_chart_v2(
            {
                "F1 phát hiện": detector_history.get("val_f1", []),
                "Recall phát hiện": detector_history.get("val_recall", []),
            },
            title="Detector theo epoch",
            yaxis_title="Tỷ lệ (%)",
            percent=True,
        ),
        width="stretch",
        config={"displayModeBar": False},
    )


@st.cache_data
def load_ui_benchmarks() -> Dict[str, object]:
    payload: Dict[str, object] = {
        "detector_f1": None,
        "detector_recall": None,
        "classifier_top1": None,
        "classifier_top3": None,
        "num_classes": None,
        "classifier_split_strategy": None,
        "classifier_test_samples": None,
        "classifier_test_images": None,
    }

    try:
        with open(DEFAULT_DEMO_DETECTOR_CHECKPOINT.parent / "test_metrics.json", "r", encoding="utf-8") as handle:
            detector_metrics = json.load(handle)
        payload["detector_f1"] = detector_metrics.get("f1")
        payload["detector_recall"] = detector_metrics.get("recall")
    except Exception:
        pass

    try:
        with open(DEFAULT_DEMO_CLASSIFIER_CHECKPOINT.parent / "test_metrics.json", "r", encoding="utf-8") as handle:
            classifier_metrics = json.load(handle)
        payload["classifier_top1"] = classifier_metrics.get("accuracy")
        payload["classifier_top3"] = classifier_metrics.get("top3_accuracy")
    except Exception:
        pass

    try:
        with open(DEFAULT_DEMO_CLASSIFIER_CHECKPOINT.parent / "dataset_summary.json", "r", encoding="utf-8") as handle:
            dataset_summary = json.load(handle)
        payload["num_classes"] = dataset_summary.get("num_classes")
        payload["classifier_split_strategy"] = dataset_summary.get("split_strategy")
        payload["classifier_test_images"] = dataset_summary.get("test_unique_source_images")
    except Exception:
        pass

    try:
        with open(DEFAULT_DEMO_CLASSIFIER_CHECKPOINT.parent / "test_metrics.json", "r", encoding="utf-8") as handle:
            classifier_metrics = json.load(handle)
        payload["classifier_test_samples"] = classifier_metrics.get("samples")
    except Exception:
        pass

    return payload


def format_metric_value(value: Optional[float], *, percent: bool = True, decimals: int = 1) -> str:
    if value is None:
        return "--"
    if percent:
        return f"{float(value) * 100:.{decimals}f}%"
    return f"{float(value):.{decimals}f}"


def normalize_delta_points(delta_points: Optional[float], *, decimals: int = 1) -> Optional[float]:
    if delta_points is None:
        return None
    rounded = round(float(delta_points), decimals)
    if rounded == 0:
        return 0.0
    return rounded


def format_delta_points(delta_points: Optional[float], *, decimals: int = 1, zero_text: str = "Không đổi") -> str:
    normalized = normalize_delta_points(delta_points, decimals=decimals)
    if normalized is None:
        return "Đang dùng"
    if normalized > 0:
        return f"+{normalized:.{decimals}f} điểm"
    if normalized < 0:
        return f"{normalized:.{decimals}f} điểm"
    return zero_text


def render_stat_card(label: str, value: str, caption: str, *, featured: bool = False) -> None:
    card_class = "quick-stat featured" if featured else "quick-stat"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="quick-stat-label">{escape(label)}</div>
            <div class="quick-stat-value">{escape(value)}</div>
            <div class="quick-stat-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_delta_points(current: Optional[float], reference: Optional[float]) -> Optional[float]:
    if current is None or reference is None:
        return None
    return (float(current) - float(reference)) * 100.0


def render_progress_metric_card(
    *,
    title: str,
    value: Optional[float],
    note: str,
    delta_points: Optional[float] = None,
) -> None:
    percentage = max(0.0, min(float(value or 0.0) * 100.0, 100.0))
    normalized_delta = normalize_delta_points(delta_points)
    if normalized_delta is None:
        badge_class = "neutral"
        badge_text = "Đang dùng"
    elif normalized_delta > 0:
        badge_class = "positive"
        badge_text = format_delta_points(normalized_delta)
    elif normalized_delta < 0:
        badge_class = "negative"
        badge_text = format_delta_points(normalized_delta)
    else:
        badge_class = "neutral"
        badge_text = "Không đổi"

    st.markdown(
        f"""
        <div class="progress-card">
            <div class="progress-card-title">{escape(title)}</div>
            <div class="progress-card-value">{percentage:.1f}%</div>
            <div class="progress-meta">
                <span class="metric-badge {badge_class}">{escape(badge_text)}</span>
                <span class="progress-card-note">{escape(note)}</span>
            </div>
            <div class="progress-rail">
                <div class="progress-fill" style="width: {percentage:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_prompt(title: str, subtitle: str, notes: Tuple[str, ...]) -> None:
    notes_html = "".join(f"<span class='upload-note'>{escape(note)}</span>" for note in notes)
    st.markdown(
        f"""
        <div class="upload-shell">
            <div class="upload-icon">↑</div>
            <div class="upload-copy">
                <div class="upload-title">{escape(title)}</div>
                <p class="upload-subtitle">{escape(subtitle)}</p>
                <div class="upload-note-row">{notes_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_result_status_copy(feedback_class: str) -> str:
    if feedback_class == "strong":
        return "Kết quả đáng tin cậy"
    if feedback_class == "medium":
        return "Nên kiểm tra thêm"
    return "Độ tin cậy thấp"


def get_score_badge_class(score_percent: float) -> str:
    if score_percent >= 90:
        return "score-high"
    if score_percent >= 75:
        return "score-medium"
    return "score-low"


def render_detail_table_html(title: str, headers: Tuple[str, ...], rows_html: str) -> None:
    st.html(
        f"""
        <div class="detail-table-shell">
            <div class="detail-table-title">{escape(title)}</div>
            <table class="detail-table">
                <thead>
                    <tr>{''.join(f'<th>{escape(header)}</th>' for header in headers)}</tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
    )


def render_detection_detail_table(detections: list) -> None:
    rows = []
    for index, item in enumerate(detections):
        label = str(item.get("display_label", "--"))
        detector_origin = str(item.get("detector_display_label", label))
        score_percent = float(item.get("score", 0.0)) * 100.0
        score_class = get_score_badge_class(score_percent)
        source = str(item.get("source", "detector"))
        extra_line = ""
        if detector_origin and detector_origin != label:
            extra_line = f"<div class='detail-sub'>Detector gốc: {escape(detector_origin)}</div>"
        rows.append(
            f"""
            <tr>
                <td><span class="detail-main">Viên {index + 1}</span></td>
                <td>
                    <div class="detail-main">{escape(label)}</div>
                    {extra_line}
                </td>
                <td><span class="detail-badge {score_class}">{score_percent:.2f}%</span></td>
                <td><span class="detail-badge source">{escape(source)}</span></td>
            </tr>
            """
        )
    render_detail_table_html("Từng viên", ("Viên", "Kết quả", "Độ tin cậy", "Nguồn"), "".join(rows))


def render_topk_detail_table(top_5: list, confidence_threshold: float) -> None:
    rows = []
    for index, item in enumerate(top_5):
        label = str(item[1])
        score_percent = float(item[2]) * 100.0
        score_class = get_score_badge_class(score_percent)
        review_text = "Mạnh" if float(item[2]) >= confidence_threshold else "Cần xem thêm"
        rows.append(
            f"""
            <tr>
                <td><span class="detail-main">#{index + 1}</span></td>
                <td><div class="detail-main">{escape(label)}</div></td>
                <td><span class="detail-badge {score_class}">{score_percent:.2f}%</span></td>
                <td><span class="detail-badge source">{escape(review_text)}</span></td>
            </tr>
            """
        )
    render_detail_table_html("Top 5 nhãn", ("Hạng", "Nhãn", "Độ tin cậy", "Trạng thái"), "".join(rows))


def render_compare_preview(original_image: np.ndarray, preview_path: Path) -> None:
    st.markdown(
        """
        <div class="compare-shell">
            <div class="compare-title">So sánh nhanh</div>
            <p class="compare-subtitle">Đối chiếu ảnh gốc và ảnh detect để kiểm tra số viên, vị trí box và mức bám đúng.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    compare_cols = st.columns(2)
    with compare_cols[0]:
        st.image(original_image, caption="Ảnh gốc", width="stretch")
    with compare_cols[1]:
        st.image(str(preview_path), caption="Ảnh detect", width="stretch")


def render_hero(benchmarks: Dict[str, object]) -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-eyebrow">VAIPE AI Workspace</div>
            <div class="hero-title">Nhận diện viên thuốc từ ảnh thật với luồng kết quả rõ ràng hơn</div>
            <p class="hero-subtitle">
                Giao diện này ưu tiên trải nghiệm đọc kết quả nhanh: xem ảnh, xem kết quả chính, xem preview detect,
                rồi mới xuống biểu đồ và bảng chi tiết. Luồng hiện tại đang dùng detector cho ảnh nhiều viên và classifier cho crop đơn.
            </p>
            <div class="hero-chip-row">
                <span class="hero-chip">Ảnh đơn và nhiều viên</span>
                <span class="hero-chip">Preview detect trực tiếp</span>
                <span class="hero-chip">Top nhãn rõ ràng hơn</span>
                <span class="hero-chip">Xuất JSON và CSV</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(4)
    with stat_cols[0]:
        render_stat_card("Detector F1", format_metric_value(benchmarks.get("detector_f1")), "Độ chính xác cho ảnh nhiều viên")
    with stat_cols[1]:
        render_stat_card("Detector Recall", format_metric_value(benchmarks.get("detector_recall")), "Khả năng bắt đủ viên trong ảnh")
    with stat_cols[2]:
        render_stat_card("Classifier Top-1", format_metric_value(benchmarks.get("classifier_top1")), "Độ chính xác trên crop đơn")
    with stat_cols[3]:
        render_stat_card("Số lớp thuốc", str(benchmarks.get("num_classes") or "--"), "Nhãn hiện có trong bộ model")


def render_empty_upload_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="hero-eyebrow">Bắt đầu phân tích</div>
            <div class="empty-state-title">Tải một ảnh lên để xem toàn bộ luồng phân tích ở đây</div>
            <p class="empty-state-text">
                Khu vực này sẽ hiển thị ảnh đầu vào, kết quả nhận diện chính, preview detect, mức tin cậy và bảng phân tích chi tiết.
                Nếu ảnh có nhiều viên, app sẽ hiển thị luôn số viên detect được.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tip_cols = st.columns(3)
    tip_payload = [
        ("Ảnh nên có gì", "Chụp đủ sáng, giữ viên thuốc rõ nét và tránh bóng đổ che mép viên."),
        ("Khi nào detector tốt hơn", "Ảnh nhiều viên nên có khoảng cách tương đối giữa các viên để box ổn định hơn."),
        ("Khi nào nên chụp lại", "Nếu confidence thấp hoặc thiếu viên detect, hãy thử một ảnh rõ hơn hoặc nền gọn hơn."),
    ]
    for column, (title, text) in zip(tip_cols, tip_payload):
        with column:
            st.markdown(
                f"""
                <div class="tip-card">
                    <h4>{escape(title)}</h4>
                    <p>{escape(text)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def get_prediction_feedback(prediction: Dict, confidence_threshold: float) -> Tuple[str, str, str]:
    confidence = float(prediction.get("confidence", 0.0))
    if confidence >= 0.92:
        return "Rất chắc chắn", "strong", "Mức tin cậy đang rất tốt cho kết quả chính."
    if confidence >= max(0.75, confidence_threshold):
        return "Khá ổn", "medium", "Kết quả đủ mạnh để tham khảo trực tiếp, nhưng vẫn nên liếc qua top nhãn."
    return "Cần kiểm tra", "low", "Nên xem thêm top nhãn hoặc thử lại bằng ảnh rõ hơn."


def render_prediction_summary(prediction: Dict, confidence_threshold: float) -> None:
    feedback_title, feedback_class, feedback_desc = get_prediction_feedback(prediction, confidence_threshold)
    is_detector_pipeline = prediction.get("mode") == "multi_pill_detection"
    mode_label = "Detector + classifier" if is_detector_pipeline else "Crop classifier"
    chips = [
        f"Ngưỡng hiển thị: {confidence_threshold:.0%}",
        f"Luồng: {mode_label}",
    ]
    if is_detector_pipeline:
        chips.append(f"Số viên detect: {prediction.get('num_detections', 0)}")
    else:
        chips.append(f"Top nhãn hiển thị: {min(5, len(prediction.get('top_5', [])))}")

    if is_detector_pipeline and int(prediction.get("suppressed_detections", 0)) > 0:
        chips.append(f"Gop box trung: {int(prediction.get('suppressed_detections', 0))}")

    chip_html = "".join(f"<span class='tag-chip'>{escape(text)}</span>" for text in chips)
    st.markdown(
        f"""
        <div class="result-shell {feedback_class}">
            <div class="result-eyebrow">Kết quả nhận diện chính</div>
            <div class="result-title">{escape(str(prediction['class_name']))}</div>
            <p class="result-summary">
                Confidence hiện tại là <span class="result-highlight">{prediction['confidence']:.1%}</span>.
                <span class="result-highlight">{escape(feedback_title)}</span>: {escape(feedback_desc)}
            </p>
            <div class="tag-row">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Độ tin cậy", f"{prediction['confidence']:.1%}")
    with metric_cols[1]:
        st.metric("Luồng xử lý", mode_label)
    with metric_cols[2]:
        st.metric("Số viên", str(prediction.get("num_detections", 1)))
    with metric_cols[3]:
        st.metric("Đánh giá nhanh", feedback_title)


def render_hero_compact(benchmarks: Dict[str, object]) -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-eyebrow">VAIPE AI</div>
            <div class="hero-title">VAIPE AI – Phân tích viên thuốc bằng AI</div>
            <p class="hero-subtitle">
                Tải ảnh, nhận diện thuốc, theo dõi hiệu năng mô hình.
            </p>
            <div class="hero-chip-row">
                <span class="hero-chip primary">Nhận diện nhanh</span>
                <span class="hero-chip secondary">Ảnh đơn / nhiều viên</span>
                <span class="hero-chip secondary">Hiệu năng mô hình</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(4)
    with stat_cols[0]:
        render_stat_card("F1 phát hiện", format_metric_value(benchmarks.get("detector_f1")), "Ảnh nhiều viên")
    with stat_cols[1]:
        render_stat_card("Recall", format_metric_value(benchmarks.get("detector_recall")), "Bắt đủ viên")
    with stat_cols[2]:
        render_stat_card("Top-1", format_metric_value(benchmarks.get("classifier_top1")), "Crop đơn")
    with stat_cols[3]:
        render_stat_card("Top-3", format_metric_value(benchmarks.get("classifier_top3")), "Mức ổn định")

def render_empty_upload_state_compact() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="hero-eyebrow">Sẵn sàng</div>
            <div class="empty-state-title">Kết quả sẽ hiện ở đây</div>
            <p class="empty-state-text">
                Sau khi tải ảnh, app sẽ hiển thị ảnh đầu vào, viên nổi bật nhất, preview detect và các phân tích cần thiết.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_summary_compact(prediction: Dict, confidence_threshold: float) -> None:
    feedback_title, feedback_class, _ = get_prediction_feedback(prediction, confidence_threshold)
    is_detector_pipeline = prediction.get("mode") == "multi_pill_detection"
    num_detections = int(prediction.get("num_detections", 1))
    has_multiple_detections = is_detector_pipeline and num_detections > 1
    pipeline_label = "Detector + classifier" if is_detector_pipeline else "Crop classifier"
    eyebrow = "Viên nổi bật nhất" if has_multiple_detections else ("Kết quả phát hiện" if is_detector_pipeline else "Kết quả chính")
    status_copy = get_result_status_copy(feedback_class)
    tags = [
        status_copy,
        f"Luồng: {pipeline_label}",
    ]
    if is_detector_pipeline:
        tags.append(f"Phát hiện: {num_detections} viên")

    if is_detector_pipeline and int(prediction.get("suppressed_detections", 0)) > 0:
        tags.append(f"Gop trung: -{int(prediction.get('suppressed_detections', 0))} box")

    confidence_width = max(6.0, min(float(prediction["confidence"]) * 100.0, 100.0))
    summary_text = (
        "Đang hiển thị viên có điểm cao nhất trong ảnh."
        if has_multiple_detections
        else ("Đã detect 1 viên rõ trong ảnh." if is_detector_pipeline else "Mức tin cậy đủ rõ để tham khảo nhanh.")
    )

    chip_html = "".join(
        f"<span class='result-status-pill {feedback_class}'>{escape(text)}</span>" if index == 0
        else f"<span class='tag-chip'>{escape(text)}</span>"
        for index, text in enumerate(tags)
    )
    source_label = pipeline_label
    st.markdown(
        f"""
        <div class="result-shell {feedback_class}">
            <div class="result-eyebrow">{eyebrow}</div>
            <div class="result-title">{escape(str(prediction['class_name']))}</div>
            <p class="result-summary">
                {summary_text}
            </p>
            <div class="tag-row">{chip_html}</div>
            <div class="confidence-rail">
                <div class="confidence-fill" style="width: {confidence_width:.1f}%"></div>
            </div>
            <div class="result-kpi-row">
                <div class="result-kpi">
                    <div class="result-kpi-label">Độ tin cậy</div>
                    <div class="result-kpi-value">{prediction['confidence']:.1%}</div>
                </div>
                <div class="result-kpi">
                    <div class="result-kpi-label">Số viên</div>
                    <div class="result-kpi-value">{prediction.get('num_detections', 1)}</div>
                </div>
                <div class="result-kpi">
                    <div class="result-kpi-label">Nguồn</div>
                    <div class="result-kpi-value status">{escape(source_label)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; margin-bottom: 40px;'>
        <h1>💊 HỆ THỐNG PHÂN TÍCH VIÊN THUỐC</h1>
        <p style='font-size: 18px; color: #B0B8C1; margin: 10px 0;'>
            Công nghệ AI tự động nhận diện & phân tích viên thuốc
        </p>
        <hr style='border-color: #00D9FF; border-width: 2px; margin: 20px 0;'>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Settings
    with st.sidebar:
        st.markdown("## ⚙️ Cài Đặt")
        confidence_threshold = st.slider(
            "Ngưỡng Tin Cậy (Confidence)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Chỉ hiển thị kết quả có độ tin cậy cao hơn ngưỡng này"
        )
        
        show_advanced = st.checkbox("🔬 Hiển Thị Chi Tiết Nâng Cao", value=True)
        
        st.markdown("---")
        st.markdown("## 📊 Thông Tin Mô Hình")
        st.info("""
        **Mô Hình**: ResNet18 + Color Fusion
        
        **Độ Chính Xác**: ~92-94%
        
        **Số Loại**: ~200 loại thuốc
        
        **Kích Thước Input**: 160×160 px
        """)
    
    # Tab 1: Upload & Analyze
    tab1, tab2, tab3 = st.tabs(["📸 Chụp & Phân Tích", "📊 Phân Tích Hàng Loạt", "❓ Trợ Giúp"])
    
    with tab1:
        st.markdown("### 📸 Tải Ảnh Viên Thuốc")
        
        st.markdown("#### 📤 Tải Ảnh Lên")
        uploaded_file = st.file_uploader(
            "Chọn ảnh viên thuốc...",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Kéo và thả ảnh hoặc nhấp để chọn"
        )
        
        # Process uploaded image
        image_to_process = None
        image_source = None
        
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            image_source = "upload"
        
        if image_to_process is not None:
            # Convert to numpy
            image_array = normalize_uploaded_image(np.array(image_to_process))
            
            st.markdown("---")
            
            # Show image
            col_img, col_analysis = st.columns([1, 1])
            
            with col_img:
                st.markdown("#### 🖼️ Ảnh Đầu Vào")
                st.image(image_array, width='stretch')
                
                # Image stats
                st.info(f"""
                **Thông Tin Ảnh:**
                - Kích thước: {image_array.shape[1]}×{image_array.shape[0]} px
                - Định dạng: {image_to_process.format}
                """)
            
            with col_analysis:
                st.markdown("#### 🔍 Kết Quả Phân Tích")
                
                # Load models and predict
                try:
                    model_info = load_models()
                    
                    with st.spinner("🤖 Đang phân tích viên thuốc..."):
                        prediction = predict_pill(image_array, model_info)
                    
                    # Show main result
                    st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, #1a2332, #16202A);
                        border: 2px solid #00D9FF;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 10px 0;
                        box-shadow: 0 0 20px rgba(0, 217, 255, 0.2);
                    '>
                        <h3 style='color: #FFFFFF; margin: 0 0 10px 0;'>🎯 Kết Quả Nhận Diện</h3>
                        <p style='font-size: 24px; color: #00D9FF; margin: 10px 0; font-weight: bold;'>
                            {prediction['class_name']}
                        </p>
                        <p style='font-size: 16px; color: #B0B8C1; margin: 10px 0;'>
                            Độ Tin Cậy: <span style='color: #2ECC71; font-weight: bold;'>{prediction['confidence']:.1%}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    if prediction.get("warning"):
                        st.warning(prediction["warning"])

                    if prediction.get("mode") == "multi_pill_detection":
                        st.info(f"Phát hiện {prediction['num_detections']} viên thuốc trong ảnh.")
                        if int(prediction.get("suppressed_detections", 0)) > 0:
                            st.caption(f"Da gop {int(prediction.get('suppressed_detections', 0))} box chong lan co score thap hon.")
                        top_labels = prediction.get("top_labels", [])[:3]
                        if top_labels:
                            st.caption("Các nhãn nổi bật: " + ", ".join(top_labels))
                        preview_path = prediction.get("preview_path")
                        if preview_path and Path(preview_path).exists():
                            st.image(str(preview_path), caption="Ảnh sau khi detect + refine", width='stretch')
                    
                    # Store prediction in session for later use
                    st.session_state.last_prediction = prediction
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
                    st.warning("💡 Kiểm tra xem model có được load không")
            
            st.markdown("---")
            
            # Detailed Analysis
            if show_advanced and st.session_state.last_prediction:
                prediction = st.session_state.last_prediction
                
                st.markdown("### 📈 Phân Tích Chi Tiết")
                
                chart_col1, chart_col2 = st.columns([1, 1])
                
                with chart_col1:
                    st.plotly_chart(
                        create_confidence_gauge(prediction['confidence']),
                        width='stretch',
                        config={"displayModeBar": False}
                    )
                
                with chart_col2:
                    if prediction.get("mode") == "multi_pill_detection":
                        st.plotly_chart(
                            create_detection_score_chart(prediction.get("detections", [])),
                            width='stretch',
                            config={"displayModeBar": False}
                        )
                    else:
                        st.plotly_chart(
                            create_top_k_chart(prediction['top_5']),
                            width='stretch',
                            config={"displayModeBar": False}
                        )
                
                # Detail table
                if prediction.get("mode") == "multi_pill_detection":
                    st.markdown("### Chi Tiết Từng Viên")
                    top_5_df = pd.DataFrame([
                        {
                            "Thứ tự": i + 1,
                            "Nhãn": item["display_label"],
                            "Độ Tin Cậy": f"{float(item['score']):.2%}",
                            "Nguồn": item.get("source", "detector"),
                            "Detector Gốc": item.get("detector_display_label", item["display_label"]),
                        }
                        for i, item in enumerate(prediction.get("detections", []))
                    ])
                else:
                    st.markdown("### 🏆 Top 5 Dự Đoán")
                    top_5_df = pd.DataFrame([
                        {
                            "Hạng": i + 1,
                            "Loại Thuốc": item[1],
                            "Độ Tin Cậy": f"{item[2]:.2%}",
                            "Trạng Thái": "✅ Khả Năng Cao" if item[2] >= confidence_threshold else "⚠️ Thấp"
                        }
                        for i, item in enumerate(prediction['top_5'])
                    ])
                
                st.dataframe(
                    top_5_df,
                    width='stretch',
                    hide_index=True,
                )
                
                if prediction.get("mode") != "multi_pill_detection":
                    st.markdown("### 📊 Phân Phối Loại Thuốc")
                    st.plotly_chart(
                        create_class_distribution_chart(prediction['probabilities']),
                        width='stretch',
                        config={"displayModeBar": False}
                    )
                
                # Download results
                st.markdown("### 💾 Tải Kết Quả")
                
                results_json = json.dumps(prediction, indent=2, ensure_ascii=False)
                results_csv = top_5_df.to_csv(index=False)
                
                col_json, col_csv = st.columns(2)
                
                with col_json:
                    st.download_button(
                        label="📄 Tải JSON",
                        data=results_json,
                        file_name=f"ketqua_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col_csv:
                    st.download_button(
                        label="📊 Tải CSV",
                        data=results_csv,
                        file_name=f"ketqua_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # Tab 2: Batch Analysis
    with tab2:
        st.markdown("### 📊 Phân Tích Hàng Loạt")
        st.info("📤 Tải nhiều ảnh để phân tích cùng lúc")
        
        batch_files = st.file_uploader(
            "Chọn nhiều ảnh...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        
        if batch_files:
            st.markdown(f"🔄 Đang phân tích **{len(batch_files)}** ảnh...")
            
            results_list = []
            model_info = load_models()
            
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            for idx, file in enumerate(batch_files):
                try:
                    image = Image.open(file)
                    image_array = np.array(image)
                    image_array = normalize_uploaded_image(image_array)
                    
                    # Make prediction
                    prediction = predict_pill(image_array, model_info)
                    
                    results_list.append({
                        "Tên Ảnh": file.name,
                        "Loại Thuốc": prediction['class_name'],
                        "Độ Tin Cậy": f"{prediction['confidence']:.2%}",
                        "ID": prediction['class_id'],
                        "Số Viên": prediction.get('num_detections', 1),
                    })
                    
                    progress_text.text(f"✅ Đã phân tích: {idx + 1}/{len(batch_files)}")
                except Exception as e:
                    results_list.append({
                        "Tên Ảnh": file.name,
                        "Loại Thuốc": f"❌ Lỗi: {str(e)[:20]}",
                        "Độ Tin Cậy": "N/A",
                        "ID": "-",
                    })
                    progress_text.text(f"⚠️ Đã xử lý: {idx + 1}/{len(batch_files)}")
                
                progress_bar.progress((idx + 1) / len(batch_files))
            
            progress_text.empty()
            progress_bar.empty()
            
            # Display results table
            st.markdown("### 📋 Kết Quả")
            results_df = pd.DataFrame(results_list)
            st.dataframe(results_df, width='stretch', hide_index=True)
            
            # Statistics
            st.markdown("### 📈 Thống Kê")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Tổng Ảnh", len(batch_files))
            
            with col_stat2:
                success_count = len([r for r in results_list if "❌" not in str(r["Loại Thuốc"])])
                st.metric("Phân Tích Thành Công", success_count)
            
            with col_stat3:
                st.metric("Tỷ Lệ Thành Công", f"{success_count/len(batch_files)*100:.0f}%")
            
            # Download batch results
            st.markdown("### 💾 Tải Kết Quả")
            st.download_button(
                label="📥 Tải Kết Quả CSV",
                data=results_df.to_csv(index=False),
                file_name=f"loakhang_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Tab 3: Help
    with tab3:
        st.markdown("### ❓ Trợ Giúp & Câu Hỏi Thường Gặp")
        
        with st.expander("🤔 Cách Sử Dụng?", expanded=True):
            st.markdown("""
            1. **📤 Tải Ảnh**: Nhấp "Tải Ảnh Lên" hoặc chụp từ camera
            2. **🔍 Xem Kết Quả**: Hệ thống tự động nhận diện loại thuốc
            3. **📊 Chi Tiết**: Kiểm tra top 5 dự đoán + biểu đồ
            4. **💾 Lưu**: Tải kết quả dưới dạng JSON hoặc CSV
            """)
        
        with st.expander("📊 Biểu Đồ Có Ý Nghĩa Gì?"):
            st.markdown("""
            - **Gauge (Kim Đo)**: Độ tin cậy của dự đoán (0-100%)
            - **Biểu Đồ Cột**: Xác suất của 5 loại thuốc hàng đầu
            - **Biểu Đồ Tròn**: Phân phối xác suất giữa các loại
            """)
        
        with st.expander("⚙️ Thông Tin Mô Hình"):
            st.markdown("""
            **Kiến Trúc**: ResNet18 + Color Fusion (CG-IMIF)
            
            **Độ Chính Xác**: 92-94% trên tập test
            
            **Dữ Liệu**: ~30,000 ảnh viên thuốc từ VAIPE dataset
            
            **Số Loại**: ~200 loại thuốc khác nhau
            
            **Kích Thước**: 160×160 pixel
            """)
        
        with st.expander("💡 Mẹo & Thủ Thuật"):
            st.markdown("""
            1. **Ảnh Rõ**: Chụp ảnh sáng, rõ ràng, không mờ
            2. **Đặt Vị Trí**: Đặt viên thuốc vào giữa ảnh
            3. **Nền Trung Lập**: Nền sáng hoặc tối đều được
            4. **Nhiều Góc**: Nếu không chắc, chụp từ nhiều góc
            5. **Hàng Loạt**: Dùng chế độ hàng loạt để phân tích nhanh
            """)

def render_hero_compact(benchmarks: Dict[str, object]) -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-eyebrow">VAIPE AI</div>
            <div class="hero-title">VAIPE AI \u2013 Nh\u1eadn di\u1ec7n vi\u00ean thu\u1ed1c nhanh t\u1eeb \u1ea3nh</div>
            <p class="hero-subtitle">
                T\u1ea3i \u1ea3nh \u0111\u1ec3 nh\u1eadn di\u1ec7n thu\u1ed1c, xem g\u1ee3i \u00fd tin c\u1eady v\u00e0 theo d\u00f5i hi\u1ec7u n\u0103ng m\u00f4 h\u00ecnh trong c\u00f9ng m\u1ed9t m\u00e0n h\u00ecnh.
            </p>
            <div class="hero-chip-row">
                <span class="hero-chip primary">S\u1eb5n s\u00e0ng nh\u1eadn di\u1ec7n</span>
                <span class="hero-chip secondary">\u1ea2nh \u0111\u01a1n / nhi\u1ec1u vi\u00ean</span>
                <span class="hero-chip secondary">K\u1ebft qu\u1ea3 + hi\u1ec7u n\u0103ng</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(4)
    with stat_cols[0]:
        render_stat_card("F1 ph\u00e1t hi\u1ec7n", format_metric_value(benchmarks.get("detector_f1")), "\u1ea2nh nhi\u1ec1u vi\u00ean")
    with stat_cols[1]:
        render_stat_card(
            "Recall",
            format_metric_value(benchmarks.get("detector_recall")),
            "Kh\u1ea3 n\u0103ng b\u1eaft \u0111\u1ee7 vi\u00ean",
            featured=True,
        )
    with stat_cols[2]:
        render_stat_card("Top-1", format_metric_value(benchmarks.get("classifier_top1")), "Nh\u1eadn di\u1ec7n \u1ea3nh \u0111\u01a1n")
    with stat_cols[3]:
        render_stat_card("Top-3", format_metric_value(benchmarks.get("classifier_top3")), "G\u1ee3i \u00fd \u1ed5n \u0111\u1ecbnh")

    if benchmarks.get("classifier_split_strategy") == "grouped_source_image":
        sample_text = f"{int(benchmarks['classifier_test_samples']):,} crop" if benchmarks.get("classifier_test_samples") else "-- crop"
        image_text = f"{int(benchmarks['classifier_test_images']):,} \u1ea3nh g\u1ed1c" if benchmarks.get("classifier_test_images") else "-- \u1ea3nh g\u1ed1c"
        st.caption(f"Benchmark classifier: held-out split theo \u1ea3nh g\u1ed1c ({sample_text}, {image_text} test).")


def main_v2():
    inject_streamlit_header_cleanup()
    benchmarks = load_ui_benchmarks()
    render_hero_compact(benchmarks)
    detector_checkpoint_name = DEFAULT_DEMO_DETECTOR_CHECKPOINT.parent.name.replace("_", " ")
    try:
        detector_updated_at = datetime.fromtimestamp(DEFAULT_DEMO_DETECTOR_CHECKPOINT.stat().st_mtime).strftime("%d/%m/%Y")
    except Exception:
        detector_updated_at = "--"

    with st.sidebar:
        st.markdown("### ⚙ Điều khiển")
        st.caption("Tùy chỉnh ngưỡng và mức chi tiết hiển thị.")
        confidence_threshold = st.slider(
            "Ngưỡng tin cậy",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
            help="Dùng để đánh dấu kết quả mạnh hoặc cần xem thêm.",
        )
        st.markdown(f"<span class='sidebar-badge'>Ngưỡng hiện tại: {confidence_threshold:.2f}</span>", unsafe_allow_html=True)
        range_cols = st.columns(2)
        with range_cols[0]:
            st.caption("Min 0.00")
        with range_cols[1]:
            st.caption("Max 1.00")
        show_advanced = st.toggle("Hiển thị phân tích chi tiết", value=False, help="Bật để xem chart và bảng phân tích.")

        st.markdown("---")
        st.markdown("### ◫ Mô hình")
        st.markdown(
            f"""
            <div class="sidebar-stack">
                <div class="sidebar-card-min">
                    <div class="sidebar-title">Trạng thái hệ thống</div>
                    <div class="sidebar-status-row">
                        <span class="sidebar-inline-badge ready">Model sẵn sàng</span>
                        <span class="sidebar-inline-badge">Ngưỡng {confidence_threshold:.2f}</span>
                        <span class="sidebar-inline-badge">Chi tiết {'Bật' if show_advanced else 'Tắt'}</span>
                    </div>
                    <div class="sidebar-row">
                        <span class="sidebar-label">Checkpoint</span>
                        <span class="sidebar-value compact">{escape(detector_checkpoint_name)}</span>
                    </div>
                    <div class="sidebar-row">
                        <span class="sidebar-label">Cập nhật</span>
                        <span class="sidebar-value compact">{detector_updated_at}</span>
                    </div>
                </div>
                <div class="sidebar-card-min">
                    <div class="sidebar-title">Chỉ số nhanh</div>
                    <div class="sidebar-row">
                        <span class="sidebar-label">F1 phát hiện</span>
                        <span class="sidebar-value">{format_metric_value(benchmarks.get('detector_f1'))}</span>
                    </div>
                    <div class="sidebar-row">
                        <span class="sidebar-label">Top-1 crop</span>
                        <span class="sidebar-value">{format_metric_value(benchmarks.get('classifier_top1'))}</span>
                    </div>
                    <div class="sidebar-row">
                        <span class="sidebar-label">Số lớp</span>
                        <span class="sidebar-value">{benchmarks.get('num_classes') or '--'}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3 = st.tabs(["Nhận diện ảnh", "Phân tích hàng loạt", "Hiệu năng mô hình"])

    with tab1:
        render_upload_prompt(
            "Tải ảnh để bắt đầu nhận diện",
            "Kéo thả ảnh hoặc bấm Browse files để xem kết quả ngay.",
            ("Hỗ trợ JPG, PNG, WEBP", "Ảnh 1 viên hoặc nhiều viên", "Ưu tiên ảnh sáng, rõ nét"),
        )
        uploaded_file = st.file_uploader(
            "Tải ảnh viên thuốc",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Hỗ trợ kéo thả hoặc chọn file trực tiếp.",
            key="single_uploader_v2",
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.session_state.last_prediction = None
            render_empty_upload_state_compact()
        else:
            image_to_process = Image.open(uploaded_file)
            image_array = normalize_uploaded_image(np.array(image_to_process))

            col_img, col_analysis = st.columns([1.05, 0.95])

            with col_img:
                st.markdown(
                    """
                    <div class="section-intro">
                        <div class="section-title">Ảnh đầu vào</div>
                        <p class="section-subtitle">Kiểm tra nhanh trước khi đọc kết quả.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(image_array, width="stretch")
                info_cols = st.columns(2)
                with info_cols[0]:
                    st.metric("Kích thước", f"{image_array.shape[1]} x {image_array.shape[0]}")
                with info_cols[1]:
                    st.metric("Định dạng", image_to_process.format or "Không rõ")

            with col_analysis:
                st.markdown(
                    """
                    <div class="section-intro">
                        <div class="section-title">Kết quả</div>
                        <p class="section-subtitle">Nếu ảnh có nhiều viên, thẻ này hiển thị viên nổi bật nhất.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                try:
                    model_info = load_models()
                    with st.spinner("Đang phân tích ảnh..."):
                        prediction = predict_pill(image_array, model_info)

                    st.session_state.last_prediction = prediction
                    render_prediction_summary_compact(prediction, confidence_threshold)

                    if prediction.get("warning"):
                        st.warning(prediction["warning"])

                    if prediction.get("mode") == "multi_pill_detection":
                        top_labels = prediction.get("top_labels", [])[:4]
                        if top_labels:
                            st.caption("Các viên đã nhận diện: " + " | ".join(top_labels))
                except Exception as exc:
                    st.session_state.last_prediction = None
                    st.error(f"Lỗi khi phân tích ảnh: {exc}")
                    st.warning("Hãy thử lại hoặc kiểm tra checkpoint nếu lỗi lặp lại.")

            if st.session_state.last_prediction and st.session_state.last_prediction.get("mode") == "multi_pill_detection":
                preview_path = st.session_state.last_prediction.get("preview_path")
                if preview_path and Path(preview_path).exists():
                    render_compare_preview(image_array, Path(preview_path))

            if show_advanced and st.session_state.last_prediction:
                prediction = st.session_state.last_prediction
                st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
                st.markdown("### Chi tiết phân tích")

                chart_col1, chart_col2 = st.columns([1, 1])
                with chart_col1:
                    st.plotly_chart(
                        create_confidence_gauge_v2(prediction["confidence"]),
                        width="stretch",
                        config={"displayModeBar": False},
                    )
                with chart_col2:
                    if prediction.get("mode") == "multi_pill_detection":
                        st.plotly_chart(
                            create_detection_score_chart_v2(prediction.get("detections", [])),
                            width="stretch",
                            config={"displayModeBar": False},
                        )
                    else:
                        st.plotly_chart(
                            create_top_k_chart_v2(prediction["top_5"]),
                            width="stretch",
                            config={"displayModeBar": False},
                        )

                if prediction.get("mode") == "multi_pill_detection":
                    detail_export_df = pd.DataFrame([
                        {
                            "Thứ tự": index + 1,
                            "Nhãn cuối": item["display_label"],
                            "Độ tin cậy": f"{float(item['score']):.2%}",
                            "Nguồn": item.get("source", "detector"),
                            "Detector gốc": item.get("detector_display_label", item["display_label"]),
                        }
                        for index, item in enumerate(prediction.get("detections", []))
                    ])
                    render_detection_detail_table(prediction.get("detections", []))
                else:
                    detail_export_df = pd.DataFrame([
                        {
                            "Hạng": index + 1,
                            "Nhãn": item[1],
                            "Độ tin cậy": f"{item[2]:.2%}",
                            "Đánh giá": "Mạnh" if item[2] >= confidence_threshold else "Cần xem thêm",
                        }
                        for index, item in enumerate(prediction["top_5"])
                    ])
                    render_topk_detail_table(prediction["top_5"], confidence_threshold)

                if prediction.get("mode") != "multi_pill_detection":
                    st.markdown("### Phân bổ xác suất")
                    st.plotly_chart(
                        create_class_distribution_chart_v2(prediction["top_5"]),
                        width="stretch",
                        config={"displayModeBar": False},
                    )

                st.markdown("### Xuất kết quả")
                results_json = json.dumps(prediction, indent=2, ensure_ascii=False)
                results_csv = detail_export_df.to_csv(index=False)
                download_cols = st.columns(2)
                with download_cols[0]:
                    st.download_button(
                        label="Xuất JSON",
                        data=results_json,
                        file_name=f"ketqua_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )
                with download_cols[1]:
                    st.download_button(
                        label="Xuất CSV",
                        data=results_csv,
                        file_name=f"ketqua_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

    with tab2:
        render_upload_prompt(
            "Tải nhiều ảnh để xử lý hàng loạt",
            "Phù hợp khi cần kiểm tra nhanh cả một nhóm ảnh trong cùng phiên.",
            ("Nhiều ảnh cùng lúc", "Tự tổng hợp CSV", "Theo dõi số viên detect"),
        )

        batch_files = st.file_uploader(
            "Tải nhiều ảnh",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key="batch_uploader_v2",
            label_visibility="collapsed",
        )

        if not batch_files:
            render_empty_upload_state_compact()
        else:
            st.caption(f"Đã nhận {len(batch_files)} ảnh.")
            results_list = []
            success_confidences = []
            total_detected_pills = 0
            model_info = load_models()

            progress_bar = st.progress(0)
            progress_text = st.empty()

            for idx, file in enumerate(batch_files):
                try:
                    image = Image.open(file)
                    image_array = normalize_uploaded_image(np.array(image))
                    prediction = predict_pill(image_array, model_info)
                    success_confidences.append(float(prediction["confidence"]))
                    total_detected_pills += int(prediction.get("num_detections", 1))

                    results_list.append({
                        "Tên ảnh": file.name,
                        "Kết quả": prediction["class_name"],
                        "Độ tin cậy": f"{prediction['confidence']:.2%}",
                        "Số viên": prediction.get("num_detections", 1),
                        "Luồng": "Detector + classifier" if prediction.get("mode") == "multi_pill_detection" else "Crop classifier",
                        "ID": prediction["class_id"],
                    })
                    progress_text.text(f"Đã phân tích {idx + 1}/{len(batch_files)} ảnh")
                except Exception as exc:
                    results_list.append({
                        "Tên ảnh": file.name,
                        "Kết quả": f"Lỗi: {str(exc)[:28]}",
                        "Độ tin cậy": "N/A",
                        "Số viên": 0,
                        "Luồng": "-",
                        "ID": "-",
                    })
                    progress_text.text(f"Đã xử lý {idx + 1}/{len(batch_files)} ảnh")

                progress_bar.progress((idx + 1) / len(batch_files))

            progress_text.empty()
            progress_bar.empty()

            results_df = pd.DataFrame(results_list)
            success_count = len([item for item in results_list if not str(item["Kết quả"]).startswith("Lỗi:")])
            avg_confidence = float(np.mean(success_confidences)) if success_confidences else 0.0

            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.metric("Tổng ảnh", len(batch_files))
            with stat_cols[1]:
                st.metric("Thành công", success_count)
            with stat_cols[2]:
                st.metric("Tổng viên detect", total_detected_pills)
            with stat_cols[3]:
                st.metric("Confidence TB", f"{avg_confidence:.1%}" if success_confidences else "--")

            st.markdown("### Bảng kết quả")
            st.dataframe(results_df, width="stretch", hide_index=True)

            st.download_button(
                label="Tải CSV",
                data=results_df.to_csv(index=False),
                file_name=f"phan_tich_hang_loat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    with tab3:
        render_model_performance_dashboard()
        return
        st.markdown(
            """
            <div class="section-intro">
                <div class="section-title">Mẹo nhanh</div>
                <p class="section-subtitle">Ít chữ hơn, xem nhanh hơn.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        help_cols = st.columns(3)
        help_cards = [
            ("Ảnh đẹp", "Sáng, nét, nền gọn."),
            ("Ảnh nhiều viên", "Để các viên cách nhau một chút."),
            ("Confidence thấp", "Xem top nhãn hoặc chụp lại."),
        ]
        for column, (title, body) in zip(help_cols, help_cards):
            with column:
                st.markdown(
                    f"""
                    <div class="tip-card">
                        <h4>{escape(title)}</h4>
                        <p>{escape(body)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.expander("Khi nào nên tin ngay", expanded=True):
            st.markdown(
                """
                - Confidence cao.
                - Preview detect đúng số viên.
                - Ảnh rõ, ít bóng đổ.
                """
            )

        with st.expander("Khi nào nên chụp lại"):
            st.markdown(
                """
                - Ảnh mờ hoặc thiếu sáng.
                - Viên bị che, sát nhau hoặc lệch khung.
                - Confidence thấp và top nhãn quá sát nhau.
                """
            )


if __name__ == "__main__":
    main_v2()
