from __future__ import annotations

import argparse
import base64
import cgi
import html
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from uuid import uuid4
import sys
from wsgiref.simple_server import make_server

import torch
from torchvision.ops import box_iou

from demo_infer import (
    DEFAULT_DEMO_CLASSIFIER_CHECKPOINT,
    DEFAULT_DEMO_DETECTOR_CHECKPOINT,
    DEFAULT_DEMO_OUTPUT_DIR,
    build_app_response,
)
from detection_test import load_ground_truth, predict_single_image
from detection_utils import DEFAULT_DETECTION_SCORE_THRESHOLD, load_detection_checkpoint, save_json
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
    load_classifier_checkpoint,
    load_or_build_knowledge_graph,
)


DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8501
DEFAULT_WEB_OUTPUT_DIR = DEFAULT_DEMO_OUTPUT_DIR / "web_app"
DEFAULT_WEB_IOU_THRESHOLD = 0.50


@dataclass(frozen=True)
class WebDemoConfig:
    host: str
    port: int
    output_dir: Path
    detector_checkpoint: Path
    classifier_checkpoint: Path
    knowledge_graph_artifact: Path
    knowledge_graph_cache_dir: Path
    build_knowledge_graph: bool
    score_threshold: float
    kg_top_k: int
    kg_visual_weight: float
    kg_context_weight: float
    kg_anchor_weight: float
    kg_selective_override: bool
    kg_max_detector_score: float
    kg_min_candidate_probability: float
    kg_max_anchor_probability: float


@lru_cache(maxsize=2)
def load_demo_bundle(
    detector_checkpoint: str,
    classifier_checkpoint: str,
    knowledge_graph_artifact: str,
    knowledge_graph_cache_dir: str,
    build_knowledge_graph: bool,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector_model, detector_payload, _, detector_index_to_label = load_detection_checkpoint(
        Path(detector_checkpoint),
        device,
    )
    classifier_model, classifier_payload, _, classifier_idx_to_class = load_classifier_checkpoint(
        Path(classifier_checkpoint),
        device,
    )

    confusion_metrics_path = Path(classifier_checkpoint).parent / "test_metrics.json"
    if not confusion_metrics_path.exists():
        confusion_metrics_path = None
    knowledge_graph = load_or_build_knowledge_graph(
        artifact_path=Path(knowledge_graph_artifact),
        cache_dir=Path(knowledge_graph_cache_dir),
        image_size=int(classifier_payload["image_size"]),
        color_bins=int(classifier_payload.get("color_bins", 8)),
        confusion_metrics_path=confusion_metrics_path,
        rebuild=bool(build_knowledge_graph),
    )
    return {
        "device": device,
        "detector_model": detector_model,
        "detector_checkpoint": detector_payload,
        "detector_index_to_label": detector_index_to_label,
        "classifier_model": classifier_model,
        "classifier_checkpoint": classifier_payload,
        "classifier_idx_to_class": classifier_idx_to_class,
        "knowledge_graph": knowledge_graph,
        "label_display_names": build_label_display_names(),
    }


def _store_upload(field: cgi.FieldStorage, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(str(field.filename or "")).suffix or ".jpg"
    target_path = target_dir / f"{uuid4().hex}{suffix}"
    with open(target_path, "wb") as handle:
        handle.write(field.file.read())
    return target_path


def _field_or_none(form: cgi.FieldStorage, field_name: str) -> cgi.FieldStorage | None:
    if field_name not in form:
        return None
    field = form[field_name]
    if isinstance(field, list):
        field = field[0]
    if not getattr(field, "filename", None):
        return None
    return field


def _image_to_data_uri(path: str | None) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    suffix = file_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _match_predictions_to_ground_truth(
    detections: Sequence[Dict[str, object]],
    gt_boxes: Sequence[Sequence[float]],
    gt_labels: Sequence[int],
    iou_threshold: float = DEFAULT_WEB_IOU_THRESHOLD,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    if not detections:
        return [], {
            "ground_truth_boxes": int(len(gt_boxes)),
            "matched_predictions": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "unmatched_predictions": 0,
        }

    pred_boxes = torch.as_tensor([list(item["box_xyxy"]) for item in detections], dtype=torch.float32).reshape(-1, 4)
    pred_scores = [float(item["score"]) for item in detections]
    gt_boxes_tensor = torch.as_tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4)

    matched_gt: set[int] = set()
    annotated: List[Dict[str, object]] = [dict(item) for item in detections]
    order = sorted(range(len(detections)), key=lambda index: pred_scores[index], reverse=True)

    for pred_index in order:
        best_gt_index = -1
        best_iou = 0.0
        if gt_boxes_tensor.numel() > 0:
            ious = box_iou(pred_boxes[pred_index].unsqueeze(0), gt_boxes_tensor)[0]
            for gt_index, iou_value in enumerate(ious.tolist()):
                if gt_index in matched_gt:
                    continue
                if float(iou_value) > best_iou:
                    best_iou = float(iou_value)
                    best_gt_index = int(gt_index)

        row = annotated[pred_index]
        row["ground_truth_matched"] = bool(best_gt_index >= 0 and best_iou >= float(iou_threshold))
        row["ground_truth_iou"] = float(best_iou)
        row["ground_truth_label_id"] = int(gt_labels[best_gt_index]) if row["ground_truth_matched"] else None
        row["is_correct"] = bool(
            row["ground_truth_matched"] and int(row["label_id"]) == int(gt_labels[best_gt_index])
        )
        if row["ground_truth_matched"]:
            matched_gt.add(best_gt_index)

    matched_predictions = sum(1 for item in annotated if bool(item["ground_truth_matched"]))
    correct_predictions = sum(1 for item in annotated if bool(item["is_correct"]))
    incorrect_predictions = sum(
        1 for item in annotated if bool(item["ground_truth_matched"]) and not bool(item["is_correct"])
    )
    unmatched_predictions = sum(1 for item in annotated if not bool(item["ground_truth_matched"]))
    return annotated, {
        "ground_truth_boxes": int(len(gt_boxes)),
        "matched_predictions": int(matched_predictions),
        "correct_predictions": int(correct_predictions),
        "incorrect_predictions": int(incorrect_predictions),
        "unmatched_predictions": int(unmatched_predictions),
    }


def run_web_inference(
    config: WebDemoConfig,
    *,
    image_path: Path,
    label_json: Path | None = None,
) -> Dict[str, object]:
    bundle = load_demo_bundle(
        detector_checkpoint=str(config.detector_checkpoint),
        classifier_checkpoint=str(config.classifier_checkpoint),
        knowledge_graph_artifact=str(config.knowledge_graph_artifact),
        knowledge_graph_cache_dir=str(config.knowledge_graph_cache_dir),
        build_knowledge_graph=bool(config.build_knowledge_graph),
    )

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = config.output_dir / "runs" / run_id
    payload = predict_single_image(
        model=bundle["detector_model"],
        checkpoint=bundle["detector_checkpoint"],
        index_to_label=bundle["detector_index_to_label"],
        device=bundle["device"],
        image_path=image_path,
        output_dir=run_dir,
        score_threshold=config.score_threshold,
        label_json=label_json,
        classifier_model=bundle["classifier_model"],
        classifier_checkpoint=bundle["classifier_checkpoint"],
        classifier_idx_to_class=bundle["classifier_idx_to_class"],
        knowledge_graph=bundle["knowledge_graph"],
        kg_top_k=config.kg_top_k,
        kg_visual_weight=config.kg_visual_weight,
        kg_context_weight=config.kg_context_weight,
        kg_anchor_weight=config.kg_anchor_weight,
        kg_selective_override=config.kg_selective_override,
        kg_max_detector_score=config.kg_max_detector_score,
        kg_min_candidate_probability=config.kg_min_candidate_probability,
        kg_max_anchor_probability=config.kg_max_anchor_probability,
    )

    label_display_names = dict(bundle["label_display_names"])
    app_response = build_app_response(payload, label_display_names=label_display_names)
    if label_json is not None and label_json.exists():
        gt_boxes, gt_labels = load_ground_truth(image_path=image_path, label_json=label_json)
        annotated_detections, ground_truth_summary = _match_predictions_to_ground_truth(
            list(app_response.get("detections", [])),
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
        )
        for item in annotated_detections:
            true_label = item.get("ground_truth_label_id")
            if true_label is None:
                item["ground_truth_label_name"] = ""
                item["ground_truth_display_label"] = "-"
            else:
                label_name = str(label_display_names.get(int(true_label), "")).strip()
                item["ground_truth_label_name"] = label_name
                item["ground_truth_display_label"] = (
                    f"{int(true_label)} | {label_name}" if label_name else str(int(true_label))
                )
        app_response["detections"] = annotated_detections
        app_response["ground_truth_summary"] = ground_truth_summary

    app_response_path = run_dir / "app_response.json"
    app_response.setdefault("artifacts", {})["app_response_json"] = str(app_response_path)
    save_json(app_response_path, app_response)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "payload": payload,
        "app_response": app_response,
    }


def _render_detection_rows(detections: List[Dict[str, object]]) -> str:
    if not detections:
        return "<tr><td colspan='7'>Không tìm thấy thẻ thuốc nào.</td></tr>"

    rows: List[str] = []
    for item in detections:
        status_class = "row-neutral"
        status_label = "Chưa có GT"
        if item.get("ground_truth_matched") is True:
            if item.get("is_correct") is True:
                status_class = "row-correct"
                status_label = "Đúng"
            else:
                status_class = "row-wrong"
                status_label = "Sai"

        rows.append(
            f"<tr class='{status_class}'>"
            f"<td><span style='font-weight:700; color:var(--primary)'>{html.escape(str(item.get('display_label', item['label_id'])))}</span></td>"
            f"<td>{html.escape(str(item.get('ground_truth_display_label', '-')))}</td>"
            f"<td><span class='badge' style='background: { 'var(--success)' if item.get('is_correct') else 'var(--danger)' if item.get('ground_truth_matched') else 'var(--secondary)'}; color:white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;'>{status_label}</span></td>"
            f"<td><strong>{float(item['score']):.2%}</strong></td>"
            f"<td><small>{html.escape(str(item['source']))}</small></td>"
            f"<td>{html.escape(str(item.get('detector_display_label', item['detector_label_id'])))}</td>"
            f"<td>{'Có' if bool(item['override_applied']) else 'Không'}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_image_card(title: str, path: str | None) -> str:
    data_uri = _image_to_data_uri(path)
    if data_uri is None:
        return ""
    return (
        "<div class='image-card'>"
        f"<h3>{html.escape(title)}</h3>"
        f"<img src='{data_uri}' alt='{html.escape(title)}' />"
        f"<p>{html.escape(str(path))}</p>"
        "</div>"
    )


def render_page(
    config: WebDemoConfig,
    *,
    message: str | None = None,
    result: Dict[str, object] | None = None,
) -> str:
    result_section = ""
    if result is not None:
        run_identifier = str(result.get("run_id") or result.get("run_dir") or "N/A")
        app_response = result["app_response"]
        payload = result["payload"]
        artifacts = dict(app_response.get("artifacts", {}))
        detection_table = _render_detection_rows(list(app_response.get("detections", [])))
        gt_summary = dict(app_response.get("ground_truth_summary", {}))
        gt_stats_html = ""
        if gt_summary:
            gt_stats_html = (
                f"<div class='stat correct'><span class='label'>Đúng</span><span class='value'>{int(gt_summary.get('correct_predictions', 0))}</span></div>"
                f"<div class='stat wrong'><span class='label'>Sai</span><span class='value'>{int(gt_summary.get('incorrect_predictions', 0))}</span></div>"
                f"<div class='stat neutral'><span class='label'>Bỏ sót</span><span class='value'>{int(gt_summary.get('unmatched_predictions', 0))}</span></div>"
                f"<div class='stat neutral'><span class='label'>Tổng GT</span><span class='value'>{int(gt_summary.get('ground_truth_boxes', 0))}</span></div>"
            )
        result_section = f"""
        <section class="panel result-panel">
          <div class="panel-header">
            <h2 class="section-title"><i class="fas fa-poll"></i> Kết quả phân tích</h2>
            <div class="run-id-badge">ID: {html.escape(run_identifier)}</div>
          </div>
          <div class="stats-grid">
            <div class="stat"><span class="label">Số lượng thuốc</span><span class="value">{int(app_response['num_detections'])}</span></div>
            <div class="stat"><span class="label">Sửa lỗi KG</span><span class="value">{int(app_response['num_overrides'])}</span></div>
            <div class="stat" style="grid-column: span 2;"><span class="label">Dự đoán chính</span><span class="value">{html.escape(', '.join(str(label) for label in app_response.get('top_label_displays', app_response['top_labels'])))}</span></div>
            {gt_stats_html}
          </div>

          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>Nhãn dự đoán</th>
                  <th>Nhãn thực tế</th>
                  <th>Trạng thái</th>
                  <th>Độ tin cậy</th>
                  <th>Nguồn</th>
                  <th>Nhãn Detector</th>
                  <th>KG Sửa?</th>
                </tr>
              </thead>
              <tbody>{detection_table}</tbody>
            </table>
          </div>

          <div class="image-grid">
            {_render_image_card("Ảnh Detector", artifacts.get("detector_preview"))}
            {_render_image_card("Ảnh Selective KG", artifacts.get("knowledge_graph_preview"))}
            {_render_image_card("Ảnh Ground Truth", artifacts.get("ground_truth_preview"))}
          </div>

          <details class="json-details">
            <summary>Xem chi tiết JSON hệ thống <i class="fas fa-chevron-down"></i></summary>
            <div class="json-grid">
              <div class="json-box">
                <h3>App Response JSON</h3>
                <pre>{html.escape(json.dumps(app_response, ensure_ascii=False, indent=2))}</pre>
              </div>
              <div class="json-box">
                <h3>Raw Payload JSON</h3>
                <pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>
              </div>
            </div>
          </details>
        </section>
        """

    message_html = f"<div class='message animate-in'>{html.escape(message)}</div>" if message else ""
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VAIPE Multi-pill Analyzer</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {{
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --secondary: #64748b;
      --success: #16a34a;
      --danger: #dc2626;
      --warning: #d97706;
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #1e293b;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --radius: 12px;
      --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
      --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    }}

    * {{ box-sizing: border-box; transition: all 0.2s ease; }}

    body {{
      margin: 0;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}

    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 40px 20px;
    }}

    header {{
      text-align: center;
      margin-bottom: 40px;
    }}

    .logo-area {{
      display: inline-flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }}

    .logo-icon {{
      background: var(--primary);
      color: white;
      width: 48px;
      height: 48px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 24px;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }}

    h1 {{
      margin: 0;
      font-size: 2.25rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      background: linear-gradient(135deg, var(--primary), #4f46e5);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .tagline {{
      color: var(--text-muted);
      font-size: 1.1rem;
      max-width: 600px;
      margin: 12px auto 0;
    }}

    .panel {{
      background: var(--card);
      border-radius: var(--radius);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      padding: 32px;
      margin-bottom: 24px;
    }}

    .section-title {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 0 0 24px;
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text);
    }}

    .upload-form {{
      display: grid;
      gap: 24px;
    }}

    .drop-zone {{
      border: 2px dashed var(--border);
      border-radius: var(--radius);
      padding: 40px;
      text-align: center;
      cursor: pointer;
      background: #fdfdfd;
    }}

    .drop-zone:hover {{
      border-color: var(--primary);
      background: #f0f7ff;
    }}

    .drop-zone input {{ display: none; }}

    .drop-zone i {{
      font-size: 2.5rem;
      color: var(--primary);
      margin-bottom: 12px;
    }}

    .drop-zone p {{
      margin: 0;
      font-weight: 600;
      color: var(--text);
    }}

    .drop-zone span {{
      font-size: 0.875rem;
      color: var(--text-muted);
    }}

    .btn {{
      appearance: none;
      background: var(--primary);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 12px 24px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      width: 100%;
    }}

    .btn:hover {{
      background: var(--primary-dark);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }}

    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }}

    .stat {{
      background: #f8fafc;
      padding: 20px;
      border-radius: 12px;
      border: 1px solid var(--border);
    }}

    .stat .label {{
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 4px;
      display: block;
    }}

    .stat .value {{
      font-size: 1.25rem;
      font-weight: 800;
      color: var(--text);
    }}

    .stat.correct {{ border-left: 4px solid var(--success); }}
    .stat.wrong {{ border-left: 4px solid var(--danger); }}

    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 20px;
      margin-top: 32px;
    }}

    .image-card {{
      background: white;
      border-radius: var(--radius);
      overflow: hidden;
      border: 1px solid var(--border);
    }}

    .image-card h3 {{
      font-size: 0.875rem;
      padding: 12px 16px;
      margin: 0;
      background: #f8fafc;
      border-bottom: 1px solid var(--border);
    }}

    .image-card img {{
      width: 100%;
      height: auto;
      display: block;
    }}

    .table-container {{
      overflow-x: auto;
      border-radius: 8px;
      border: 1px solid var(--border);
      margin-bottom: 32px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }}

    th {{
      background: #f8fafc;
      text-align: left;
      padding: 12px 16px;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
    }}

    td {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
    }}

    tr:last-child td {{ border-bottom: none; }}

    .row-correct {{ background-color: rgba(22, 163, 74, 0.05); }}
    .row-wrong {{ background-color: rgba(220, 38, 38, 0.05); }}

    .json-details {{
      margin-top: 24px;
    }}

    .json-details summary {{
      cursor: pointer;
      font-weight: 600;
      color: var(--primary);
      padding: 8px 0;
    }}

    pre {{
      background: #1e293b;
      color: #e2e8f0;
      padding: 20px;
      border-radius: 8px;
      font-size: 12px;
      max-height: 400px;
      overflow: auto;
    }}

    .message {{
      padding: 12px 16px;
      border-radius: 8px;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      color: #1d4ed8;
      margin-bottom: 24px;
    }}

    .config-info {{
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 40px;
      text-align: center;
      padding: 20px;
      border-top: 1px solid var(--border);
    }}

    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .animate-in {{ animation: fadeIn 0.3s ease-out forwards; }}

    @media (max-width: 640px) {{
      .stats-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container animate-in">
    <header>
      <div class="logo-area">
        <div class="logo-icon"><i class="fas fa-capsules"></i></div>
        <h1>VAIPE Analyzer</h1>
      </div>
      <p class="tagline">Hệ thống nhận diện và kiểm chứng thuốc đa loại dựa trên biểu đồ tri thức (KG) và Multi-stream Fusion</p>
    </header>

    <section class="panel">
      <h2 class="section-title"><i class="fas fa-upload"></i> Tải ảnh lên</h2>
      {message_html}
      <form class="upload-form" method="post" enctype="multipart/form-data">
        <div class="drop-zone" onclick="document.getElementById('image').click()">
          <i class="fas fa-cloud-upload-alt"></i>
          <p>Nhấp vào đây để chọn ảnh thuốc</p>
          <span>Hỗ trợ JPG, PNG, WEBP</span>
          <input id="image" name="image" type="file" accept=".jpg,.jpeg,.png,.webp" required onchange="this.parentElement.querySelector('p').innerText = this.files[0].name" />
        </div>
        
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.875rem;">
           <i class="fas fa-file-json" style="color: var(--warning)"></i>
           <label for="label_json" style="font-weight:600">Nhãn Ground Truth (tùy chọn):</label>
           <input id="label_json" name="label_json" type="file" accept=".json,application/json" style="font-size: 0.8rem;" />
        </div>

        <button type="submit" class="btn">
          Phân tích ảnh <i class="fas fa-arrow-right"></i>
        </button>
      </form>
    </section>

    {result_section}

    <footer class="config-info">
      <div>Detector: {html.escape(Path(config.detector_checkpoint).name)} | Classifier: {html.escape(Path(config.classifier_checkpoint).name)}</div>
      <div>KG Selective: <span style="color: var(--primary)">{'BẬT' if config.kg_selective_override else 'TẮT'}</span> | Conf: {config.score_threshold} | KG Weight: {config.kg_visual_weight}</div>
    </footer>
  </div>
</body>
</html>
"""


def app(environ, start_response):
    config = getattr(app, "config")
    message = None
    result = None

    try:
        if environ["REQUEST_METHOD"] == "POST":
            # Xử lý form với độ an toàn cao hơn
            form = cgi.FieldStorage(fp=environ["wsgi.input"], environ=environ, keep_blank_values=True)
            
            # Cần kiểm tra field bằng cách khác vì cgi.FieldStorage không cho phép ép kiểu bool trực tiếp
            image_field = _field_or_none(form, "image")
            label_field = _field_or_none(form, "label_json")

            if image_field is not None:
                image_path = _store_upload(image_field, config.output_dir / "uploads")
                label_json = None
                if label_field is not None:
                    label_json = _store_upload(label_field, config.output_dir / "uploads")

                result = run_web_inference(config, image_path=image_path, label_json=label_json)
                message = "Phân tích hoàn tất!"
            else:
                message = "Lỗi: Không tìm thấy ảnh trong dữ liệu gửi lên."
                
        content = render_page(config, message=message, result=result)
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [content.encode("utf-8")]

    except Exception as e:
        import traceback
        err_msg = f"SERVER ERROR: {str(e)}\n\n{traceback.format_exc()}"
        print(err_msg) # Log ra terminal để debug
        start_response("500 Internal Server Error", [("Content-Type", "text/plain; charset=utf-8")])
        return [f"Đã xảy ra lỗi trên hệ thống:\n\n{err_msg}".encode("utf-8")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_WEB_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WEB_OUTPUT_DIR)
    parser.add_argument("--detector-checkpoint", type=Path, default=DEFAULT_DEMO_DETECTOR_CHECKPOINT)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_DEMO_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--knowledge-graph", type=Path, default=DEFAULT_KNOWLEDGE_GRAPH_PATH)
    parser.add_argument("--kg-cache-dir", type=Path, default=DEFAULT_KNOWLEDGE_GRAPH_CACHE_DIR)
    parser.add_argument("--build-kg", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_DETECTION_SCORE_THRESHOLD)
    parser.add_argument("--kg-top-k", type=int, default=DEFAULT_KG_CANDIDATES)
    parser.add_argument("--kg-visual-weight", type=float, default=DEFAULT_KG_VISUAL_WEIGHT)
    parser.add_argument("--kg-context-weight", type=float, default=DEFAULT_KG_CONTEXT_WEIGHT)
    parser.add_argument("--kg-anchor-weight", type=float, default=DEFAULT_KG_ANCHOR_WEIGHT)
    parser.add_argument("--kg-selective-override", action="store_true", default=DEFAULT_KG_SELECTIVE_OVERRIDE)
    parser.add_argument("--kg-max-detector-score", type=float, default=DEFAULT_KG_MAX_DETECTOR_SCORE)
    parser.add_argument("--kg-min-candidate-prob", type=float, default=DEFAULT_KG_MIN_CANDIDATE_PROBABILITY)
    parser.add_argument("--kg-max-anchor-prob", type=float, default=DEFAULT_KG_MAX_ANCHOR_PROBABILITY)

    args = parser.parse_args()
    config = WebDemoConfig(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        detector_checkpoint=args.detector_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        knowledge_graph_artifact=args.knowledge_graph,
        knowledge_graph_cache_dir=args.kg_cache_dir,
        build_knowledge_graph=args.build_kg,
        score_threshold=args.score_threshold,
        kg_top_k=args.kg_top_k,
        kg_visual_weight=args.kg_visual_weight,
        kg_context_weight=args.kg_context_weight,
        kg_anchor_weight=args.kg_anchor_weight,
        kg_selective_override=args.kg_selective_override,
        kg_max_detector_score=args.kg_max_detector_score,
        kg_min_candidate_probability=args.kg_min_candidate_prob,
        kg_max_anchor_probability=args.kg_max_anchor_prob,
    )

    app.config = config
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"Khởi chạy web demo tại http://{config.host}:{config.port}")
    with make_server(config.host, config.port, app) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
