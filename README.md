# VAIPE Pill Classification

> Bộ công cụ nhận diện viên thuốc dựa trên dataset VAIPE, gồm classifier cho ảnh crop từng viên,
> detector cho ảnh nhiều viên, giao diện demo, và knowledge-graph reranking để cải thiện kết quả.
>
> Tài liệu chi tiết bằng tiếng Việt: `README_VI.md`

## Chạy nhanh giao diện Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_streamlit_modern.py --server.port 8515

# Hoặc dùng script có sẵn
scripts\run_app_8515.bat
```

Mở tại:

```text
http://localhost:8515
```

Thiết lập môi trường và artifact:

- `docs/SETUP.md`
- `checkpoints/README.md`
- `data/README.md`

## Tổng quan

Dự án này đã được rút gọn về một bài toán duy nhất:

- **Đề tài:** Phân loại **một viên thuốc đơn lẻ** từ ảnh crop của dataset VAIPE
- **Đầu vào:** Ảnh crop của 1 viên thuốc
- **Đầu ra:** VAIPE `label_id` và xác suất dự đoán

Lý do chọn bài toán này:

- Dataset `public_train/pill` có bbox cho từng viên thuốc, phù hợp để cắt thành crop rồi train classifier.
- Yêu cầu của bạn là "đưa ảnh thuốc vào rồi đoán đúng hay chưa", nên classification trên crop là phù hợp hơn detection trên toa thuốc.
- Pipeline cũ trong project bị lệch bài toán: file train và file test không dùng chung một mục tiêu. Bản mới này đã được đồng nhất.

Hiện tại repo có 2 nhánh sử dụng rõ ràng:

- `single-pill classification`: phân loại 1 viên thuốc đã crop
- `multi-pill detection`: phát hiện nhiều viên thuốc trên ảnh pill gốc có bbox

## Paper và kiến trúc được chọn

- **Backbone chính:** `ResNet18`
- **Nhánh bổ sung:** `Multi-stream (Color)` lấy cảm hứng từ hướng `CG-IMIF`, fuse thêm histogram màu HSV vào đặc trưng ảnh
- **Paper nền tảng:** *Deep Residual Learning for Image Recognition* (He et al.)
- **Mốc tham khảo từ literature:** hướng `CG-IMIF` báo cáo khoảng `98.8-99%` khi khai thác thêm thông tin màu/đa kênh; đây là mục tiêu tham chiếu cho nhánh color stream, không phải cam kết kết quả mặc định của checkpoint hiện tại.
- **Lý do chọn:** ResNet18 vẫn nhẹ, ổn định, dễ hội tụ; nhánh color stream giúp bổ sung thông tin sắc thái/hình thái đặc biệt hữu ích với ảnh viên thuốc.

## Cấu trúc tối giản

- `train.py`: train model, early stopping, lưu `best_model.pth`, `final_model.pth`, `history.json`, `test_metrics.json`, `training_curves.png`, `confusion_matrix.png`
- `test.py`: đánh giá trên held-out test split hoặc dự đoán 1 ảnh thuốc, xuất thêm confusion matrix và `Grad-CAM`
- `detection_train.py`: train baseline multi-pill detector trên ảnh gốc + bbox
- `detection_test.py`: đánh giá detector hoặc infer 1 ảnh pill gốc và vẽ bbox
- `demo_infer.py`: wrapper 1 lệnh cho demo/app, mặc định dùng detector + classifier + selective KG tốt nhất hiện tại
- `web_demo.py`: local web app không cần framework ngoài, upload ảnh và trả bbox + label + JSON cho lớp giao diện
- `detection_utils.py`: utility dùng chung cho detection
- `knowledge_graph.py`: dùng knowledge graph để rerank nhãn sau detection bằng color + shape + imprint signature + prescription context
- `knowledge_graph_benchmark.py`: benchmark full held-out split để đo detector-only vs classifier vs classifier+KG
- `README.md`: tổng quan dự án
- `archive (1)/`: dataset VAIPE

## Hướng train

Lệnh mặc định:

```bash
python train.py
```

Nếu bạn muốn train ra thư mục mới nhưng vẫn tái sử dụng crop cache cũ:

```bash
python train.py --output-dir checkpoints/my_run --cache-dir checkpoints/crop_cache_160
```

Mặc định hiện tại:

- `epochs = 50`
- `patience = 6`
- `early stopping = on`
- `model_variant = cg_imif_color_fusion`
- `deterministic = on (mặc định)`
- `optimizer = AdamW`
- `scheduler = ReduceLROnPlateau`
- `image_size = 160`
- `color stream = HSV histogram fusion`
- `label smoothing = 0.05`
- `augmentation = flip + rotation + color jitter + random erasing`
- `anti-overfitting = augmentation + dropout + weight decay + early stopping`

Dữ liệu được xử lý như sau:

1. Đọc `public_train/pill/image` và `public_train/pill/label`
2. Cắt từng bbox thành một crop riêng
3. Tách `train / val / test` theo từng class
4. Train classifier trên crop
5. Đánh giá model tốt nhất trên held-out test split

## File đầu ra sau khi train

Tất cả nằm trong thư mục `checkpoints/`:

- `best_model.pth`: model tốt nhất theo `val_loss`
- `final_model.pth`: model ở epoch cuối cùng
- `history.json`: loss, accuracy, learning rate, thời gian train
- `test_metrics.json`: kết quả trên test split
- `training_curves.png`: biểu đồ loss/accuracy/gap/lr
- `confusion_matrix.png`: visualization confusion matrix trên held-out test split
- `split_manifest.json`: exact train/val/test split để test lại đúng cùng tập dữ liệu
- `dataset_summary.json`: thống kê crop và số lớp
- `crop_cache_160/`: cache crop viên thuốc để train/test nhanh hơn

## Multi-pill detection baseline

Baseline detection hiện tại dùng:

- detector: `Faster R-CNN`
- backbone mặc định: `fasterrcnn_mobilenet_v3_large_fpn`
- split theo `image` thay vì theo crop
- class-balanced image sampling + hard-example replay để giảm thiên lệch class và tập trung vào ảnh khó
- lưu `detection_split_manifest.json` để đánh giá lại đúng cùng tập test

Lệnh train mặc định:

```bash
python detection_train.py
```

Nếu bạn muốn fine-tune từ checkpoint detector có sẵn:

```bash
python detection_train.py --init-checkpoint checkpoints/detection/best_model.pth
```

Một số tham số hữu ích cho hướng cải thiện detection:

- `--sampler-power`: độ mạnh của class-balanced sampling, đặt `0` nếu muốn tắt
- `--hard-mining-topk`: tỷ lệ ảnh train loss cao sẽ được replay ở epoch sau
- `--hard-mining-boost`: hệ số tăng tần suất cho nhóm ảnh khó
- `--hard-mining-warmup`: số epoch warmup trước khi bắt đầu replay ảnh khó

Lệnh evaluate detector:

```bash
python detection_test.py --checkpoint checkpoints/detection/best_model.pth
```

Lệnh infer 1 ảnh pill gốc:

```bash
python detection_test.py --checkpoint checkpoints/detection/best_model.pth ^
                         --image "archive (1)/public_train/pill/image/VAIPE_P_0_0.jpg"
```

## Knowledge graph sau detection

Pipeline mới hỗ trợ refine nhãn sau detection theo hướng:

- detector tìm bbox viên thuốc
- classifier single-pill sinh `top-k` candidate cho từng bbox
- knowledge graph rerank candidate bằng:
  - `color prototype`
  - `shape prototype`
  - `imprint signature` từ texture/edge trên viên
  - `drug name` và `prescription co-occurrence` từ `pill_pres_map.json` + `prescription/label`
  - `top_confusions` từ checkpoint classifier để biết các cặp dễ nhầm
- mặc định dùng `selective reranking`:
  - giữ nguyên nhãn của detector nếu detector đã tự tin
  - chỉ cho KG override khi `detector score` thấp hơn ngưỡng
  - classifier phải rất chắc vào nhãn mới
  - nhãn gốc của detector phải có hỗ trợ rất thấp trong `top-k` của classifier

Lệnh demo:

```bash
python detection_test.py --checkpoint checkpoints/detection/best_model.pth ^
                         --image "archive (1)/public_train/pill/image/VAIPE_P_1011_0.jpg" ^
                         --classifier-checkpoint checkpoints/best_model.pth ^
                         --build-knowledge-graph ^
                         --kg-selective-override ^
                         --kg-max-detector-score 0.90 ^
                         --kg-min-candidate-probability 0.90 ^
                         --kg-max-anchor-probability 0.02
```

File đầu ra bổ sung:

- `single_image_detection_knowledge_graph.png`: bbox với nhãn sau khi rerank bằng graph
- `knowledge_graph_vaipe.json`: artifact graph đã build
- `single_image_detection.json`: chứa cả detector label, classifier top-k, graph-reranked candidates, `selected_source`, `override_applied`, `override_checks`

Lệnh demo/app-ready nhanh nhất:

```bash
python demo_infer.py --image "archive (1)/public_train/pill/image/VAIPE_P_1011_0.jpg"
```

Script này mặc định dùng:

- detector: `checkpoints/detection_mnv3_hardmining_ft_v2/best_model.pth`
- classifier: `checkpoints/best_model.pth`
- selective KG: `on`

Và sinh thêm:

- `app_response.json`: JSON gọn cho web app/mobile app layer, gồm `num_detections`, `num_overrides`, `top_labels`, `detections`, `artifacts`

## Web app local

Không cần cài thêm `streamlit` hay `fastapi`. Repo đã có sẵn một web app nhỏ dùng Python standard library:

```bash
python web_demo.py
```

Sau đó mở trình duyệt tại:

```text
http://127.0.0.1:8501
```

Web app hỗ trợ:

- upload ảnh pill gốc
- upload thêm `label json` nếu muốn so sánh với ground truth
- chạy detector + classifier + selective KG
- hiện preview detector, preview selective KG, bảng detections, và JSON response
- hiện `label_id | drug name` cho nhãn dự đoán và detector label khi map được từ prescription
- hiện `true label` cạnh `pred label` nếu có upload ground truth JSON
- tô màu xanh/đỏ để phân biệt `correct / wrong`
- hiện `override_checks` để biết vì sao selective KG có sửa hay không

File kết quả mỗi request được lưu trong:

- `checkpoints/demo_app_output/web_app/runs/<run_id>/`

Trong đó có:

- `single_image_detection.json`
- `single_image_detection.png`
- `single_image_detection_knowledge_graph.png`
- `single_image_ground_truth.png` nếu có upload json
- `app_response.json`

Để đo KG giúp bao nhiêu thật sự trên held-out split:

```bash
python knowledge_graph_benchmark.py --detector-checkpoint checkpoints/detection/best_model.pth ^
                                    --classifier-checkpoint checkpoints/best_model.pth ^
                                    --kg-selective-override ^
                                    --kg-max-detector-score 0.90 ^
                                    --kg-min-candidate-probability 0.90 ^
                                    --kg-max-anchor-probability 0.02
```

Script này giữ nguyên bbox từ detector và so sánh công bằng 3 chế độ:

- `detector_only`
- `detector_plus_classifier`
- `detector_plus_classifier_plus_kg`

File kết quả:

- `knowledge_graph_benchmark.json`
- `knowledge_graph_benchmark.png`

Config selective KG hiện tại:

- `kg_selective_override = true`
- `kg_max_detector_score = 0.90`
- `kg_min_candidate_probability = 0.90`
- `kg_max_anchor_probability = 0.02`

## Cách test

### 1. Đánh giá trên held-out test split

```bash
python test.py
```

Kết quả sẽ được lưu vào:

- `checkpoints/analysis_outputs/evaluation_metrics.json`
- `checkpoints/analysis_outputs/confusion_matrix.png`
- `checkpoints/analysis_outputs/low_support_accuracy.png`

### 2. Dự đoán 1 ảnh thuốc đã crop sẵn

```bash
python test.py --image path/to/pill_crop.jpg
```

### 3. Dự đoán từ ảnh pill gốc của VAIPE + file json bbox

```bash
python test.py --image "archive (1)/public_train/pill/image/VAIPE_P_0_0.jpg" ^
               --label-json "archive (1)/public_train/pill/label/VAIPE_P_0_0.json" ^
               --box-index 0
```

Nếu bạn biết nhãn đúng, có thể thêm:

```bash
python test.py --image path/to/pill_crop.jpg --true-label 64
```

Lúc đó `test.py` sẽ báo:

- label dự đoán
- xác suất %
- đúng hay sai nếu có `true_label`
- top-k dự đoán
- `single_image_gradcam.png`: heatmap giải thích model đang nhìn vào đâu trên ảnh input

Bạn có thể điều chỉnh:

- `--gradcam` / `--no-gradcam`: bật hoặc tắt visualization
- `--gradcam-topk`: số top prediction được vẽ heatmap

## Cách đọc overfitting

Mở `checkpoints/training_curves.png`:

- `Train loss` và `Val loss` cùng giảm: học ổn
- `Val loss` dừng giảm sớm và `Train loss` tiếp tục giảm: bắt đầu overfitting
- `Train acc - Val acc` gap quá lớn: model có dấu hiệu học thuộc

Model đã được setup để giảm overfitting bằng:

- dropout
- weight decay
- augmentation
- label smoothing
- early stopping patience = 6

## Ghi chú

- Đầu vào tốt nhất cho `test.py --image` là **ảnh crop một viên thuốc**, không phải nguyên toa thuốc.
- Nếu đưa ảnh gốc của VAIPE có nhiều viên thuốc, hãy dùng thêm `--label-json` và `--box-index` để cắt đúng viên cần test.
- `Grad-CAM` hiện tại giải thích **nhánh ảnh ResNet18**, không trực tiếp visual hóa nhánh histogram màu trong `cg_imif_color_fusion`.
- Knowledge graph hiện tại dùng `imprint signature` từ edge/texture của crop, không phải OCR trực tiếp trên mặt viên.
