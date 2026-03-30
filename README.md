# VAIPE Pill Classification

> Bộ công cụ nhận diện viên thuốc dựa trên dataset VAIPE, gồm classifier cho ảnh crop từng viên,
> detector cho ảnh nhiều viên, giao diện demo, và knowledge-graph reranking để cải thiện kết quả.
>
> Tài liệu chi tiết bằng tiếng Việt: `README_VI.md`

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

## Cau truc toi gian

- `train.py`: train model, early stopping, luu `best_model.pth`, `final_model.pth`, `history.json`, `test_metrics.json`, `training_curves.png`, `confusion_matrix.png`
- `test.py`: danh gia tren held-out test split hoac du doan 1 anh thuoc, xuat them visualization confusion matrix
- `test.py`: danh gia tren held-out test split hoac du doan 1 anh thuoc, xuat them confusion matrix va `Grad-CAM`
- `detection_train.py`: train baseline multi-pill detector tren anh goc + bbox
- `detection_test.py`: danh gia detector hoac infer 1 anh pill goc va ve bbox
- `demo_infer.py`: wrapper 1 lenh cho demo/app, mac dinh dung detector + classifier + selective KG tot nhat hien tai
- `web_demo.py`: local web app khong can framework ngoai, upload anh va tra bbox + label + JSON cho lop giao dien
- `detection_utils.py`: utility dung chung cho detection
- `knowledge_graph.py`: dung knowledge graph de rerank nhan sau detection bang color + shape + imprint signature + prescription context
- `knowledge_graph_benchmark.py`: benchmark full held-out split de do detector-only vs classifier vs classifier+KG
- `README.md`: tong quan du an
- `archive (1)/`: dataset VAIPE

## Huong train

Lenh mac dinh:

```bash
python train.py
```

Neu ban muon train ra thu muc moi nhung van tai su dung crop cache cu:

```bash
python train.py --output-dir checkpoints/my_run --cache-dir checkpoints/crop_cache_160
```

Mac dinh hien tai:

- `epochs = 50`
- `patience = 6`
- `early stopping = on`
- `model_variant = cg_imif_color_fusion`
- `deterministic = on (mac dinh)`
- `optimizer = AdamW`
- `scheduler = ReduceLROnPlateau`
- `image_size = 160`
- `color stream = HSV histogram fusion`
- `label smoothing = 0.05`
- `augmentation = flip + rotation + color jitter + random erasing`
- `anti-overfitting = augmentation + dropout + weight decay + early stopping`

Du lieu duoc xu ly nhu sau:

1. Doc `public_train/pill/image` va `public_train/pill/label`
2. Cat tung bbox thanh mot crop rieng
3. Tach `train / val / test` theo tung class
4. Train classifier tren crop
5. Danh gia model tot nhat tren held-out test split

## File dau ra sau khi train

Tat ca nam trong thu muc `checkpoints/`:

- `best_model.pth`: model tot nhat theo `val_loss`
- `final_model.pth`: model o epoch cuoi cung
- `history.json`: loss, accuracy, learning rate, thoi gian train
- `test_metrics.json`: ket qua tren test split
- `training_curves.png`: bieu do loss/accuracy/gap/lr
- `confusion_matrix.png`: visualization confusion matrix tren held-out test split
- `split_manifest.json`: exact train/val/test split de test lai dung cung tap du lieu
- `dataset_summary.json`: thong ke crop va so lop
- `crop_cache_160/`: cache crop vien thuoc de train/test nhanh hon

## Multi-pill detection baseline

Baseline detection hien tai dung:

- detector: `Faster R-CNN`
- backbone mac dinh: `fasterrcnn_mobilenet_v3_large_fpn`
- split theo `image` thay vi theo crop
- class-balanced image sampling + hard-example replay de giam thien lech class va tap trung vao anh kho
- luu `detection_split_manifest.json` de evaluate lai dung cung tap test

Lenh train mac dinh:

```bash
python detection_train.py
```

Neu ban muon fine-tune tu checkpoint detector co san:

```bash
python detection_train.py --init-checkpoint checkpoints/detection/best_model.pth
```

Mot so tham so huu ich cho huong cai thien detection:

- `--sampler-power`: do manh cua class-balanced sampling, dat `0` neu muon tat
- `--hard-mining-topk`: ty le anh train loss cao se duoc replay o epoch sau
- `--hard-mining-boost`: he so tang tan suat cho nhom anh kho
- `--hard-mining-warmup`: so epoch warmup truoc khi bat dau replay anh kho

Lenh evaluate detector:

```bash
python detection_test.py --checkpoint checkpoints/detection/best_model.pth
```

Lenh infer 1 anh pill goc:

```bash
python detection_test.py --checkpoint checkpoints/detection/best_model.pth ^
                         --image "archive (1)/public_train/pill/image/VAIPE_P_0_0.jpg"
```

## Knowledge graph sau detection

Pipeline moi ho tro refine nhan sau detection theo huong:

- detector tim bbox vien thuoc
- classifier single-pill sinh `top-k` candidate cho tung bbox
- knowledge graph rerank candidate bang:
  - `color prototype`
  - `shape prototype`
  - `imprint signature` tu texture/edge tren vien
  - `drug name` va `prescription co-occurrence` tu `pill_pres_map.json` + `prescription/label`
  - `top_confusions` tu checkpoint classifier de biet cac cap de nham
- mac dinh dung `selective reranking`:
  - giu nguyen nhan cua detector neu detector da tu tin
  - chi cho KG override khi `detector score` thap hon nguong
  - classifier phai rat chac vao nhan moi
  - nhan goc cua detector phai co ho tro rat thap trong `top-k` cua classifier

Lenh demo:

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

File dau ra bo sung:

- `single_image_detection_knowledge_graph.png`: bbox voi nhan sau khi rerank bang graph
- `knowledge_graph_vaipe.json`: artifact graph da build
- `single_image_detection.json`: chua ca detector label, classifier top-k, graph-reranked candidates, `selected_source`, `override_applied`, `override_checks`

Lenh demo/app-ready nhanh nhat:

```bash
python demo_infer.py --image "archive (1)/public_train/pill/image/VAIPE_P_1011_0.jpg"
```

Script nay mac dinh dung:

- detector: `checkpoints/detection_mnv3_hardmining_ft_lr5e5_e3/best_model.pth`
- classifier: `checkpoints/retrain_cgimif_s42_det8/best_model.pth`
- selective KG: `on`

Va sinh them:

- `app_response.json`: JSON gon cho web app/mobile app layer, gom `num_detections`, `num_overrides`, `top_labels`, `detections`, `artifacts`

## Web app local

Khong can cai them `streamlit` hay `fastapi`. Repo da co san mot web app nho dung Python standard library:

```bash
python web_demo.py
```

Sau do mo trinh duyet tai:

```text
http://127.0.0.1:8501
```

Web app ho tro:

- upload anh pill goc
- upload them `label json` neu muon so sanh voi ground truth
- chay detector + classifier + selective KG
- hien preview detector, preview selective KG, bang detections, va JSON response
- hien `label_id | drug name` cho nhan du doan va detector label khi map duoc tu prescription
- hien `true label` canh `pred label` neu co upload ground truth JSON
- to mau xanh/do de phan biet `correct / wrong`
- hien `override_checks` de biet vi sao selective KG co sua hay khong

File ket qua moi request duoc luu trong:

- `checkpoints/demo_app_output/web_app/runs/<run_id>/`

Trong do co:

- `single_image_detection.json`
- `single_image_detection.png`
- `single_image_detection_knowledge_graph.png`
- `single_image_ground_truth.png` neu co upload json
- `app_response.json`

De do KG giup bao nhieu that su tren held-out split:

```bash
python knowledge_graph_benchmark.py --detector-checkpoint checkpoints/detection/best_model.pth ^
                                    --classifier-checkpoint checkpoints/best_model.pth ^
                                    --kg-selective-override ^
                                    --kg-max-detector-score 0.90 ^
                                    --kg-min-candidate-probability 0.90 ^
                                    --kg-max-anchor-probability 0.02
```

Script nay giu nguyen bbox tu detector va so sanh cong bang 3 che do:

- `detector_only`
- `detector_plus_classifier`
- `detector_plus_classifier_plus_kg`

File ket qua:

- `knowledge_graph_benchmark.json`
- `knowledge_graph_benchmark.png`

Config selective KG hien tai:

- `kg_selective_override = true`
- `kg_max_detector_score = 0.90`
- `kg_min_candidate_probability = 0.90`
- `kg_max_anchor_probability = 0.02`

## Cach test

### 1. Danh gia tren held-out test split

```bash
python test.py
```

Ket qua se duoc luu vao:

- `checkpoints/analysis_outputs/evaluation_metrics.json`
- `checkpoints/analysis_outputs/confusion_matrix.png`
- `checkpoints/analysis_outputs/low_support_accuracy.png`

### 2. Du doan 1 anh thuoc da crop san

```bash
python test.py --image path/to/pill_crop.jpg
```

### 3. Du doan tu anh pill goc cua VAIPE + file json bbox

```bash
python test.py --image "archive (1)/public_train/pill/image/VAIPE_P_0_0.jpg" ^
               --label-json "archive (1)/public_train/pill/label/VAIPE_P_0_0.json" ^
               --box-index 0
```

Neu ban biet nhan dung, co the them:

```bash
python test.py --image path/to/pill_crop.jpg --true-label 64
```

Luc do `test.py` se bao:

- label du doan
- xac suat %
- dung hay sai neu co `true_label`
- top-k du doan
- `single_image_gradcam.png`: heatmap giai thich model dang nhin vao dau tren anh input

Ban co the dieu chinh:

- `--gradcam` / `--no-gradcam`: bat hoac tat visualization
- `--gradcam-topk`: so top prediction duoc ve heatmap

## Cach doc overfitting

Mo `checkpoints/training_curves.png`:

- `Train loss` va `Val loss` cung giam: hoc on
- `Val loss` dung giam som va `Train loss` tiep tuc giam: bat dau overfitting
- `Train acc - Val acc` gap qua lon: model co dau hieu hoc thuoc

Model da duoc setup de giam overfitting bang:

- dropout
- weight decay
- augmentation
- label smoothing
- early stopping patience = 6

## Ghi chu

- Dau vao tot nhat cho `test.py --image` la **anh crop mot vien thuoc**, khong phai nguyen toa thuoc.
- Neu dua anh goc cua VAIPE co nhieu vien thuoc, hay dung them `--label-json` va `--box-index` de cat dung vien can test.
- `Grad-CAM` hien tai giai thich **nhanh anh ResNet18**, khong truc tiep visual hoa nhanh histogram mau trong `cg_imif_color_fusion`.
- Knowledge graph hien tai dung `imprint signature` tu edge/texture cua crop, khong phai OCR truc tiep tren mat vien.
