# Thiết Lập Môi Trường Và Artifact

## 1. Môi trường Python

- Khuyến nghị: `Python 3.12`
- Hệ điều hành đã test gần đây: `Windows`

Tạo môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Chạy app hiện tại

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_streamlit_modern.py --server.port 8515
```

Mở tại:

```text
http://localhost:8515
```

Hoặc dùng script:

```powershell
scripts\run_app_8515.bat
```

## 3. Repo này KHÔNG chứa các file nặng

Các thư mục sau không được đẩy lên Git:

- `archive (1)/` - dataset VAIPE gốc
- `checkpoints/` - model weights, metrics, cache
- `data/` - dữ liệu cục bộ và ảnh thật bổ sung
- `.venv/` - môi trường Python của từng máy

Nếu clone repo về máy khác, bạn cần tự chuẩn bị lại các artifact này.

## 4. Artifact tối thiểu để app chạy inference nhiều viên

App hiện tại mong đợi tối thiểu:

- `checkpoints/detection_mnv3_hardmining_ft_v2/best_model.pth`
- `checkpoints/best_model.pth`

Artifact nên có thêm để dashboard đủ số liệu:

- `checkpoints/detection_mnv3_hardmining_ft_v2/test_metrics.json`
- `checkpoints/test_metrics.json`
- `checkpoints/dataset_summary.json`

Artifact knowledge graph:

- `checkpoints/knowledge_graph_vaipe.json`

Nếu thiếu knowledge graph, app vẫn có thể chạy detector + classifier, nhưng phần reranking sẽ yếu hơn hoặc bị tắt.

## 5. Dataset để train / fine-tune

### Dataset VAIPE gốc

```text
archive (1)/
└── public_train/
    └── pill/
        ├── image/
        └── label/
```

### Ảnh thật bổ sung của người dùng

```text
data/
└── user_real_photos/
    └── pill/
        ├── image/
        └── label/
```

## 6. Nếu clone về bị lỗi

Ưu tiên kiểm tra theo thứ tự:

1. Đã tạo `.venv` và cài `requirements.txt` chưa
2. Đã đặt đúng checkpoint detector/classifier chưa
3. Đã chạy đúng lệnh `.\.venv\Scripts\python.exe -m streamlit run ...` chưa
4. Nếu app từng fallback trước đó, hãy `Ctrl+C` và chạy lại hẳn để xóa cache cũ của Streamlit

## 7. Tài liệu liên quan

- `README.md` - tổng quan nhanh
- `README_VI.md` - tài liệu tiếng Việt chi tiết
- `checkpoints/README.md` - mô tả artifact model
- `data/README.md` - mô tả dữ liệu cục bộ / ảnh thật
