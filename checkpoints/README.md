# Checkpoints Và Artifact Model

Thư mục này không được đẩy toàn bộ lên Git vì chứa file nặng.

## App inference hiện tại cần gì

Tối thiểu để `app_streamlit_modern.py` chạy được detector nhiều viên:

- `detection_mnv3_hardmining_ft_v2/best_model.pth`
- `best_model.pth`

## Artifact nên có thêm

- `detection_mnv3_hardmining_ft_v2/test_metrics.json`
- `test_metrics.json`
- `dataset_summary.json`
- `knowledge_graph_vaipe.json`

## Gợi ý

- Nếu chỉ muốn chạy app nhanh: đặt đúng 2 file `best_model.pth`
- Nếu muốn dashboard đủ số liệu: thêm các file `test_metrics.json` và `dataset_summary.json`
- Nếu muốn reranking tốt hơn: thêm `knowledge_graph_vaipe.json`
