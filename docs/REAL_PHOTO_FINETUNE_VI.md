# Fine-Tune Với Ảnh Thật Của Bạn

Để model nhận tốt hơn ảnh chụp thật của bạn, hãy dùng đúng format sau:

## 1. Đặt dữ liệu vào thư mục này

`data/user_real_photos/pill/image`

`data/user_real_photos/pill/label`

## 2. Tên file phải khớp nhau

Ví dụ:

- `data/user_real_photos/pill/image/real_001.jpg`
- `data/user_real_photos/pill/label/real_001.json`

## 3. Tạo nhãn nháp tự động nếu muốn

Nếu bạn chưa có file JSON, có thể để model hiện tại sinh nhãn nháp trước:

```bat
scripts\bootstrap_real_photo_labels.bat
```

Sau đó mở các file JSON trong `data/user_real_photos/pill/label` và sửa lại box/label cho đúng.

## 4. Format file nhãn

Mỗi file `.json` là một mảng các object:

```json
[
  { "x": 120, "y": 240, "w": 180, "h": 170, "label": 88 },
  { "x": 380, "y": 210, "w": 165, "h": 165, "label": 64 }
]
```

Ý nghĩa:

- `x`, `y`: góc trên bên trái của box
- `w`, `h`: chiều rộng và chiều cao box
- `label`: mã viên thuốc

## 5. Kiểm tra dữ liệu trước khi train

```bat
scripts\validate_real_photo_dataset.bat
```

## 6. Train detector với ảnh thật

Detector quan trọng nhất cho ảnh nhiều viên:

```bat
scripts\train_detector_real_adapt.bat
```

## 7. Train classifier với ảnh thật

```bat
scripts\train_classifier_real_adapt.bat
```

## 8. Gợi ý thực tế

- Ưu tiên chụp đúng kiểu ảnh bạn sẽ dùng trong app.
- Nên có cả ảnh dễ và ảnh khó: lệch sáng, bóng đổ, nền khác nhau, nhiều viên gần nhau.
- Với mỗi loại thuốc bạn muốn nhận tốt hơn, cố gắng có ít nhất 20-50 box thật.
- Nếu chưa đủ thời gian, hãy train detector trước rồi mới train classifier.
