```md
# 🎓 Dự án: Dự đoán Khả năng Tốt nghiệp Đúng hạn của Sinh viên

## 🌟 Giới thiệu
Đây là một dự án **Phân loại Nhị phân (Binary Classification)** nhằm dự đoán khả năng một sinh viên sẽ tốt nghiệp đúng hạn (`1`) hay không (`0`), dựa trên các thông tin học tập và cá nhân tích lũy theo từng kỳ học.

**Ý tưởng chính:** Triển khai 4 mô hình độc lập cho 4 mốc thời gian (Sem 5, 6, 7, 8) — gọi chung là **Multi-Stage Prediction** — để có thể đưa cảnh báo sớm và hành động can thiệp kịp thời.

**Mô hình chính:** `RandomForestClassifier` (scikit-learn) với `class_weight='balanced'` để xử lý mất cân bằng lớp.

---

## 📌 Điểm nổi bật
- **Dự đoán đa giai đoạn:** Mỗi kỳ có bộ đặc trưng tích lũy riêng và mô hình riêng.
- **Tiền xử lý tự động:** `Pipeline` + `ColumnTransformer` kết hợp `OneHotEncoder` cho categorical và `SimpleImputer(strategy='mean')` cho numeric.
- **Triển khai thân thiện người dùng:** Ứng dụng Streamlit để nhập thông tin sinh viên và xem dự đoán cùng độ tin cậy.

---

## 🛠️ Cấu trúc Dự án
```
Predict_Student_Result/
├── data/
│   └── graduation_dataset_final1.csv   # Dữ liệu gốc
├── models/
│   ├── model_sem5.pkl                  # Mô hình Random Forest Sem 5
│   ├── model_sem6.pkl                  # Mô hình Random Forest Sem 6
│   ├── model_sem7.pkl                  # Mô hình Random Forest Sem 7
│   └── model_sem8.pkl                  # Mô hình Random Forest Sem 8
├── src/
│   ├── Config.py                       # Hằng số: đường dẫn, tên cột, FEATURE_SETS
│   ├── model_utils.py                  # Hàm load/save model, preprocess, predict
│   └── train.py                        # Script huấn luyện 4 mô hình
├── Streamlit_App.py                    # Giao diện web
├── .env                                # Biến môi trường (ví dụ: PORT)
├── requirements.txt
└── README.md
```

---

## ⚙️ Chi tiết Mô hình và Kỹ thuật

### 1. Dữ liệu & Đặc trưng
- **File dữ liệu:** `data/graduation_dataset_final1.csv`
- **Target:** `target` (1 = tốt nghiệp đúng hạn, 0 = trễ hạn/không tốt nghiệp)

- **Đặc trưng cố định (FIXED_FEATURES):**
  - `gender`, `major`, `admission_type`, `admission_score`

- **Đặc trưng theo kỳ (ví dụ):**
  - GPA: `gpa_sem1`, `gpa_sem2`, ..., `gpa_sem8`
  - Tín chỉ tích lũy: `credits_sem1`, ..., `credits_sem8`
  - Số môn trượt: `failed_sem1`, ..., `failed_sem8`
  - Cảnh báo học tập: `warn_sem1`, ..., `warn_sem8`

> Chi tiết bộ đặc trưng cho từng mô hình được định nghĩa trong `src/Config.py` (biến `FEATURE_SETS`).


### 2. Tiền xử lý & Pipeline
Mỗi mô hình (`sem5`..`sem8`) dùng cùng một pipeline gồm:

- **ColumnTransformer `preprocessor`**
  - `categorical_features` → `OneHotEncoder(handle_unknown='ignore', sparse=False)`
  - `numerical_features` → `SimpleImputer(strategy='mean')`

- **Pipeline:** `Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=SEED))])`

### 3. Huấn luyện
- Script `src/train.py` thực hiện:
  1. Tải dữ liệu `graduation_dataset_final1.csv`.
  2. Lặp qua `FEATURE_SETS` cho từng semester (sem5..sem8).
  3. Chuẩn hóa dữ liệu, tách train/test (ví dụ `train_test_split`), huấn luyện RandomForest.
  4. Đánh giá (accuracy, precision, recall, f1, AUC) và lưu model bằng `joblib.dump` vào `models/model_semX.pkl`.

---

## 🧩 Hàm Dự đoán (`predict_graduation`)
Mô tả chức năng của `src/model_utils.py::predict_graduation(data: dict, semester_point: int)`:

1. **Chọn mô hình** theo `semester_point` (5/6/7/8) — tải file pkl tương ứng.
2. **Lấy danh sách `required_features`** từ `Config.FEATURE_SETS[semester_point]`.
3. **Tạo DataFrame** 1 hàng từ `data` đảm bảo cột khớp thứ tự với `required_features`.
4. **Gọi `model.predict` và `model.predict_proba`** để lấy nhãn và xác suất.
5. **Trả về:** `{'label': int, 'label_name': 'Đúng Hạn'|'Trễ Hạn', 'confidence': float}`

**Lưu ý triển khai:**
- Nếu thiếu cột trong input, điền `np.nan` để `SimpleImputer` xử lý.
- Xử lý categorical unseen bằng `handle_unknown='ignore'` trong OneHotEncoder.

---

## 🔧 Ví dụ: Gọi hàm `predict_graduation`
```python
from src.model_utils import predict_graduation

sample = {
    'gender': 'Male',
    'major': 'Computer Science',
    'admission_type': 'Regular',
    'admission_score': 21.5,
    # các cột tính đến kỳ 5
    'gpa_sem1': 3.2, 'gpa_sem2': 3.1, 'gpa_sem3': 3.25, 'gpa_sem4': 3.3, 'gpa_sem5': 3.15,
    'credits_sem5': 90, 'failed_sem5': 0, 'warn_sem5': 0
}

result = predict_graduation(sample, semester_point=5)
print(result)
# -> {'label': 1, 'label_name': 'Đúng Hạn', 'confidence': 0.87}
```

---

## 🚀 Cài đặt & Chạy

### 1. Tạo môi trường & cài đặt
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` tối thiểu có thể gồm:
```
scikit-learn
pandas
numpy
joblib
streamlit
python-dotenv
```

### 2. Huấn luyện mô hình
```bash
python -m src.train
```
Script sẽ lưu 4 file `models/model_sem5.pkl` ... `model_sem8.pkl`.

### 3. Chạy Streamlit app
```bash
streamlit run Streamlit_App.py
```
Mở trình duyệt tới `http://localhost:8501` (hoặc cổng bạn cấu hình trong `.env`).

---

## 🧾 Gợi ý cấu trúc file quan trọng

### `src/Config.py`
- Định nghĩa `FIXED_FEATURES`, `CATEGORICAL_FEATURES`, `NUMERICAL_FEATURES`, `FEATURE_SETS`, `TARGET_COLUMN`, `MODEL_PATHS`.

### `src/model_utils.py`
- `load_model(path)`, `save_model(model, path)`, `prepare_input(data, required_features)`, `predict_graduation`.

### `src/train.py`
- Tải CSV, chuẩn hóa, huấn luyện, đánh giá, lưu model.

### `Streamlit_App.py` (gợi ý nội dung)
- Form nhập thông tin sinh viên (các trường cố định + các trường theo kỳ).
- Nút chọn `semester_point` (5/6/7/8).
- Gọi `predict_graduation` và hiển thị nhãn + thanh tiến độ confidence.

---

## 📌 Ghi chú & Best Practices
- Bảo toàn thứ tự cột khi tạo DataFrame đầu vào cho pipeline.
- Sử dụng `ColumnTransformer` để tránh leak thông tin giữa categorical/numeric.
- Lưu `training_features` (danh sách cột sau fit của preprocessor) nếu cần deploy ổn định.
- Đảm bảo `random_state` cố định cho reproducibility.

---

## 🔁 Phiên bản và Bảo trì
- Nên lưu cả `preprocessor` bên trong pipeline để khi deploy còn dùng lại bước tiền xử lý.
- Nếu thêm/loại đặc trưng, huấn luyện lại tất cả các mô hình Sem.

---

## 📚 Tài liệu tham khảo
- scikit-learn docs: Pipeline, ColumnTransformer, OneHotEncoder, SimpleImputer, RandomForestClassifier

```
