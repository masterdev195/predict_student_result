# 🎓 Dự án: Dự đoán Khả năng Tốt nghiệp Đúng hạn của Sinh viên

## 🌟 Giới thiệu

Đây là một dự án **Phân loại Nhị phân (Binary Classification)** nhằm dự đoán khả năng một sinh viên sẽ **tốt nghiệp đúng hạn** (lớp 1) hay **không** (lớp 0), dựa trên các thông tin học tập và cá nhân tích lũy theo từng kỳ học.

Giải pháp sử dụng mô hình **Random Forest Classifier** và triển khai theo phương pháp **Dự đoán đa giai đoạn (Multi-Stage Prediction)**, cho phép dự đoán tại 4 thời điểm quan trọng trong quá trình học tập: **Cuối kỳ 5, Kỳ 6, Kỳ 7 và Kỳ 8**.

### 💡 Điểm nổi bật của Mô hình:

- **Dự đoán Đa giai đoạn**: Hỗ trợ 4 mô hình độc lập (Sem 5, 6, 7, 8) với các đặc trưng tích lũy khác nhau, giúp dự đoán sớm hơn.  
- **Tiền xử lý Tích hợp**: Sử dụng `ColumnTransformer` và `Pipeline` để tự động hóa việc mã hóa **One-Hot (OneHotEncoder)** cho các đặc trưng phân loại (`gender`, `major`, `admission_type`) và **điền giá trị thiếu (SimpleImputer)** cho các đặc trưng số.  
- **Giao diện Trực quan**: Ứng dụng web được xây dựng bằng **Streamlit** cho phép người dùng nhập thông tin sinh viên và nhận kết quả dự đoán một cách dễ dàng.

---

## 🧱 Cấu trúc Dự án

Cấu trúc thư mục được tổ chức rõ ràng để phân tách các thành phần dữ liệu, mô hình, mã nguồn và giao diện người dùng.
Predict_Student_Result/
│
├── data/
│ └── graduation_dataset_final1.csv # Dữ liệu gốc
│
├── models/
│ ├── model_sem5.pkl # Mô hình Random Forest Sem 5
│ ├── model_sem6.pkl # Mô hình Random Forest Sem 6
│ ├── model_sem7.pkl # Mô hình Random Forest Sem 7
│ └── model_sem8.pkl # Mô hình Random Forest Sem 8
│
├── src/
│ ├── Config.py # Cấu hình hằng số: đường dẫn, tên cột, bộ đặc trưng...
│ ├── model_utils.py # Các hàm hỗ trợ tải/lưu mô hình, tiến hành dự đoán
│ └── train.py # Script huấn luyện 4 mô hình
│
├── Streamlit_App.py # Ứng dụng Web Streamlit
├── .env # Cấu hình môi trường (ví dụ: PORT)
├── requirements.txt # Danh sách thư viện Python
└── README.md # Tài liệu dự án

---

## ⚙️ Chi tiết Mô hình và Kỹ thuật

### 1. Dữ liệu và Đặc trưng

- **File Dữ liệu**: `data/graduation_dataset_final1.csv`
- **Biến Mục tiêu (TARGET_COLUMN)**: `target`  
  - `1`: Tốt nghiệp đúng hạn  
  - `0`: Tốt nghiệp trễ/không tốt nghiệp  

**Đặc trưng Cố định** (`FIXED_FEATURES`):  
`gender`, `major`, `admission_type`, `admission_score`

**Đặc trưng Tùy biến**:  
Các đặc trưng học tập như `gpa_semX`, `credits_semX`, `failed_semX`, `warn_semX` được thêm vào tương ứng với từng kỳ học.

➡ Tham khảo `src/Config.py` để xem các bộ đặc trưng chi tiết (`FEATURE_SETS`).

---

### 2. Pipeline Huấn luyện

Mỗi mô hình (Sem 5, 6, 7, 8) được huấn luyện thông qua một **Pipeline** bao gồm hai bước chính (xem `src/train.py`):

#### 1️⃣ `preprocessor` (`ColumnTransformer`)
- **Phân loại (`CATEGORICAL_FEATURES`)**: xử lý bằng `OneHotEncoder`.  
- **Số học**: xử lý bằng `SimpleImputer(strategy='mean')` để điền các giá trị thiếu (nếu có) bằng giá trị trung bình.  

#### 2️⃣ `classifier` (`RandomForestClassifier`)
- Sử dụng tham số `class_weight='balanced'` để xử lý tình trạng mất cân bằng lớp (nếu có).

---

### 3. Hàm Dự đoán (`predict_graduation`)

Trong `src/model_utils.py`, hàm `predict_graduation` thực hiện các bước:

1. Tải mô hình phù hợp với `semester_point` được cung cấp.  
2. Lấy bộ đặc trưng cần thiết (`required_features`) từ `FEATURE_SETS`.  
3. Tạo `DataFrame` đầu vào từ dữ liệu người dùng (`data`), đảm bảo thứ tự cột khớp với các đặc trưng huấn luyện.  
4. Thực hiện dự đoán và tính toán độ tin cậy (`predict_proba`).  
5. Trả về kết quả **(Đúng Hạn / Trễ Hạn)** và độ tin cậy.

---

## 🚀 Cài đặt và Khởi chạy

### 1. Cài đặt Môi trường

```bash
# Tạo môi trường ảo (nếu chưa có)
python -m venv venv

# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate

# Cài đặt thư viện cần thiết
pip install -r requirements.txt
```

### 2. Huấn luyện Mô hình
```
Bạn cần chạy script huấn luyện để tạo ra 4 mô hình (.pkl) trong thư mục models/:

python -m src.train

Lưu ý: Script src.train sẽ tự động tải dữ liệu, tiền xử lý, huấn luyện 4 mô hình, đánh giá và lưu chúng.
```


### 3. Khởi chạy Ứng dụng Web (Streamlit)
```
Chạy ứng dụng web để tương tác với các mô hình đã huấn luyện:

streamlit run Streamlit_App.py


Ứng dụng sẽ khởi chạy trên trình duyệt của bạn (thường là http://localhost:8501).

🧠 Công nghệ Sử dụng
Thành phần	Công nghệ
Ngôn ngữ	Python
Machine Learning	scikit-learn (RandomForestClassifier, Pipeline, ColumnTransformer)
Xử lý Dữ liệu	pandas, numpy, joblib
Giao diện Web	Streamlit
Cấu hình	python-dotenv
```

✨ Tác giả: Dany
📅 Phiên bản: 1.0
📁 Dự án Học máy – Dự đoán Khả năng Tốt nghiệp Đúng hạn của Sinh viên
