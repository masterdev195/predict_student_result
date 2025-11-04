# 🎓 Dự án: Dự đoán Khả năng Tốt nghiệp Đúng hạn của Sinh viên

## 🌟 Giới thiệu
Đây là một dự án phân loại nhị phân (Binary Classification) nhằm dự đoán khả năng một sinh viên sẽ **tốt nghiệp đúng hạn** hay không, dựa trên các yếu tố học tập, tài chính và xã hội.

Mô hình được xây dựng trên thuật toán **Random Forest** và tích hợp cơ chế **điền giá trị thiếu (Imputation)** bằng **Hồi quy Tuyến tính (Linear Regression)** để xử lý các điểm GPA còn thiếu trong quá trình dự đoán.

**Điểm mạnh của giải pháp:**
- Khắc phục lỗi điền giá trị `0.0` cho các thuộc tính bị thiếu.
- Mô hình dự đoán dựa trên giá trị trung bình hợp lý thay vì dữ liệu bị lệch cực đoan.

---

## 🛠️ Cấu trúc Dự án
```
Predict_Student_Result/
├── data/
│   └── graduated_students_dataset.csv  # Dữ liệu gốc
├── models/
│   ├── rf_graduate_model.pkl           # Mô hình Random Forest đã huấn luyện
│   ├── training_features.pkl           # Danh sách các đặc trưng (cột)
│   └── reg_gpaX_model.pkl              # Các mô hình hồi quy GPA (2, 3, 4)
│
├── scripts/
│   └── train_imputation_regressors.py            # Các mô hình hồi quy GPA (2, 3, 4)
├── src/
│   ├── Config.py                       # Cấu hình, giá trị trung bình (ALL_NUMERIC_MEANS)
│   ├── model_utils.py                  # Hàm tiền xử lý (preprocess_data) và dự đoán
│   ├── App.py                          # Ứng dụng Flask API (được đề cập)
│   ├── train.py                        # Script huấn luyện mô hình chính(randomforest)
│   └── __init__.py                          
├── Streamlit_App.py                    # Ứng dụng giao diện Streamlit
├── requirement.txt                     # các thự viện cần dùng
└── README.md                           # Tài liệu dự án                          
```

---

## 🚀 Cài đặt và Khởi chạy
### 1. Tạo môi trường
```bash
# Tạo môi trường ảo (nếu chưa có)
python -m venv venv

# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate

# Cài đặt thư viện cần thiết
pip install -r requirements.txt
```

---

### 2. Quy trình Huấn luyện Mô hình
Bạn **bắt buộc** phải chạy lại toàn bộ quá trình để tạo ra các mô hình mới đã khắc phục lỗi điền giá trị thiếu.

#### ✅ 1. Huấn luyện các mô hình Imputation
```bash
python -m scripts.train_imputation_regressors
```

#### ✅ 2. Huấn luyện mô hình Phân loại chính (Random Forest)
```bash
python -m src.train
```
#### ✅ 3. Chạy IPA
```bash
python -m src.App
```

---

### 4. Khởi chạy ứng dụng Web (Streamlit)
```bash
streamlit run Streamlit_App.py
```

---

## 📌 Các Tính năng và Giải pháp
- **Imputation Vững chắc:** Sử dụng giá trị trung bình thực tế (`ALL_NUMERIC_MEANS`) thay vì 0.0 gây sai lệch.
- **Hồi quy GPA Thông minh:** Dự đoán GPA năm tiếp theo dựa trên các năm trước.
- **Kiểm soát Phạm vi:** Giá trị GPA dự đoán được cắt (clip) trong khoảng [0.0, 4.0].
- **Random Forest mạnh mẽ:** Khả năng phân loại tốt và ổn định.

---

## 🧰 Công nghệ sử dụng
- **Ngôn ngữ:** Python
- **Triển khia IPA:** Flask, python-dotenv
- **Thư viện ML:** Scikit-learn (RandomForestClassifier, LinearRegression)
- **Lưu và tải mô hình:** joblib
- **Xử lý dữ liệu:** Pandas, NumPy
- **Giao diện:** Streamlit
