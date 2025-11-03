# 🎓 Predict Student Graduation – Machine Learning Project

## 📌 Giới thiệu

Dự án này xây dựng một mô hình máy học nhằm **dự đoán khả năng sinh viên có thể tốt nghiệp hay không**, dựa trên các thông tin đầu vào như:

* Giới tính
* Chuyên ngành
* Hình thức tuyển sinh
* Hoàn cảnh tài chính
* GPA các năm học

Dự án gồm 2 phần chính:
✅ Huấn luyện mô hình bằng Python (scikit-learn)
✅ Xây dựng giao diện dự đoán nhanh bằng Streamlit

---

## 🛠 Công nghệ sử dụng

| Thành phần       | Công nghệ     |
| ---------------- | ------------- |
| Ngôn ngữ         | Python 3.x    |
| Xử lý dữ liệu    | pandas, numpy |
| Machine Learning | scikit-learn  |
| Lưu mô hình      | joblib        |
| Giao diện Web    | Streamlit     |

---

## 📥 Cài đặt

Clone dự án:

```bash
git clone https://github.com/<masterdev195>/Predict_Student_Result
cd Predict_Student_Result
```

Tạo môi trường ảo:

```bash
python -m venv venv
venv\Scripts\activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

---

## ▶️ Chạy ứng dụng Streamlit

```bash
streamlit run Streamlit_App.py
```

## thứ tự chạy 
## 🎯 Huấn luyện mô hình

```bash
python -m src.train
```
## chạy IPA
```bash
python -m src.App
```
## chạy streamlit
```bash
streamlit run Streamlit_App.py
```




File mô hình sau khi huấn luyện sẽ được lưu tại:

```
model/
    rf_graduate_model.pkl
```

---

## 📁 Cấu trúc thư mục dự án

```
Predict_Student_Result/
│
├── Streamlit_App.py          # Ứng dụng Streamlit
├── requirements.txt
├── README.md
│
├── src/
│   ├── App.py              # Chạy IPA flask 
│   ├── Config.py             # Load dữ liệu, load features
│   ├── model_utils.py            # Hàm xử lý dữ liệu và dữ liệu đầu vào
│   └── train.py              # train và lưu mô hình
│
├── data/
│   └── student_data.csv      # Dữ liệu nguồn
│
└── model/
    └── student_model.pkl     # Mô hình đã train
    |── training_features.pkl    
```

---



## 👨‍💻 Tác giả

**Họ tên:** <Hoàng Đức Tài>
**Email:** <taih02551@gmail.com>
**GitHub:** <https://github.com/masterdev195>

---

## ✅ Ghi chú

Bạn có thể chỉnh sửa README để phù hợp hơn với cấu trúc thực tế của dự án hoặc thêm các phần như License, Demo, Kết quả mô hình.
