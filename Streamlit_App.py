import streamlit as st
import pandas as pd
import os
import sys

# Thêm thư mục src vào PATH để import
sys.path.insert(0, os.path.abspath('src'))

# Import hàm dự đoán từ model_utils
from src.model_utils import load_data, load_model, predict_graduation 
from src.Config import SEMESTER_POINTS, DATA_FILE_PATH

# --- Cấu hình Ban đầu ---
MODELS_READY = all(os.path.exists(f"models/model_sem{s}.pkl") for s in SEMESTER_POINTS)
# ... (Phần code kiểm tra và chạy huấn luyện giữ nguyên) ...

# Tải tất cả 4 mô hình (nếu có)
MODELS = {s: load_model(s) for s in SEMESTER_POINTS}

# Tải dữ liệu để lấy danh sách Major, Admission Type
df_ref = load_data(DATA_FILE_PATH)
MAJOR_LIST = sorted(df_ref['major'].unique().tolist())
ADMISSION_TYPES = sorted(df_ref['admission_type'].unique().tolist())

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Hệ Thống Dự Đoán Tốt Nghiệp Sớm", layout="wide")

st.title("🎓 Hệ Thống Dự Đoán Khả Năng Tốt Nghiệp Đúng Hạn")
st.markdown("Sử dụng mô hình Random Forest để dự đoán khả năng sinh viên tốt nghiệp đúng hạn.")

# BƯỚC 1: CHỌN KỲ HỌC (NGOÀI FORM) ĐỂ KÍCH HOẠT TÍNH ĐỘNG

semester_map = {
    5: "Năm 3, Kỳ 1 (End Sem 5)",
    6: "Năm 3, Kỳ 2 (End Sem 6)",
    7: "Năm 4, Kỳ 1 (End Sem 7)",
    8: "Năm 4, Kỳ 2 (End Sem 8)",
}

# Chọn kỳ học hiện tại (kích hoạt rerun script)
semester_point = st.selectbox(
    " Sinh viên đang ở thời điểm (Chọn Mô hình Dự đoán)", 
    options=list(semester_map.keys()), 
    format_func=lambda x: semester_map[x], 
    index=0
)


#  FORM NHẬP LIỆU (SỬ DỤNG GIÁ TRỊ semester_point ĐÃ CHỌN)

with st.form("prediction_form"):
    st.header("2. Nhập Thông tin Đặc trưng")

    col1, col2, col3 = st.columns(3)
    
    # Thông tin cơ bản
    with col1:
        st.subheader("Thông tin Cơ bản")
        full_name = st.text_input("Họ và Tên Sinh viên", "Nguyễn Văn A")
        gender = st.selectbox("Giới tính", ['Nam', 'Nữ'])
        major = st.selectbox("Ngành học (Major)", MAJOR_LIST)
        
    # Thông tin Học tập Tích lũy
    with col2:
        st.subheader("Trạng thái Tích lũy")
        admission_type = st.selectbox("Phương thức Xét tuyển", ADMISSION_TYPES)
        admission_score = st.number_input("Điểm thi Đầu vào (VD: 26.53)", min_value=0.0, max_value=30.0, value=26.0, step=0.01)
        
        # Tên cột tích lũy sẽ thay đổi dựa trên semester_point đã chọn
        current_credits_col = f'credits_sem{semester_point}'
        current_failed_col = f'failed_sem{semester_point}'
        current_warn_col = f'warn_sem{semester_point}'

        credits_acc = st.number_input(f"Số Tín chỉ tích lũy (đến hết Sem {semester_point-1} hoặc {semester_point})", min_value=0, value=75)
        failed_count = st.number_input(f"Số môn/tín chỉ Trượt tích lũy (đến hết Sem {semester_point-1} hoặc {semester_point})", min_value=0, value=0)
        warn_count = st.number_input(f"Số lần bị Cảnh báo Học vụ tích lũy (đến hết Sem {semester_point-1} hoặc {semester_point})", min_value=0, value=0)
        
    # Cột 3: GPA Từng Kỳ
    with col3:
        st.subheader("Điểm GPA Từng Kỳ")
        
        gpa_inputs = {}
        max_sem_input = semester_point
        
        # Hiển thị GPA từng kỳ học từ Sem 1 đến kỳ hiện tại (max_sem_input thay đổi)
        for i in range(1, max_sem_input):
            # Tính GPA trung bình tích lũy cho các kỳ đã qua (từ 1 đến i-1)
            # Khởi tạo giá trị mặc định dựa trên kỳ học
            if i <= 4:
                default_gpa = 3.0
            elif i == 5:
                default_gpa = 2.8
            else:
                default_gpa = 2.5
                
            gpa_inputs[f'gpa_sem{i}'] = st.number_input(f"GPA Kỳ {i}", min_value=0.0, max_value=4.0, value=default_gpa, step=0.01)

    # Nút gửi
    submitted = st.form_submit_button("Dự đoán Khả năng Tốt nghiệp")

if submitted:
    # ... (Phần logic xử lý khi submitted giữ nguyên) ...
    if not MODELS_READY:
        st.error("Lỗi: Các mô hình chưa được tải hoặc huấn luyện thành công. Vui lòng kiểm tra console.")
    else:
        # Chuẩn bị dữ liệu cho mô hình
        input_data = {
            'gender': gender,
            'major': major,
            'admission_type': admission_type,
            'admission_score': admission_score,
            **gpa_inputs, # Thêm các trường GPA theo kỳ
            current_credits_col: credits_acc,
            current_failed_col: failed_count,
            current_warn_col: warn_count
        }
        
        # Dự đoán
        result_class, confidence, error_msg = predict_graduation(input_data, semester_point, MODELS)

        if error_msg:
            st.error(f"Lỗi: {error_msg}")
        else:
            st.success(f"✅ Dự đoán Thành công cho sinh viên **{full_name}**")
            
            # Hiển thị kết quả
            if result_class == "Đúng Hạn (On Time)":
                st.balloons()
                st.markdown(f"### Kết quả Dự đoán: <span style='color:green'>**{result_class}**</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"### Kết quả Dự đoán: <span style='color:red'>**{result_class}**</span>", unsafe_allow_html=True)
                
            st.markdown(f"**Độ Tin cậy:** `{confidence*100:.2f}%` (Sử dụng Mô hình End Sem {semester_point})")
            
            if result_class == "Không Đúng Hạn (Delayed/Drop)":
                 st.warning("⚠️ **Khuyến nghị:** Cần có sự can thiệp và hỗ trợ kịp thời để cải thiện tình hình học tập.")