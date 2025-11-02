import streamlit as st
import joblib 
import pandas as pd
from src.model_utils import Predict_student_status
from src.Config import MODEL_PATH,FEATURES_PATH

@st.cache_resource
def load_model():
      model = joblib.load(MODEL_PATH)
      features = joblib.load(FEATURES_PATH)
      return model,features
model, features = load_model()

""" Giao diện web """
 
st.title("Dự đoán khả năng tốt nghiệp đúng hạn")
st.title("Nhập thông tin sinh viên bên dưới để dự đoán khả năng tốt nghiệp.")
gender = st.selectbox("Giới tính", ["Male", "Female"])
major = st.text_input("Ngành học (vd: IT, Business, Education)")
admission_type = st.text_input("Kiểu xét tuyển(vd: tuyển thảng, thi,...)")
financial_state = st.selectbox(
      "Điều kiện tài chính", ["Difficult", "Average", "Stable"]
)

st.subheader("GPA theotuwngf năm")

Year = st.selectbox(
      "Bạn đang là sinh viên năm mấy", 
      options = [1,2,3,4],
      format_func=lambda x: f"Năm {x}"
)

gpa_year1 = gpa_year2 = gpa_year3 =gpa_year4 =0.0
if Year >=1:
      gpa_year1 = st.number_input("GPA năm 1", min_value = 0.0, max_value =4.0, step=0.01)
if Year >=2:
      gpa_year2 = st.number_input("GPA năm 2", min_value = 0.0, max_value =4.0, step=0.01)
if Year >=3:
      gpa_year3 = st.number_input("GPA năm 3", min_value = 0.0, max_value =4.0, step=0.01)
if Year >=4:
      gpa_year4 = st.number_input("GPA năm 4", min_value = 0.0, max_value =4.0, step=0.01)



if st.button("Dự đoán"):
      if major.strip() =="" or admission_type =="":
            st.error("vui lòng nhập đu thông tin")
      else:
            # tạo dict input
            data = {
                  'gender': gender,
                  'major': major,
                  'admission_type': admission_type,
                  'financial_state': financial_state,

                  'gpa_year1': gpa_year1,
                  'gpa_year2': gpa_year2,
                  'gpa_year3': gpa_year3,
                  'gpa_year4': gpa_year4



            }

            # gọi hàm predict
            result = Predict_student_status(
                  new_data_dict= data,
                  model=model,
                  training_features= features
            )

            #hiển thị

            st.success(f"Kết quả dự đoán: {result['prediction_text']}")
            st.write("**Chi tiết dự đoán:**")
            st.json(result)
