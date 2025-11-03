import pandas as pd
import joblib 
import os
from sklearn.linear_model import LinearRegression 

import sys

# lấy thư mục scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
# lấy thư mục gốc (predict_student_result)
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.Config import DATA_FILE, REG_MODEL_GPA2_PATH,REG_MODEL_GPA3_PATH,REG_MODEL_GPA4_PATH

def train_gpa_regressors():
      print(f'Loading the dât from: {DATA_FILE}')
      try:
            #đinh nghĩa chính xác DATA_FILE
            df = pd.read_csv(DATA_FILE)
      except FileNotFoundError:
            print("Lỗi: Không tìm thấy file data. vui lòng kiểm tra lại đường dẫn")
            return    
      
      """ Phần Huấn luyện mô hình Hồi quy tuyến tính(LinearRegression) """
      # chuyển hóa tên cột
      df.columns = df.columns.str.lower().str.strip()

      gpa_cols = [c for c in df.columns if c.startswith("gpa")]
      for col in gpa_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
      
      # mô hình dự đoán gpa năm 2
      df_gpa2 = df[['gpa_year1', 'gpa_year2']].dropna()
      if not df_gpa2.empty:
            X2 = df_gpa2[['gpa_year1']]
            y2 = df_gpa2['gpa_year2']
            model_gpa2 = LinearRegression()
            model_gpa2.fit(X2, y2)
            joblib.dump(model_gpa2, REG_MODEL_GPA2_PATH)
            print(f" Mô hình dự đoán GPA_YEAR2 đã huấn luyện và lưu tại: {REG_MODEL_GPA2_PATH}")
      
      # mô hình dự đoán gpa năm 3
      df_gpa3 = df[['gpa_year1', 'gpa_year2', 'gpa_year3']].dropna()
      if not df_gpa3.empty:
            X3 = df_gpa3[['gpa_year1', 'gpa_year2']]
            y3 = df_gpa3['gpa_year3']
            model_gpa3 = LinearRegression()
            model_gpa3.fit(X3, y3)
            joblib.dump(model_gpa3, REG_MODEL_GPA3_PATH)
            print(f" Mô hình dự đoán GPA_YEAR3 đã huấn luyện và lưu tại: {REG_MODEL_GPA3_PATH}")

      # mô hình dự đoán gpa năm 4
      df_gpa4 = df[['gpa_year1', 'gpa_year2', 'gpa_year3', 'gpa_year4']].dropna()
      if not df_gpa4.empty:
            X4 = df_gpa4[['gpa_year1', 'gpa_year2', 'gpa_year3']]
            y4 = df_gpa4['gpa_year4']
            model_gpa4 = LinearRegression()
            model_gpa4.fit(X4, y4)
            joblib.dump(model_gpa4, REG_MODEL_GPA4_PATH)
            print(f" Mô hình dự đoán GPA_YEAR4 đã huấn luyện và lưu tại: {REG_MODEL_GPA4_PATH}")


if __name__ == '__main__':
    train_gpa_regressors()


      


