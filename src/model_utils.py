import pandas as pd
import joblib 
import numpy as np
from src.Config import (FINANCIAL_ORDER,NOMINAL_COLS,COLS_TO_DROP, ALL_NUMERIC_MEANS)

def preprocess_data(df: pd.DataFrame, is_training: bool = True,reg_gpa2=None, reg_gpa3=None, reg_gpa4=None) -> pd.DataFrame:
      df_copy = df.copy()
      
      # Chuẩn hóa tên cột
      df_copy.columns = df_copy.columns.str.lower().str.strip()

      # Thay missing values
      df_copy = df_copy.fillna(
            {
                  "finanacial_state" :"medium",
                  "major" : "Unknown",
                  "addmission_type": "Other",
                  "gender": "Other"
            }
      )

      # chuyển gpa về chuẩn dạng số
      gpa_cols = [c for c in df_copy.columns if c.startswith("gpa")]
      for col in gpa_cols:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")\
      
      # 2. KHỐI THÊM MỚI: Imputation cho các cột số (GPA và Non-GPA)
      # Đây là bước BẮT BUỘC để điền các giá trị như attendance_rate, failed_courses, v.v.
      numeric_cols_to_impute = [
            'total_credits_required', 'failed_courses', 'academic_warnings', 
            'attendance_rate', 'assignment_submission_rate', 'extra_activities',
            'achieved_scholarship', 'total_cumulative_gpa'
      ]

      for col in numeric_cols_to_impute:
          # Ép kiểu số (vì các cột này có thể bị thiếu từ input)
          df_copy[col] = pd.to_numeric(df_copy.get(col, np.nan), errors="coerce")
          
          # Điền giá trị thiếu bằng giá trị trung bình từ Config.py
          fill_val = ALL_NUMERIC_MEANS.get(col, 0.5) 
          df_copy[col] = df_copy[col].fillna(fill_val)

            # CHÚ THÍCH THAY ĐỔI MỚI: Xử lý GPA_YEAR1 thiếu
      # Vì gpa_year1 là biến đầu vào độc lập, ta không dùng hồi quy mà dùng giá trị trung bình để điền ngay.
      fill_value_gpa1 = ALL_NUMERIC_MEANS.get('gpa_year1', 2.5) 
      df_copy['gpa_year1'] = df_copy['gpa_year1'].fillna(fill_value_gpa1)
      # thực hiện imputation dựa trên hồi quy(only trong chế độ dự đoán)
      if not is_training:
            # CHÚ THÍCH THAY ĐỔI 2: Đảo ngược thứ tự dự đoán (2 -> 3 -> 4) 
            # để đảm bảo các biến đầu vào (X) không có NaN, vì gpa_year1 đã được điền ở bước trên.
            
            # 1. điền giá trị còn thiếu của gpa_year2 (dùng gpa_year1 đã được điền)
            if df_copy['gpa_year2'].isna().any() and reg_gpa2 is not None:
                  missing_gpa2_rows = df_copy['gpa_year2'].isna()
                  X_gpa2 = df_copy.loc[missing_gpa2_rows, ['gpa_year1']]
                  predicted_gpa2 = reg_gpa2.predict(X_gpa2)
                  df_copy.loc[missing_gpa2_rows, 'gpa_year2'] = np.clip(predicted_gpa2, 0.0, 4.0)
                  
            # 2. điền giá trị còn thiếu của gpa_year3 (dùng gpa_year1, gpa_year2 đã được điền)
            if df_copy['gpa_year3'].isna().any() and reg_gpa3 is not None:
                  missing_gpa3_rows = df_copy['gpa_year3'].isna()
                  X_gpa3 = df_copy.loc[missing_gpa3_rows, ['gpa_year1', 'gpa_year2']]
                  predicted_gpa3 = reg_gpa3.predict(X_gpa3)
                  df_copy.loc[missing_gpa3_rows, 'gpa_year3'] = np.clip(predicted_gpa3, 0.0, 4.0)

            # 3. điền giá trị còn thiếu của gpa_year4 (dùng gpa_year1, gpa_year2, gpa_year3 đã được điền)
            if df_copy['gpa_year4'].isna().any() and reg_gpa4 is not None:
                  missing_gpa4_rows = df_copy['gpa_year4'].isna()
                  X_gpa4 = df_copy.loc[missing_gpa4_rows, ['gpa_year1', 'gpa_year2', 'gpa_year3']]
                  predicted_gpa4 = reg_gpa4.predict(X_gpa4)
                  df_copy.loc[missing_gpa4_rows, 'gpa_year4'] = np.clip(predicted_gpa4, 0.0, 4.0)

             
      # Code điền thiếu 0.0 cho Training (Giữ nguyên logic cũ của bạn)
      for col in ['gpa_year2','gpa_year3','gpa_year4']:
            fill_value = ALL_NUMERIC_MEANS.get(col,2.5)
            df_copy[col] = df_copy[col].fillna(fill_value)
                  
      # ordinal Encoding 
      df_copy['financial_state_encoded'] = df_copy['financial_state'].map(FINANCIAL_ORDER)
      df_copy = df_copy.drop('financial_state', axis =1)
      
      # One-Hot Encoding 
      df_encoded = pd.get_dummies(df_copy, columns =NOMINAL_COLS, drop_first = True)

      # delete unnecessary columns
      df_final = df_encoded.drop(COLS_TO_DROP, axis=1, errors='ignore')

      if is_training and 'graduate_on_time' in df_final.columns:
            df_final = df_final.drop('graduate_on_time', axis =1 )
      return df_final

def Predict_student_status(new_data_dict: dict, model, training_features, reg_gpa2=None, reg_gpa3=None, reg_gpa4=None)-> dict:
      """
      Khi có input mới thì dự đoán 
      """

      df_new = pd.DataFrame([new_data_dict])

      # Tiền xử lý
      X_new_processed = preprocess_data(df_new, is_training=False,
                                        reg_gpa2=reg_gpa2, reg_gpa3 = reg_gpa3, reg_gpa4=reg_gpa4)

      X_final = X_new_processed.reindex(columns=training_features, fill_value=0)

      # Dữ liệu cuối cùng mà mô hình nhìn thấy (rất quan trọng để debug)
      final_features = X_final.iloc[0].to_dict()

      # dự đoán 
      prediction = model.predict(X_final)[0]
      prediction_proba = model.predict_proba(X_final)[0]

      result_text = "Tốt nghiệp đúng hạn" if prediction ==1 else "Tốt nghiệp không đúng hạn"

      return{
            'prediction_text': result_text,
            'graduate_on_time': int((prediction)),
            'probability_on_time': round(prediction_proba[1], 4), # Xác suất tốt nghiệp đúng hạn
            'probability_late': round(prediction_proba[0], 4),   # Xác suất tốt nghiệp không đúng hạn
            'imputed_features': final_features # TRẢ VỀ CÁC GIÁ TRỊ ĐÃ ĐƯỢC ĐIỀN
      }

      