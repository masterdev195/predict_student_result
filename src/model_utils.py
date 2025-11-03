import pandas as pd
import joblib 
from src.Config import (FINANCIAL_ORDER,NOMINAL_COLS,COLS_TO_DROP, GPA_MEANS)

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
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0)
      # thực hiện imputation dựa trên hồi quy(only trong chế độ dự đoán)
      if not is_training:
            # điền giá trị còn thiếu của gpa_year4
            if df_copy['gpa_year4'].isna().any() and reg_gpa4 is not None:
                  X_gpa4 = df_copy[['gpa_year1', 'gpa_year2', 'gpa_year3']].values
                  predicted_gpa4 = reg_gpa4.predict(X_gpa4.reshape(1,-1))[0]

                  predicted_gpa4 = max(0.0, min(4.0, predicted_gpa4))
                  df_copy['gpa_year4'] = df_copy["gpa_gpa4"].fillna(predicted_gpa4)

            # điền giá trị còn thiếu của gpa_year3 (dựa trên gpa1, gpa2)
            if df_copy['gpa_year3'].isna().any() and reg_gpa3 is not None:
                  X_gpa3 = df_copy[['gpa_year1', 'gpa_year2']].values
                  predicted_gpa3 = reg_gpa3.predict(X_gpa3.reshape(1, -1))[0]
                  predicted_gpa3 = max(0.0, min(4.0, predicted_gpa3))
                  df_copy['gpa_year3'] = df_copy['gpa_year3'].fillna(predicted_gpa3)
                  
            # điền giá trị còn thiếu của gpa_year2 (dựa trên gpa1)
            if df_copy['gpa_year2'].isna().any() and reg_gpa2 is not None:
                  X_gpa2 = df_copy[['gpa_year1']].values
                  predicted_gpa2 = reg_gpa2.predict(X_gpa2.reshape(1, -1))[0]
                  predicted_gpa2 = max(0.0, min(4.0, predicted_gpa2))
                  df_copy['gpa_year2'] = df_copy['gpa_year2'].fillna(predicted_gpa2)

            # Điền giá tị còn thiếu gpa_year1 (Fallback Mean Imputation)
            if df_copy['gpa_year1'].isna().any():
                  mean_gpa1 = GPA_MEANS.get('gpa_year1', 3.0) 
                  df_copy['gpa_year1'] = df_copy['gpa_year1'].fillna(mean_gpa1)
      else: 
            # Code điền thiếu 0.0 cho Training (Giữ nguyên logic cũ của bạn)
            for col in gpa_cols:
                  df_copy[col] = df_copy[col].fillna(0.0)
                  
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

      # dự đoán 
      prediction = model.predict(X_final)[0]

      result_text = "Tốt nghiệp đúng hạn" if prediction ==1 else "Tốt nghiệp không đúng hạn"

      return{
            'prediction_text': result_text,
            'graduate_on_time': int((prediction))
      }

      