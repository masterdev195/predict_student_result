import pandas as pd
import joblib 
from src.Config import (FINANCIAL_ORDER,NOMINAL_COLS,COLS_TO_DROP)

def preprocess_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
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

def Predict_student_status(new_data_dict: dict, model, training_features)-> dict:
      """
      Khi có input mới thì dự đoán 
      """

      df_new = pd.DataFrame([new_data_dict])

      # Tiền xử lý
      X_new_processed = preprocess_data(df_new, is_training=False)

      X_final = X_new_processed.reindex(columns=training_features, fill_value=0)

      # dự đoán 
      prediction = model.predict(X_final)[0]

      result_text = "Tốt nghiệp đúng hạn" if prediction ==1 else "Tốt nghiệp không đúng hạn"

      return{
            'prediction_text': result_text,
            'graduate_on_time': int((prediction))
      }

      