import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.Config import DATA_FILE, MODEL_PATH, FEATURES_PATH
from src.model_utils import preprocess_data

def train_and_save_model():
      print("Chào anh tài đẹp trai")
      print("Bắt đầu xử lý dữ liệu và huấn luyện mô hình ")

      model_dir = os.path.dirname(MODEL_PATH)
      os.makedirs(model_dir, exist_ok= True)

      # Download data and pre contr
      df_raw = pd.read_csv(DATA_FILE)

      X = preprocess_data(df_raw.copy(), is_training=True)
      Y = df_raw['graduate_on_time']

      TRAIN_FEATURES = X.columns
      
      # Chia tập train
      X_train, X_test, Y_train, Y_test = train_test_split(
            X,Y, test_size=0.2, random_state=42,stratify=Y
      )

      # Huấn luyện
      rf_model = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
      rf_model.fit(X_train,Y_train)

      # Đóng gói 
      joblib.dump(rf_model, MODEL_PATH)
      joblib.dump(TRAIN_FEATURES, FEATURES_PATH)

      print("\n --- HOÀN TẤT HUẤN LUYỆN VÀ ĐÓNG GÓI")
      print(f"Độ chính xác trên tập kiểm tra: {rf_model.score(X_test,Y_test):.4f}")
      print(f"Đã lưu mô hình vào'{MODEL_PATH}'")
if __name__ == '__main__':
      train_and_save_model()