# src/model_utils.py
import os
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.Config import TARGET_COLUMN, CATEGORICAL_FEATURES, MODELS_DIR,FEATURE_SETS, FIXED_FEATURES

def load_data(file_path):
    """Tải dữ liệu từ file CSV."""
    return pd.read_csv(file_path)

def get_preprocessor(X_train, features_for_sem):
    """Tạo và fit ColumnTransformer cho việc tiền xử lý."""
    numerical_features = [col for col in features_for_sem if col not in CATEGORICAL_FEATURES]
    numerical_pipeline = Pipeline(steps=[
        # Dùng 'constant' và fill_value=0 hoặc 'mean' tùy ngữ cảnh.
        # Ở đây ta giả định dữ liệu đã được làm sạch tốt, nên chỉ cần giữ nguyên.
        ('imputer', SimpleImputer(strategy='mean')) 
    ])
    # 1. Tạo Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
            ('num', numerical_pipeline, numerical_features)
        ],
        remainder='drop'
    )
    
    # 2. Fit preprocessor trên tập huấn luyện
    preprocessor.fit(X_train)
    return preprocessor

def save_model(model, semester):
    """Lưu mô hình đã huấn luyện."""
    model_name = os.path.join(MODELS_DIR, f'model_sem{semester}.pkl')
    joblib.dump(model, model_name)
    print(f"-> Đã lưu mô hình Sem {semester} vào: {model_name}")

def load_model(semester):
    """Tải mô hình đã lưu."""
    model_name = os.path.join(MODELS_DIR, f'model_sem{semester}.pkl')
    try:
        return joblib.load(model_name)
    except FileNotFoundError:
        return None
    
def predict_graduation(data: dict, semester_point: int, models: dict):
       
    #Tải mô hình phù hợp và dự đoán khả năng tốt nghiệp.
    model = models.get(semester_point)
    if not model:
        return None, None, f"Mô hình cho Kỳ {semester_point} chưa được tải hoặc không tồn tại."

    # Lấy bộ đặc trưng cần thiết cho mô hình này
    required_features = FEATURE_SETS[semester_point]
    
    try:
        # Chuẩn bị DataFrame đầu vào
        # 1. Chuyển dict sang DataFrame 1 hàng
        # 2. Lọc/sắp xếp DataFrame theo đúng required_features của mô hình
        X_new = pd.DataFrame([data])[required_features]
        
        # Dự đoán
        prediction = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0]
        
        result_class = "Đúng Hạn (On Time)" if prediction == 1 else "Không Đúng Hạn (Delayed/Drop)"
        confidence = prob[prediction]
        
        return result_class, confidence, None
    
    except Exception as e:
        # Bắt lỗi nếu dữ liệu đầu vào thiếu cột hoặc sai định dạng
        return None, None, f"Lỗi trong quá trình dự đoán (Kiểm tra dữ liệu đầu vào): {e}"