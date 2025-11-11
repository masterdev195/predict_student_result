# src/train.py

import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from src.Config import (
    DATA_FILE_PATH, FEATURE_SETS, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, N_ESTIMATORS, MODELS_DIR, SEMESTER_POINTS
)
from src.model_utils import load_data, get_preprocessor, save_model

def train_multi_stage_models():
    
    #Huấn luyện mô hình Random Forest riêng biệt cho từng kỳ học (Sem 5, 6, 7, 8).
    
    # 1. Tạo thư mục models
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 2. Tải toàn bộ dữ liệu
    df = load_data(DATA_FILE_PATH)
    
    print("--- BẮT ĐẦU HUẤN LUYỆN 4 MÔ HÌNH DỰ ĐOÁN ---")
    
    # 3. Lặp qua từng điểm thời gian (kỳ học)
    for sem in SEMESTER_POINTS:
        print(f"\n[MODEL SEM {sem}] Đang huấn luyện mô hình cho sinh viên đến hết Kỳ {sem}...")
        
        # Lấy bộ đặc trưng cụ thể cho kỳ học này
        features = FEATURE_SETS[sem]
        X = df[features]
        y = df[TARGET_COLUMN]
        
        # Chia tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Tạo Preprocessor và Pipeline
        preprocessor = get_preprocessor(X_train, features)
        
        rf_classifier = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, 
            random_state=RANDOM_STATE, 
            class_weight='balanced'
        )
        
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', rf_classifier)
        ])
        
        # Huấn luyện mô hình
        model_pipeline.fit(X_train, y_train)
        
        # Đánh giá mô hình
        y_pred = model_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (Correct Graduation - Class 1): {report['1']['f1-score']:.4f}")

        feature_names_out = model_pipeline['preprocessor'].get_feature_names_out()
        # 2. Lấy độ quan trọng từ Random Forest
        importances = model_pipeline['classifier'].feature_importances_  
        # 3. Tạo DataFrame để dễ dàng sắp xếp
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names_out,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\n--- TOP 10 FEATURE IMPORTANCE ---")
        print(feature_importance_df.head(10).to_string(index=False))
        print("-----------------------------------")
        
        # Lưu mô hình
        save_model(model_pipeline, sem)

if __name__ == '__main__':
    train_multi_stage_models()