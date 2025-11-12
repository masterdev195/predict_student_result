# src/train.py

import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,confusion_matrix,
                             roc_auc_score,log_loss,precision_score,recall_score)
from sklearn.pipeline import Pipeline
from src.Config import (
    DATA_FILE_PATH, FEATURE_SETS, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, N_ESTIMATORS, MODELS_DIR, SEMESTER_POINTS
)
from src.model_utils import load_data, get_preprocessor, save_model

def train_multi_stage_models():
    
    #Huấn luyện mô hình Random Forest riêng biệt cho từng kỳ học
    
    #  Tạo thư mục models
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Tải toàn bộ dữ liệu
    df = load_data(DATA_FILE_PATH)
    
    print("--- BẮT ĐẦU HUẤN LUYỆN 4 MÔ HÌNH DỰ ĐOÁN ---")
    
    #  Lặp qua từng điểm thời gian (kỳ học)
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
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]
        # Tính toán Ma trận Nhầm lẫn
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        specificity = report['0']['recall']

        try:
            roc_auc = roc_auc_score(y_test,y_prob)
        except ValueError:
            # Xử lý trường hợp chỉ có 1 lớp trong tập test (hiếm nhưng có thể xảy ra)
            roc_auc=0.5

        log_l = log_loss(y_test,y_prob)
        print("\n--- KẾT QUẢ ĐÁNH GIÁ HIỆU SUẤT TOÀN DIỆN ---")
        
        # 5. Ma trận Nhầm lẫn
        print("\n[A] MA TRẬN NHẦM LẪN (Confusion Matrix):")
        print("---------------------------------------")
        print(f"                  | Predicted No (0) | Predicted Yes (1)")
        print(f"Actual No (0):    | {cm[0, 0]:<16} | {cm[0, 1]:<15} (FP)") 
        print(f"Actual Yes (1):   | {cm[1, 0]:<16} (FN) | {cm[1, 1]:<15} (TP)")
        print("---------------------------------------")

        # 6. Các Metrics Chính
        print("\n[B] CHỈ SỐ PHÂN LOẠI CƠ BẢN:")
        print(f"  Accuracy (Tổng quan):  {accuracy_score(y_test, y_pred):.4f}")
        print(f"  F1-Score (Lớp 1):      {report['1']['f1-score']:.4f}")
        print(f"  Precision (Lớp 1):     {report['1']['precision']:.4f}")
        print(f"  Recall (Lớp 1):        {report['1']['recall']:.4f}")
        print(f"  Specificity (Lớp 0):   {specificity:.4f}")
        
        # 7. Metrics Dựa trên Xác suất
        print("\n[C] CHỈ SỐ DỰA TRÊN XÁC SUẤT:")
        print(f"  ROC AUC Score:         {roc_auc:.4f}")
        print(f"  Log Loss:              {log_l:.4f}")

        feature_names_out = model_pipeline['preprocessor'].get_feature_names_out()
        importances = model_pipeline['classifier'].feature_importances_  
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names_out,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\n--- TOP 10 FEATURE IMPORTANCE ---")
        print(feature_importance_df.head(10).to_string(index=False))
        print("-----------------------------------")
        
        save_model(model_pipeline, sem)

if __name__ == '__main__':
    train_multi_stage_models()