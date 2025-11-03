import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# cau hinh

DATA_FILE = os.path.join(BASE_DIR,'data', 'graduated_students_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', "rf_graduate_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR,'models', 'training_features.pkl')

REG_MODEL_GPA2_PATH = os.path.join(BASE_DIR, 'models', 'reg_gpa2_model.pkl')
REG_MODEL_GPA3_PATH = os.path.join(BASE_DIR, 'models', 'reg_gpa3_model.pkl')
REG_MODEL_GPA4_PATH = os.path.join(BASE_DIR, 'models', 'reg_gpa4_model.pkl')

# cau hinh ma hoa
FINANCIAL_ORDER = {'Difficult':0 , 'Average':1, 'Stable':2}
NOMINAL_COLS = ['gender', 'major', 'admission_type']
COLS_TO_DROP = ['student_id', 'graduate_year']

NUMERIC_COLS_FOR_MEANS = [
    'gpa_year1', 'gpa_year2', 'gpa_year3', 'gpa_year4',
    'total_cumulative_gpa', 'total_credits_required', 'failed_courses',
    'academic_warnings', 'attendance_rate', 'assignment_submission_rate',
    'extra_activities', 'achieved_scholarship'
]

def calculate_all_means():
    # Đảm bảo đã chạy train_imputation_regressors.py trước khi dùng hàm này
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        fallback_means = {
            'gpa_year1': 2.5, 'gpa_year2': 2.5, 'gpa_year3': 2.5, 'gpa_year4': 2.5,
            'total_cumulative_gpa': 2.5,
            'total_credits_required': 130.0,
            'failed_courses': 0.0, # An toàn khi mặc định là 0 nếu không có data
            'academic_warnings': 0.0,
            'attendance_rate': 0.85, # Giá trị trung bình an toàn
            'assignment_submission_rate': 0.85, # Giá trị trung bình an toàn
            'extra_activities': 1.0,
            'achieved_scholarship': 0.0
        }
        return fallback_means
        
    df.columns = df.columns.str.lower().str.strip()
    means = {}
    for col in NUMERIC_COLS_FOR_MEANS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # THAY ĐỔI: Tính trung bình, dùng giá trị gần nhất thay vì 0.0 cho các cột số.
            means[col] = df[col].mean()
    return means

ALL_NUMERIC_MEANS = calculate_all_means()

def Get_Major_List():
      df = pd.read_csv(DATA_FILE)
      return sorted(df["major"].dropna().unique())

def Get_Admission_type_List():
      df = pd.read_csv(DATA_FILE)
      return sorted(df["admission_type"].dropna().unique())