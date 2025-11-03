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


def calculate_gpa_means():
    # Đảm bảo đã chạy train_imputation_regressors.py trước khi dùng hàm này
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        return {'gpa_year1': 3.0, 'gpa_year2': 3.0, 'gpa_year3': 3.0, 'gpa_year4': 3.0} # Giá trị Fallback
        
    df.columns = df.columns.str.lower().str.strip()
    gpa_cols = [c for c in df.columns if c.startswith("gpa")]
    gpa_means = {}
    for col in gpa_cols:
        gpa_series = pd.to_numeric(df[col], errors="coerce")
        mean_val = gpa_series.mean()
        gpa_means[col] = np.round(mean_val, 2)
    return gpa_means

GPA_MEANS = calculate_gpa_means()

def Get_Major_List():
      df = pd.read_csv(DATA_FILE)
      return sorted(df["major"].dropna().unique())

def Get_Admission_type_List():
      df = pd.read_csv(DATA_FILE)
      return sorted(df["admission_type"].dropna().unique())