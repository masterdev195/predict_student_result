import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# cau hinh

DATA_FILE = os.path.join(BASE_DIR,'data', 'graduated_students_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', "rf_graduate_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR,'models', 'training_features.pkl')

# cau hinh ma hoa
FINANCIAL_ORDER = {'Difficult':0 , 'Average':1, 'Stable':2}
NOMINAL_COLS = ['gender', 'major', 'admission_type']
COLS_TO_DROP = ['student_id', 'graduate_year']

def Get_Major_List():
      df = pd.read_csv(DATA_FILE)
      return sorted(df["major"].dropna().unique())

def Get_Admission_type_List():
      df = pd.read_csv(DATA_FILE)
      return sorted(df["admission_type"].dropna().unique())