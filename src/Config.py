
import os
from dotenv import load_dotenv
load_dotenv()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8501))

DATA_DIR = "data"
MODELS_DIR = "models"
DATA_FILE_PATH = os.path.join(DATA_DIR, "graduation_dataset_final1.csv")
MODEL_NAMES = {
    5: "model_sem5.pkl",  
    6: "model_sem6.pkl",  
    7: "model_sem7.pkl", 
    8: "model_sem8.pkl",  
}


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
TARGET_COLUMN = 'target'
 
FIXED_FEATURES = ['gender', 'major', 'admission_type', 'admission_score']
CATEGORICAL_FEATURES = ['gender', 'major', 'admission_type']


FEATURE_SETS = {
    5: FIXED_FEATURES + [f'gpa_sem{i}' for i in range(1, 5)] + ['credits_sem5','failed_sem5', 'warn_sem5'],
    6: FIXED_FEATURES + [f'gpa_sem{i}' for i in range(1, 6)] + ['credits_sem6','failed_sem6', 'warn_sem6'],
    7: FIXED_FEATURES + [f'gpa_sem{i}' for i in range(1, 7)] + ['credits_sem7', 'failed_sem7', 'warn_sem7'],
    8: FIXED_FEATURES + [f'gpa_sem{i}' for i in range(1, 8)] + ['credits_sem8', 'failed_sem8', 'warn_sem8'],
}

SEMESTER_POINTS = list(FEATURE_SETS.keys())