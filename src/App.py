from flask import Flask, request, jsonify
import joblib
from dotenv import load_dotenv
from pathlib import Path
import os

from src.Config import MODEL_PATH, FEATURES_PATH
from src.model_utils import Predict_student_status

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)
print("DEBUG ENV PORT =", os.getenv("PORT"))
print("DEBUG ENV HOST =", os.getenv("HOST"))
HOST  = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

app = Flask(__name__)

try:
      LOADED_MODEL = joblib.load(MODEL_PATH)
      LOADED_FEATURES = joblib.load(FEATURES_PATH)
      print("mô hình và cấu hình đã được tải")
except FileNotFoundError:
      print(f"Lỗi: Không tìm thấy file mô hình tại {MODEL_PATH}. Hãy chạy train.py trước")
      exit()

# Thiết lập ứng dụng Flask



@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint nhận dữ liệu JSON và trả về kết quả dự đoán."""
    
    # Nhận dữ liệu đầu vào từ client
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Vui lòng cung cấp dữ liệu JSON.'}), 400

    try:
        # Thực hiện dự đoán bằng hàm đóng gói
        result = Predict_student_status(
            new_data_dict=data, 
            model=LOADED_MODEL, 
            training_features=LOADED_FEATURES
        )
        
        # Trả về kết quả JSON
        return jsonify({
            'status': 'success',
            'result': result
        })

    except Exception as e:
        return jsonify({'error': f'Lỗi hệ thống hoặc định dạng dữ liệu: {str(e)}'}), 500

if __name__ == '__main__':
    print("Ứng dụng dự đoán bắt đầu chạy tại http://127.0.0.1:5000/predict")
    app.run(host=HOST, port=PORT)

