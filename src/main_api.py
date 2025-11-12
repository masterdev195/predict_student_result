# phần backend 

# src/main_api.py (SERVER BACKEND)

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import sys
import os

# Thiết lập PATH để import từ các file khác trong src/
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_utils import load_model, predict_graduation
from Config import SEMESTER_POINTS, API_HOST, API_PORT

# --- Khởi tạo và Tải Mô hình (Chỉ 1 LẦN) ---
app = FastAPI(title="Graduation Prediction API")

# Mô hình được tải khi server khởi động, tránh xung đột và chậm trễ
MODELS = {s: load_model(s) for s in SEMESTER_POINTS}
print(f"Backend: Đã tải {len(MODELS)} mô hình thành công.")


# Định nghĩa cấu trúc dữ liệu đầu vào
class PredictionRequest(BaseModel):
    semester_point: int
    input_data: Dict[str, Any]

# --- Endpoint Dự đoán ---
@app.post("/predict_graduation")
async def predict_endpoint(request: PredictionRequest):
    """Xử lý yêu cầu dự đoán bằng cách gọi logic từ model_utils."""
    
    # Gọi hàm logic dự đoán
    result_class, confidence, error_msg = predict_graduation(
        data=request.input_data, 
        semester_point=request.semester_point, 
        models=MODELS
    )

    if error_msg:
        return {"status": "error", "message": error_msg}
    
    return {
        "status": "success",
        "prediction_class": result_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    # Chạy server Uvicorn với host và port từ Config (file .env)
    uvicorn.run("main_api:app", host=API_HOST, port=API_PORT, reload=True)