import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_utils import load_model, predict_graduation
from Config import SEMESTER_POINTS, API_HOST, API_PORT

app = FastAPI(title="Graduation Prediction API")
MODELS = {s: load_model(s) for s in SEMESTER_POINTS}
print(f"Backend: Đã tải {len(MODELS)} mô hình thành công.")

class PredictionRequest(BaseModel):
    semester_point: int
    input_data: Dict[str, Any]

@app.post("/predict_graduation")
async def predict_endpoint(request: PredictionRequest):
    
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
    uvicorn.run("main_api:app", host=API_HOST, port=API_PORT, reload=True)