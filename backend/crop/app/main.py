from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware   # 👈 ADD THIS

from app.schema import CropInput
from app.utils import predict_crop

app = FastAPI(
    title="Crop Recommendation API 🌾",
    version="1.0"
)

# 👇 ADD THIS BLOCK (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running 🚀"}

@app.post("/predict")
def predict(input_data: CropInput):
    data = input_data.dict()
    prediction = predict_crop(data)

    return {
        "recommended_crop": prediction
    }