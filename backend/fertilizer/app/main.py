from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 👈 ADD THIS
import pandas as pd

from app.schema import FertilizerInput
from app.model_loader import load_artifacts
from app.preprocess import preprocess_input

app = FastAPI(title="Fertilizer Recommendation API")

# ✅ CORS FIX (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load everything once at startup
model, scaler, crop_encoder, fertilizer_encoder, poly, feature_names = load_artifacts()


@app.get("/")
def home():
    return {"message": "Fertilizer Recommendation API is running 🚀"}


@app.post("/predict")
def predict(data: FertilizerInput):

    input_df = pd.DataFrame([data.dict()])

    processed = preprocess_input(
        input_df,
        scaler,
        crop_encoder,
        poly,
        feature_names
    )

    processed = processed.astype("float32") 

    prediction = model.predict(processed.values)[0]

    fertilizer_name = fertilizer_encoder.inverse_transform([prediction])[0]

    return {
        "recommended_fertilizer": fertilizer_name
    }