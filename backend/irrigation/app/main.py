from fastapi import FastAPI
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware  # 👈 ADD THIS

from app.model_loader import load_artifacts
from app.preprocess import preprocess_input
from app.schema import IrrigationInput

# Load artifacts
model, scaler, encoder, feature_columns = load_artifacts()

app = FastAPI(title="Smart Irrigation API")

# 👇 ADD CORS HERE (RIGHT AFTER app = FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "🌾 Smart Irrigation API Running"}

@app.post("/predict")
def predict(data: IrrigationInput):

    input_df = pd.DataFrame([data.dict()])

    processed = preprocess_input(
        input_df,
        encoder,
        scaler,
        feature_columns
    )

    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "message": "Irrigation Needed" if prediction == 1 else "No Irrigation Needed"
    }