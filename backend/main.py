from fastapi import FastAPI
import pandas as pd
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from potato.app.utils import predict_potato

# =========================
# Crop Module
# =========================
from crop.app.schema import CropInput
from crop.app.utils import predict_crop

# =========================
# Fertilizer Module
# =========================
from fertilizer.app.schema import FertilizerInput
from fertilizer.app.model_loader import (
    load_artifacts as load_fertilizer_artifacts
)
from fertilizer.app.preprocess import (
    preprocess_input as fertilizer_preprocess
)

# =========================
# Irrigation Module
# =========================
from irrigation.app.schema import IrrigationInput
from irrigation.app.model_loader import (
    load_artifacts as load_irrigation_artifacts
)
from irrigation.app.preprocess import (
    preprocess_input as irrigation_preprocess
)


# =========================
# Load Models Once
# =========================

# Crop
# (loaded internally by predict_crop)

# Fertilizer
(
    fert_model,
    fert_scaler,
    fert_crop_encoder,
    fert_target_encoder,
    fert_poly,
    fert_feature_names
) = load_fertilizer_artifacts()

# Irrigation
(
    irr_model,
    irr_scaler,
    irr_encoder,
    irr_feature_columns
) = load_irrigation_artifacts()



app = FastAPI(
    title="KrishiTrishi AI Backend",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "KrishiTrishi Backend Running"
    }

@app.post("/api/crop/predict")
def crop_predict(data: CropInput):

    prediction = predict_crop(data.model_dump())

    return {
        "recommended_crop": prediction
    }
    
@app.post("/api/fertilizer/predict")
def fertilizer_predict(data: FertilizerInput):

    input_df = pd.DataFrame([data.model_dump()])

    processed = fertilizer_preprocess(
        input_df,
        fert_scaler,
        fert_crop_encoder,
        fert_poly,
        fert_feature_names
    )

    processed = processed.astype("float32")

    prediction = fert_model.predict(
        processed.values
    )[0]

    fertilizer_name = (
        fert_target_encoder
        .inverse_transform([prediction])[0]
    )

    return {
        "recommended_fertilizer": fertilizer_name
    }
    
    
    
    
@app.post("/api/irrigation/predict")
def irrigation_predict(data: IrrigationInput):

    input_df = pd.DataFrame([data.model_dump()])

    processed = irrigation_preprocess(
        input_df,
        irr_encoder,
        irr_scaler,
        irr_feature_columns
    )

    prediction = irr_model.predict(processed)[0]

    probability = (
        irr_model.predict_proba(processed)[0][1]
    )

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "message": (
            "Irrigation Needed"
            if prediction == 1
            else "No Irrigation Needed"
        )
    }
    
    
    
    
    
@app.post("/api/potato/predict")
async def potato_predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return predict_potato(image_bytes)