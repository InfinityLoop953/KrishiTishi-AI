from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Absolute Imports
from backend.potato.app.utils import predict_potato
from backend.crop.app.schema import CropInput
from backend.crop.app.utils import predict_crop
from backend.fertilizer.app.schema import FertilizerInput
from backend.fertilizer.app.model_loader import load_artifacts as load_fertilizer_artifacts
from backend.fertilizer.app.preprocess import preprocess_input as fertilizer_preprocess
from backend.irrigation.app.schema import IrrigationInput
from backend.irrigation.app.model_loader import load_artifacts as load_irrigation_artifacts
from backend.irrigation.app.preprocess import preprocess_input as irrigation_preprocess



# for chatbot
from backend.chatbot.schema import ChatRequest
from backend.chatbot.gemini_service import ask_gemini


# Global state dictionary to store models safely
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block executes RIGHT when the server starts
    print("🤖 Loading Machine Learning artifacts safely...")
    
    # Load Fertilizer artifacts
    fert_artifacts = load_fertilizer_artifacts()
    ml_models["fertilizer"] = {
        "model": fert_artifacts[0],
        "scaler": fert_artifacts[1],
        "crop_encoder": fert_artifacts[2],
        "target_encoder": fert_artifacts[3],
        "poly": fert_artifacts[4],
        "feature_names": fert_artifacts[5]
    }
    
    # Load Irrigation artifacts
    irr_artifacts = load_irrigation_artifacts()
    ml_models["irrigation"] = {
        "model": irr_artifacts[0],
        "scaler": irr_artifacts[1],
        "encoder": irr_artifacts[2],
        "feature_columns": irr_artifacts[3]
    }
    
    print("✅ All models loaded. Application is ready!")
    yield
    # Clean up on shutdown if necessary
    ml_models.clear()

app = FastAPI(
    title="KrishiTrishi AI Backend",
    version="1.0",
    lifespan=lifespan
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
    return {"message": "KrishiTrishi Backend Running Successfully"}

@app.post("/api/crop/predict")
def crop_predict(data: CropInput):
    prediction = predict_crop(data.model_dump())
    return {"recommended_crop": prediction}
    
@app.post("/api/fertilizer/predict")
def fertilizer_predict(data: FertilizerInput):
    # Fetch models from the lifespan state
    fert = ml_models["fertilizer"]
    
    input_df = pd.DataFrame([data.model_dump()])

    processed = fertilizer_preprocess(
        input_df,
        fert["scaler"],
        fert["crop_encoder"],
        fert["poly"],
        fert["feature_names"]
    )
    processed = processed.astype("float32")

    prediction = fert["model"].predict(processed)[0]
    fertilizer_name = fert["target_encoder"].inverse_transform([prediction])[0]

    return {"recommended_fertilizer": fertilizer_name}
    
@app.post("/api/irrigation/predict")
def irrigation_predict(data: IrrigationInput):
    # Fetch models from the lifespan state
    irr = ml_models["irrigation"]

    input_df = pd.DataFrame([data.model_dump()])

    processed = irrigation_preprocess(
        input_df,
        irr["encoder"],
        irr["scaler"],
        irr["feature_columns"]
    )

    prediction = irr["model"].predict(processed)[0]
    probability = irr["model"].predict_proba(processed)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "message": "Irrigation Needed" if prediction == 1 else "No Irrigation Needed"
    }

@app.post("/api/potato/predict")
async def potato_predict_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return predict_potato(image_bytes)



# chatbot
@app.post("/api/chat")
def chatbot_endpoint(data: ChatRequest):
    try:
        # Pass the message to your service file
        reply_text = ask_gemini(data.message)
        
        # Return it in the JSON format your gemini.html expects ({"reply": ...})
        return {"reply": reply_text}
        
    except Exception as e:
        return {"reply": f"দুঃখিত, একটি অভ্যন্তরীণ সমস্যা হয়েছে। Error: {str(e)}"}