from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model("artifacts/potato_disease_model.keras")
print(model.input_shape)
print(model.output_shape)
# Classes
class_names = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

IMG_SIZE = 300


def preprocess_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


@app.get("/")
def home():

    return {
        "message": "Potato Disease Prediction API Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(np.max(prediction)) * 100

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    }