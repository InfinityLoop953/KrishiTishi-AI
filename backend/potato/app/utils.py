import io
import numpy as np
from PIL import Image
from backend.potato.app.model_loader import model

class_names = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

IMG_SIZE = 300

def predict_potato(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)

    idx = np.argmax(prediction)

    return {
        "prediction": class_names[idx],
        "confidence": round(float(np.max(prediction)) * 100, 2)
    }