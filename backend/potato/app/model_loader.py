import os
import tensorflow as tf
from keras.src.models.functional import Functional
from keras.src.saving import serialization_lib

# 1. Register 'Functional' globally in the Keras serialization engine
serialization_lib.register_keras_serializable(package="keras.src.models.functional")(Functional)

# 2. Match exact container paths
MODEL_PATH = "/app/backend/potato/artifacts/potato_disease_model.keras"

if not os.path.exists(MODEL_PATH):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    MODEL_PATH = os.path.join(BASE_DIR, "backend", "potato", "artifacts", "potato_disease_model.keras")

print(f"🤖 Loading Potato Model from verified location: {MODEL_PATH}")

try:
    # Pass a robust custom object mapping array to rebuild the sequential wrapper safely
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "Functional": Functional,
            "functional": Functional
        },
        compile=False # Prevents optimizer compilation structure errors
    )
    print("✅ Potato Model and sub-layers loaded flawlessly!")

except Exception as e:
    print(f"⚠️ Primary deserialization mapping failed. error: {str(e)}")
    print("🔄 Initializing deep structural bypass reconstruction...")
    
    # Ultimate fallback: Force map over the root Keras Model structure
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "Functional": tf.keras.Model,
            "Sequential": tf.keras.Sequential
        },
        compile=False
    )
    print("✅ Potato Model successfully salvaged via fallback definition maps!")