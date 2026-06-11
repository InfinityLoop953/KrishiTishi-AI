import os
import tensorflow as tf
import keras

# 1. Use the proper, public Keras 3 API to register the structural class
@keras.saving.register_keras_serializable(package="Custom")
class FunctionalBypass(keras.Model):
    pass

# 2. Match exact container paths
MODEL_PATH = "/app/backend/potato/artifacts/potato_disease_model.keras"

if not os.path.exists(MODEL_PATH):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    MODEL_PATH = os.path.join(BASE_DIR, "backend", "potato", "artifacts", "potato_disease_model.keras")

print(f"🤖 Loading Potato Model from verified location: {MODEL_PATH}")

try:
    # Pass structural definitions safely via custom_objects
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "Functional": FunctionalBypass,
            "functional": FunctionalBypass
        },
        compile=False  # Skips training/optimizer compilation metrics
    )
    print("✅ Potato Model and sub-layers loaded flawlessly!")

except Exception as e:
    print(f"⚠️ Primary deserialization mapping failed. error: {str(e)}")
    print("🔄 Initializing native fallback reconstruction...")
    
    # Absolute generic fallback mapping
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "Functional": tf.keras.Model,
            "Sequential": tf.keras.Sequential
        },
        compile=False
    )
    print("✅ Potato Model successfully salvaged via native fallback maps!")