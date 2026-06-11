import os
import tensorflow as tf
from keras.src.models.functional import Functional

# Base path calculation
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "potato_disease_model.keras")

print(f"🤖 Loading Potato Model from: {MODEL_PATH}")

try:
    # We pass 'Functional' in custom_objects to bridge version disparities
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={"Functional": Functional}
    )
    print("✅ Potato Model loaded successfully with custom deserialization!")
except Exception as e:
    print(f"⚠️ Standard load failed, trying alternate layer map...")
    # Secondary fallback mapping for older sequential/functional definitions
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={"Functional": tf.keras.Model}
    )
    print("✅ Potato Model loaded successfully via fallback mapper!")