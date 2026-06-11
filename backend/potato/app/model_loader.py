import os
import tensorflow as tf
from keras.src.models.functional import Functional

# 1. This goes up 3 levels from model_loader.py to reach /app (the root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 2. MATCH EXACT TREE: backend -> potato -> artifacts -> model
MODEL_PATH = os.path.join(BASE_DIR, "backend", "potato", "artifacts", "potato_disease_model.keras")

print(f"🤖 Absolute lookup path calculated: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"🚨 FOLDER TREE MISMATCH! Content of base dir: {os.listdir(BASE_DIR)}")
    raise FileNotFoundError(f"Could not find model file at {MODEL_PATH}")

try:
    print("🤖 Loading Potato Model with custom layer dictionary...")
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={"Functional": Functional}
    )
    print("✅ Potato Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Primary map failed, attempting fallback...")
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={"Functional": tf.keras.Model}
    )
    print("✅ Potato Model loaded successfully via fallback!")