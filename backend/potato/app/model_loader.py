import os
import tensorflow as tf
import tf_keras as legacy_keras  # Import the compatibility engine

# Set the path to your model file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "potatoes.h5") # or .keras, whatever your file extension is

# FIX: Use the legacy engine to open your older model structure seamlessly
print("🤖 Loading Potato Model via legacy deserializer...")
model = legacy_keras.models.load_model(MODEL_PATH)
print("✅ Potato Model loaded successfully!")