import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "potato_disease_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)