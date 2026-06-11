from pathlib import Path
import tensorflow as tf

# Load the potato disease model from the artifacts directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "potato_disease_model.keras"

print(f"Loading potato model from {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Potato model loaded")