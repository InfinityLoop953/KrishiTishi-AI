import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "fertilizer_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "fertilizer_scaler.pkl"
CROP_ENCODER_PATH = ARTIFACTS_DIR / "fertilizer_crop_encoder.pkl"
TARGET_ENCODER_PATH = ARTIFACTS_DIR / "fertilizer_target_encoder.pkl"
POLY_PATH = ARTIFACTS_DIR / "fertilizer_poly.pkl"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "fertilizer_feature_names.pkl"

def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(CROP_ENCODER_PATH, "rb") as f:
        crop_encoder = pickle.load(f)

    with open(TARGET_ENCODER_PATH, "rb") as f:
        fertilizer_encoder = pickle.load(f)

    with open(POLY_PATH, "rb") as f:
        poly = pickle.load(f)

    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)

    return model, scaler, crop_encoder, fertilizer_encoder, poly, feature_names