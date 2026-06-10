import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def load_artifacts():

    model = joblib.load(
        ARTIFACTS_DIR / "irrigation_model.pkl"
    )

    scaler = joblib.load(
        ARTIFACTS_DIR / "irrigation_scaler.pkl"
    )

    encoder = joblib.load(
        ARTIFACTS_DIR / "irrigation_encoder.pkl"
    )

    feature_columns = joblib.load(
        ARTIFACTS_DIR / "feature_columns.pkl"
    )

    return model, scaler, encoder, feature_columns