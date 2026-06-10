import pickle
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
    
    
    # Load artifacts



with open(ARTIFACTS_DIR / "crop_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(ARTIFACTS_DIR / "crop_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(ARTIFACTS_DIR / "crop_label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open(ARTIFACTS_DIR / "crop_target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

with open(ARTIFACTS_DIR / "crop_features.pkl", "rb") as f:
    selected_features = pickle.load(f)


# ===============================
# PREPROCESS FUNCTION
# ===============================

def preprocess_data(df):

    def detect_season(temp, humidity):

        if temp < 22 and humidity < 75:
            return 'Winter'
        elif 22 <= temp < 28 and humidity < 80:
            return 'Spring'
        elif temp >= 28 and humidity >= 80:
            return 'Rainy'
        elif temp >= 30 and humidity < 75:
            return 'Summer'
        else:
            return 'Autumn'

    # Feature Engineering

    df['Season'] = df.apply(
        lambda row: detect_season(
            row['Temperature'],
            row['Humidity']
        ),
        axis=1
    )

    season_cycle = {
        'Winter': 'Spring',
        'Spring': 'Summer',
        'Summer': 'Rainy',
        'Rainy': 'Autumn',
        'Autumn': 'Winter'
    }

    df['Next_Season'] = df['Season'].map(season_cycle)

    df['Soil_SoilMoisture_Ratio'] = df['Soil'] / (df['Soil_Moisture'] + 0.001)

    df['Temp_Humidity_Interaction'] = (
        df['Temperature'] * df['Humidity'] / 100
    )

    df['Moisture_Humidity_Diff'] = (
        df['Soil_Moisture'] - df['Humidity']
    )

    # Encoding

    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    # Feature selection

    df = df[selected_features]

    # Scaling

    X_scaled = scaler.transform(df)

    return X_scaled


# ===============================
# PREDICTION FUNCTION
# ===============================

def predict_crop(data_dict):

    df = pd.DataFrame([data_dict])

    processed = preprocess_data(df)
    
    pred = model.predict(processed)

    crop = target_encoder.inverse_transform(pred)

    return crop[0]