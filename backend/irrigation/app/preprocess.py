import pandas as pd
import numpy as np

def feature_engineering(df):

    df = df.copy()

    df['soil_moisture_deficit'] = np.maximum(0, 50 - df['soil_moisture'])

    df['temp_stress'] = np.maximum(
        0,
        np.abs(df['temperature'] - 25) - 5
    )

    df['vapor_pressure_deficit'] = (
        df['temperature'] * (1 - df['humidity']/100)
    )

    df['water_stress_index'] = (
        df['soil_moisture_deficit'] * 0.6 +
        df['temp_stress'] * 0.3 +
        df['vapor_pressure_deficit'] * 0.1
    )

    df['temp_humidity_interaction'] = (
        df['temperature'] * df['humidity']
    )

    df['rainfall_effect'] = (
        df['rainfall'] > 5
    ).astype(int)

    return df


def preprocess_input(
        input_df,
        encoder,
        scaler,
        feature_columns
):

    input_df = feature_engineering(input_df)

    categorical_cols = [
        "location",
        "crop_type",
        "season"
    ]

    encoded = encoder.transform(
        input_df[categorical_cols]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    input_df = pd.concat(
        [input_df, encoded_df],
        axis=1
    )

    input_df.drop(
        columns=categorical_cols,
        inplace=True
    )

    # Ensure same column order
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_scaled = scaler.transform(input_df)

    return input_scaled