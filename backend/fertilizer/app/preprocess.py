import pandas as pd

def preprocess_input(
    df,
    scaler,
    crop_encoder,
    poly,
    feature_names,
    fit=False
):

    df = df.copy()

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    df["NPK_Sum"] = df["Nitrogen"] + df["Phosphorus"] + df["Potassium"]
    df["NPK_Mean"] = df["NPK_Sum"] / 3
    df["NPK_Product"] = df["Nitrogen"] * df["Phosphorus"] * df["Potassium"]

    df["N_to_P"] = df["Nitrogen"] / (df["Phosphorus"] + 1)
    df["N_to_K"] = df["Nitrogen"] / (df["Potassium"] + 1)
    df["P_to_K"] = df["Phosphorus"] / (df["Potassium"] + 1)

    df["NPK_Ratio_Index"] = df["Nitrogen"] / (df["Phosphorus"] + df["Potassium"] + 1)

    df["pH_Distance_From_7"] = abs(df["pH"] - 7)
    df["pH_Squared"] = df["pH"] ** 2

    df["Temp_pH"] = df["Temperature"] * df["pH"]
    df["Temp_NPK"] = df["Temperature"] * df["NPK_Sum"]

    df["Low_N"] = (df["Nitrogen"] < 40).astype(int)
    df["Low_P"] = (df["Phosphorus"] < 40).astype(int)
    df["Low_K"] = (df["Potassium"] < 40).astype(int)

    # -----------------------------
    # Polynomial Features
    # -----------------------------

    poly_cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Temperature"]

    poly_matrix = poly.transform(df[poly_cols])
    poly_df = pd.DataFrame(poly_matrix, columns=poly.get_feature_names_out(poly_cols))

    df = pd.concat([df, poly_df], axis=1)

    # -----------------------------
    # Encode Crop
    # -----------------------------

    df["Crop_Encoded"] = crop_encoder.transform(df["Crop"])
    df.drop(columns=["Crop"], inplace=True)

    # -----------------------------
    # Scaling
    # -----------------------------

    df_scaled = scaler.transform(df)

    df_scaled = pd.DataFrame(df_scaled, columns=feature_names)
    

    return df_scaled