import pandas as pd
import joblib

def predict_yield(N, P, K, pH, temp, humidity, crop_type, irrigation_type, area_acres):
    """
    Returns:
        yield_per_ha (float): predicted yield in tons/hectare
        total_yield (float): predicted total yield in tons
    """
    # Load model and feature names (do this once when the app starts, not per request)
    model = joblib.load('crop_yield_model.pkl')
    feature_names = joblib.load('feature_names.pkl')

    # Create a DataFrame with one row
    data = {
        'N': N,
        'P': P,
        'K': K,
        'Soil_pH': pH,
        'Temperature': temp,
        'Humidity': humidity,
        'Crop_Type': crop_type,
        'Irrigation_Type': irrigation_type
    }
    df = pd.DataFrame([data])

    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=['Crop_Type', 'Irrigation_Type'], drop_first=True)

    # Add any missing columns (should be 0)
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ensure correct column order
    df_encoded = df_encoded[feature_names]

    # Predict
    yield_per_ha = model.predict(df_encoded)[0]
    # Convert acres to hectares (1 acre = 0.404686 ha)
    total_yield = yield_per_ha * area_acres * 0.404686

    return yield_per_ha, total_yield