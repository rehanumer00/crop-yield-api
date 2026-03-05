from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and feature names once when the server starts
model = joblib.load('crop_yield_model.pkl')
feature_names = joblib.load('feature_names.pkl')

app = FastAPI(title="Crop Yield Prediction API")

class PredictionRequest(BaseModel):
    area_acres: float
    crop_type: str
    irrigation_type: str
    N: float
    P: float
    K: float
    pH: float
    temperature: float
    humidity: float

def get_suggestions(N, P, K, pH, irrigation_type, crop_type, predicted_yield):
    suggestions = []
    if N < 50:
        suggestions.append("🔹 Nitrogen is low – consider applying nitrogen fertilizer (e.g., urea).")
    if P < 30:
        suggestions.append("🔹 Phosphorus is low – consider applying phosphate fertilizer (e.g., DAP).")
    if K < 40:
        suggestions.append("🔹 Potassium is low – consider applying potash.")
    if pH < 5.5:
        suggestions.append("🔹 Soil pH is too acidic – liming may improve yields.")
    elif pH > 7.5:
        suggestions.append("🔹 Soil pH is too alkaline – adding sulfur or organic matter can help.")
    if predicted_yield < 5:
        suggestions.append("🔹 Predicted yield is low – review your crop management practices.")
    if irrigation_type.lower() == 'rainfed' and predicted_yield < 3:
        suggestions.append("🔹 Consider using irrigation to reduce water stress.")
    return suggestions

class PredictionResponse(BaseModel):
    yield_per_hectare: float
    total_yield_tons: float
    suggestions: list[str]

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    # (Optional) Validate crop and irrigation types
    valid_crops = ['Maize', 'Potato', 'Rice', 'Sugarcane', 'Wheat', 'Cotton']
    valid_irrigation = ['Canal', 'Rainfed', 'Sprinkler', 'Drip']
    if data.crop_type not in valid_crops:
        return {"error": f"Crop type must be one of {valid_crops}"}
    if data.irrigation_type not in valid_irrigation:
        return {"error": f"Irrigation type must be one of {valid_irrigation}"}

    # Prepare input for prediction
    input_dict = {
        'N': data.N,
        'P': data.P,
        'K': data.K,
        'Soil_pH': data.pH,
        'Temperature': data.temperature,
        'Humidity': data.humidity,
        'Crop_Type': data.crop_type,
        'Irrigation_Type': data.irrigation_type
    }
    df_input = pd.DataFrame([input_dict])
    df_encoded = pd.get_dummies(df_input, columns=['Crop_Type', 'Irrigation_Type'], drop_first=True)

    # Align columns with training features
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_names]

    # Predict
    yield_per_ha = model.predict(df_encoded)[0]
    total_yield = yield_per_ha * data.area_acres * 0.404686

    # Generate suggestions
    suggestions = get_suggestions(
        N=data.N, P=data.P, K=data.K, pH=data.pH,
        irrigation_type=data.irrigation_type,
        crop_type=data.crop_type,
        predicted_yield=yield_per_ha
    )

    return PredictionResponse(
        yield_per_hectare=round(yield_per_ha, 2),
        total_yield_tons=round(total_yield, 2),
        suggestions=suggestions
    )

@app.get("/")
async def root():
    return {"message": "Crop Yield Prediction API is running. Send POST requests to /predict"}