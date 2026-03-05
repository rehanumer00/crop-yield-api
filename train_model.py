import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Load the Excel file
df = pd.read_excel('crop-yield Pridiction Dataset.xlsx', sheet_name='crop-yield main')

# Look at the first few rows
print(df.head())

# Columns we need
feature_cols = ['N', 'P', 'K', 'Soil_pH', 'Temperature', 'Humidity',
                'Crop_Type', 'Irrigation_Type']
target = 'Crop_Yield_ton_per_hectare'

X = df[feature_cols].copy()
y = df[target]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, columns=['Crop_Type', 'Irrigation_Type'], drop_first=True)

# Save the list of column names for later use
feature_names = X_encoded.columns.tolist()
print("Encoded feature names:", feature_names)
print("Number of features:", len(feature_names))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# Create the model
model = XGBRegressor(
    n_estimators=200,        # number of trees
    max_depth=6,             # tree depth
    learning_rate=0.1,       # step size
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.3f} tons/ha")

joblib.dump(model, 'crop_yield_model.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("Model and feature names saved!")