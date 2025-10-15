# predict_dynamic.py

import pandas as pd
import joblib
import os

# ------------------------------------------
# Load Model & Metadata
# ------------------------------------------
MODEL_FILE = "medical_price_model.pkl"
ENCODERS_FILE = "label_encoders.pkl"
FEATURES_FILE = "feature_names.pkl"

# Check for files
for file in [MODEL_FILE, ENCODERS_FILE, FEATURES_FILE]:
    if not os.path.exists(file):
        print(f" Required file '{file}' not found. Please train the model first.")
        exit()

# Load objects
model = joblib.load(MODEL_FILE)
label_encoders = joblib.load(ENCODERS_FILE)
feature_names = joblib.load(FEATURES_FILE)

# ------------------------------------------
# Dynamic User Input
# ------------------------------------------
print("🔢 Please enter values for the following features:")
user_input = {}

for feature in feature_names:
    value = input(f"{feature}: ")
    user_input[feature] = value

input_df = pd.DataFrame([user_input])

# ------------------------------------------
# Preprocess Input
# ------------------------------------------
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        if input_df[col].values[0] not in le.classes_:
            if "Unknown" in le.classes_:
                input_df[col] = le.transform(["Unknown"])
            else:
                input_df[col] = 0  # fallback
        else:
            input_df[col] = le.transform(input_df[col])
    else:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

input_df.fillna(0, inplace=True)

# ------------------------------------------
# Predict
# ------------------------------------------
prediction = model.predict(input_df)[0]
print(f"\n💰 Predicted Treatment Cost: ₹{prediction:,.2f}")
