from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# === Initialize app ===
app = FastAPI(
    title="Funding Prediction API",
    description="API that predicts U.S. foreign aid funding amounts using an optimized XGBoost pipeline.",
    version="1.0"
)

# === Load trained model ===
model = joblib.load("best_xgb_pipeline.pkl")

# === Define input schema ===
class FundingInput(BaseModel):
    managing_agency_name: str
    us_sector_name: str
    fiscal_year: int
    implementing_partner_name: str | None = None
    activity_title: str | None = None
    us_category_name: str | None = None
    # Add or remove fields based on your original training dataset columns

# === Define prediction endpoint ===
@app.post("/predict")
def predict_funding(data: FundingInput):
    """
    Predicts funding amount given input features.
    """
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {
        "predicted_constant_dollar_amount": float(prediction[0])
    }

# === Root endpoint ===
@app.get("/")
def home():
    return {"message": "Welcome to the Funding Prediction API! Visit /docs for the interactive Swagger UI."}