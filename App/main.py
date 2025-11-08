from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize app
app = FastAPI(title="Funding Prediction API")

# Load model
model = joblib.load("best_xgb_pipeline.pkl")

# Input schema
class InputData(BaseModel):
    managing_agency_name: str
    us_sector_name: str
    fiscal_year: int
    # Add all other features your model expects, with correct data types

@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    # Make prediction
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}