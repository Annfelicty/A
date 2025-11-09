# api.py - Your NEW production script

# 1. Import necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np # Make sure numpy is included

# 2. Initialize the FastAPI app
app = FastAPI(
    title="Funding Prediction API",
    description="An API that predicts U.S. foreign aid funding amounts using an optimized XGBoost pipeline.",
    version="1.0"
)

# 3. Load the trained model pipeline
#    This pipeline includes the preprocessor and the XGBoost model.
try:
    model_pipeline = joblib.load('best_xgb_pipeline.pkl')
    print("Model pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading model pipeline: {e}")
    model_pipeline = None # Set to None if loading fails

# 4. Define the input data model using Pydantic
#    These fields MUST match the columns your model was trained on,
#    BEFORE feature engineering.
class FundingInput(BaseModel):
    fiscal_year: int
    is_refund: int
    managing_agency_name: str
    funding_agency_name: str
    sector: str
    # Add any other columns that were in your 'X' DataFrame before training
    # For example, if you had 'transaction_type_name', you would add:
    # transaction_type_name: str
    
    # Example of a full Pydantic model based on your notebook
    # You may need to adjust this!
    lag_1: float = 0
    lag_2: float = 0
    rolling_mean_3yr: float = 0
    rolling_std_3yr: float = 0
    funding_growth_rate: float = 0


# 5. Define the prediction endpoint
@app.post("/predict")
def predict_funding(data: FundingInput):
    """
    Takes funding data as input and returns a predicted funding amount.
    """
    if model_pipeline is None:
        return {"error": "Model not loaded. Please check server logs."}

    # Convert the input data into a pandas DataFrame
    # The structure must match the training data EXACTLY.
    input_df = pd.DataFrame([data.dict()])
    
    # The pipeline will handle preprocessing (OneHotEncoding, etc.)
    # and then make a prediction.
    log_prediction = model_pipeline.predict(input_df)
    
    # The model predicts the log-transformed value, so we must reverse it
    # using np.expm1() as you did in your notebook.
    predicted_amount = np.expm1(log_prediction)[0]
    
    return {
        "predicted_constant_dollar_amount": float(predicted_amount)
    }

# 6. Define a root endpoint for health checks
@app.get("/")
def home():
    return {"message": "Welcome to the Funding Prediction API. The model is ready for predictions."}


