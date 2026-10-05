from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
)

from src.features.inference import build_prediction_features


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "artifacts"
    / "demand_forecasting_model.joblib"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "sales_data.csv"
)

app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "API for predicting next-day retail product demand."
    ),
    version="1.0.0"
)

model = joblib.load(MODEL_PATH)

historical_data = pd.read_csv(DATA_PATH)


@app.get("/")
def root():
    return {
        "message": "Retail Demand Forecasting API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    try:

        features, prediction_date = (
            build_prediction_features(
                historical_data,
                request.store_id,
                request.product_id,
            )
        )

        prediction = model.predict(features)[0]

        return PredictionResponse(
            store_id=request.store_id,
            product_id=request.product_id,
            prediction_date=prediction_date.strftime(
                "%Y-%m-%d"
            ),
            predicted_demand=float(prediction),
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
