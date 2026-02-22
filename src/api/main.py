from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path
import time
import logging

# ======================
# Logging estruturado
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("petr4-api")

# ======================
# Configurações
# ======================
LOOKBACK = 60
HORIZON = 30

BASE_DIR = Path(__file__).resolve().parent.parent.parent
WEIGHTS_PATH = BASE_DIR / "models" / "model.weights.h5"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# ======================
# Inicialização FastAPI
# ======================
app = FastAPI(
    title="PETR4 LSTM Forecast API",
    version="1.0.0"
)

model = None
scaler = None

# ======================
# Construção do modelo
# ======================
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, 1)),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(HORIZON)
    ])
    return model

# ======================
# Startup
# ======================
@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        model = build_model()
        model.load_weights(WEIGHTS_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Modelo (weights) e scaler carregados com sucesso.")
    except Exception as e:
        logger.error(f"Erro no startup: {e}")
        raise e

# ======================
# Schema
# ======================
class PredictionRequest(BaseModel):
    last_60_days: List[float] = Field(
        ...,
        min_items=60,
        max_items=60,
        description="Lista com exatamente 60 preços de fechamento"
    )

# ======================
# Health check
# ======================
@app.get("/")
def health_check():
    return {"status": "ok"}

# ======================
# Endpoint versionado
# ======================
@app.post("/v1/predict")
def predict(request: PredictionRequest):

    global model, scaler

    start_time = time.time()

    try:

        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Modelo não carregado")

        input_array = np.array(request.last_60_days).reshape(LOOKBACK, 1)
        input_scaled = scaler.transform(input_array)
        input_scaled = input_scaled.reshape(1, LOOKBACK, 1)

        prediction_scaled = model.predict(input_scaled, verbose=0)

        latency = time.time() - start_time

        prediction = scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        ).flatten()

        logger.info({
            "event": "prediction",
            "latency_seconds": round(latency, 4),
            "input_size": len(request.last_60_days)
        })

        return {
            "prediction_30_days": prediction.tolist(),
            "latency_seconds": round(latency, 4)
        }

    except Exception as e:

        logger.error({
            "event": "prediction_error",
            "error": str(e)
        })

        raise HTTPException(status_code=500, detail="Erro na inferência")