from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

# ======================
# Configurações
# ======================
LOOKBACK = 60
HORIZON = 30

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / "lstm_petr4"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# ======================
# Inicialização FastAPI
# ======================
app = FastAPI(
    title="PETR4 LSTM Forecast API",
    version="1.0.0"
)

# ======================
# Variáveis globais
# ======================
model = None
scaler = None

# ======================
# Startup event
# ======================
@app.on_event("startup")
def load_artifacts():

    global model, scaler

    try:

        print(f"BASE_DIR: {BASE_DIR}")
        print(f"MODEL_PATH: {MODEL_PATH}")
        print(f"SCALER_PATH: {SCALER_PATH}")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler não encontrado: {SCALER_PATH}")

        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        print("Modelo e scaler carregados com sucesso")

    except Exception as e:

        print(f"ERRO CRÍTICO NO STARTUP: {e}")
        raise e

# ======================
# Schema
# ======================
class PredictionRequest(BaseModel):
    last_60_days: list[float]

# ======================
# Health check
# ======================
@app.get("/")
def health_check():
    return {"status": "ok"}

# ======================
# Predict endpoint
# ======================
@app.post("/predict")
def predict(request: PredictionRequest):

    global model, scaler

    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    if len(request.last_60_days) != LOOKBACK:
        raise HTTPException(
            status_code=400,
            detail=f"Informe exatamente {LOOKBACK} valores"
        )

    input_array = np.array(request.last_60_days).reshape(LOOKBACK, 1)

    input_scaled = scaler.transform(input_array)
    input_scaled = input_scaled.reshape(1, LOOKBACK, 1)

    prediction_scaled = model.predict(input_scaled)

    prediction = scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).flatten()

    return {"prediction_30_days": prediction.tolist()}