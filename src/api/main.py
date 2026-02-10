from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path

# ======================
# Configurações
# ======================
LOOKBACK = 60
HORIZON = 30

WEIGHTS_PATH = Path("models/lstm_petr4.weights.h5")
SCALER_PATH = Path("models/scaler.pkl")

# ======================
# Inicialização
# ======================
app = FastAPI(
    title="PETR4 LSTM Forecast API",
    description="API para previsão de preços da ação PETR4.SA usando LSTM",
    version="1.0.0"
)

# ======================
# Carregar modelo e scaler
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

try:
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Erro ao carregar modelo ou scaler: {e}")

# ======================
# Schema de entrada
# ======================
class PredictionRequest(BaseModel):
    last_60_days: list[float]

# ======================
# Health check
# ======================
@app.get("/")
def health_check():
    return {"status": "ok", "message": "API PETR4 LSTM ativa"}

# ======================
# Endpoint de previsão
# ======================
@app.post("/predict")
def predict(request: PredictionRequest):

    if len(request.last_60_days) != LOOKBACK:
        raise HTTPException(
            status_code=400,
            detail=f"É necessário informar exatamente {LOOKBACK} valores."
        )

    try:
        input_array = np.array(request.last_60_days).reshape(LOOKBACK, 1)
    except Exception:
        raise HTTPException(status_code=400, detail="Valores inválidos.")

    # Normalizar
    input_scaled = scaler.transform(input_array)

    # Ajustar shape para LSTM
    input_scaled = input_scaled.reshape(1, LOOKBACK, 1)

    # Previsão
    prediction_scaled = model.predict(input_scaled)

    # Desnormalizar
    prediction = scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).flatten()

    return {
        "prediction_30_days": prediction.tolist()
    }
