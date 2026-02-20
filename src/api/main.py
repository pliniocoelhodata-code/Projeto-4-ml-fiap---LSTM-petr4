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

# Diretório raiz do projeto (funciona no Render e local)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / "lstm_petr4.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

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
try:

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"SCALER_PATH: {SCALER_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler não encontrado em: {SCALER_PATH}")

    model = load_model(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    print("Modelo e scaler carregados com sucesso")

except Exception as e:

    print(f"ERRO ao carregar modelo/scaler: {e}")

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

    input_scaled = scaler.transform(input_array)

    input_scaled = input_scaled.reshape(1, LOOKBACK, 1)

    prediction_scaled = model.predict(input_scaled)

    prediction = scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).flatten()

    return {
        "prediction_30_days": prediction.tolist()
    }
