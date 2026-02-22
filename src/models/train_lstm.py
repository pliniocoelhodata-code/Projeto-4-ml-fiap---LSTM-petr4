import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import json
from datetime import datetime

from src.data.preprocess import preprocess_pipeline

# ======================
# Configurações
# ======================
EPOCHS = 50
BATCH_SIZE = 32

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

WEIGHTS_PATH = MODEL_DIR / "model.weights.h5"
METADATA_PATH = MODEL_DIR / "metadata.json"

# ======================
# Carregar dados
# ======================
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_pipeline()

# ======================
# Construção do modelo
# ======================
def build_model():
    model = Sequential([
        tf.keras.layers.Input(shape=(60, 1)),
        LSTM(32, return_sequences=True),
        Dropout(0.3),
        LSTM(16),
        Dense(30)
    ])
    return model

model = build_model()

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ======================
# Callbacks
# ======================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        str(WEIGHTS_PATH),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True
    )
]

# ======================
# Treinamento
# ======================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ======================
# Avaliação
# ======================
y_pred = model.predict(X_test)

y_true_1 = y_test[:, 0]
y_pred_1 = y_pred[:, 0]

mae = np.mean(np.abs(y_true_1 - y_pred_1))
rmse = np.sqrt(np.mean((y_true_1 - y_pred_1) ** 2))
mape = np.mean(np.abs((y_true_1 - y_pred_1) / (y_true_1 + 1e-8))) * 100

print(f"MAE (t+1): {mae:.4f}")
print(f"RMSE (t+1): {rmse:.4f}")
print(f"MAPE (t+1): {mape:.2f}%")

# ======================
# Baseline simples
# ======================
baseline_pred = X_test[:, -1, 0]
baseline_mae = np.mean(np.abs(y_true_1 - baseline_pred))

print(f"Baseline MAE (último valor): {baseline_mae:.4f}")

# ======================
# Salvar metadata
# ======================
metadata = {
    "model_type": "LSTM",
    "lookback": 60,
    "horizon": 30,
    "date_trained": datetime.now().isoformat(),
    "metrics": {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "baseline_mae": float(baseline_mae)
    }
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("Metadata salva com sucesso.")