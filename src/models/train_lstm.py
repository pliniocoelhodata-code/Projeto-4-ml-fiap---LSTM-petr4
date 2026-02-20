import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import joblib

from src.data.preprocess import preprocess_pipeline

# ======================
# Configurações
# ======================
EPOCHS = 50
BATCH_SIZE = 32

WEIGHTS_PATH = "models/lstm_petr4.weights.h5"
MODEL_PATH = "models/lstm_petr4.keras"

# ======================
# Criar pasta models
# ======================
Path("models").mkdir(exist_ok=True)

# ======================
# Carregar dados
# ======================
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_pipeline()

# IMPORTANTE: ajuste o preprocess_pipeline para retornar o scaler também

# ======================
# Modelo
# ======================
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.3),
    LSTM(16),
    Dense(30)
])

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

    # checkpoint de pesos (opcional)
    ModelCheckpoint(
        WEIGHTS_PATH,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True
    ),

    # checkpoint do modelo completo (RECOMENDADO)
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False
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
# Garantir salvamento final
# ======================

model.save("models/lstm_petr4", save_format="tf")



# ======================
# Avaliação
# ======================
test_loss, test_mae = model.evaluate(X_test, y_test)

print(f"Test MAE: {test_mae:.4f}")
print(f"Test MSE: {test_loss:.4f}")
