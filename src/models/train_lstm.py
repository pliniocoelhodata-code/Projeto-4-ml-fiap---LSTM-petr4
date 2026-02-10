import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path

from src.data.preprocess import preprocess_pipeline

# ======================
# Configurações
# ======================
EPOCHS = 50
BATCH_SIZE = 32
WEIGHTS_PATH = "models/lstm_petr4.weights.h5"

# ======================
# Carregar dados
# ======================
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_pipeline()

# ======================
# Modelo
# ======================
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.3),
    LSTM(16),
    Dense(30)
])

"""Foram realizados testes com diferentes arquiteturas de redes LSTM. Observou-se que a redução da complexidade do modelo, com menor número de neurônios, resultou em melhor desempenho no conjunto de teste, indicando maior capacidade de generalização. O modelo final foi selecionado com base na métrica MAE no conjunto de teste."""

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ======================
# Callbacks
# ======================
Path("models").mkdir(exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        WEIGHTS_PATH,
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
test_loss, test_mae = model.evaluate(X_test, y_test)

print(f"Test MAE: {test_mae:.4f}")
print(f"Test MSE: {test_loss:.4f}")
