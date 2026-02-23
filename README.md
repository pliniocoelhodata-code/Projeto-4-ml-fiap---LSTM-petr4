# 📈 PETR4 LSTM Forecast API

API de previsão de preços da ação **PETR4.SA** utilizando rede neural
LSTM.

------------------------------------------------------------------------

# 🧠 Objetivo

Desenvolver um sistema de previsão de séries temporais para o preço da
ação PETR4 utilizando Deep Learning (LSTM), com:

-   Separação adequada de treino / validação / teste (time-series split)
-   Avaliação formal do modelo
-   Comparação com baseline simples
-   Versionamento de modelo
-   API versionada
-   Monitoramento básico de latência

------------------------------------------------------------------------

# 📂 Estrutura do Projeto

    ├── src/
    │   ├── data/
    │   ├── models/
    │   └── api/
    │
    ├── models/
    │   ├── saved_model
    │   ├── scaler.pkl
    │   └── metadata.json
    │
    ├── requirements.txt
    ├── render.yaml
    └── README.md

------------------------------------------------------------------------

# 🔄 Pipeline

## 1️⃣ Coleta de dados

Dados históricos obtidos via `yfinance` (ticker PETR4.SA).

## 2️⃣ Pré-processamento

-   Normalização com MinMaxScaler
-   Lookback: 60 dias
-   Horizon: 30 dias
-   Time-series split (sem leakage)

## 3️⃣ Modelo LSTM

Arquitetura:

    LSTM(32, return_sequences=True)
    Dropout(0.3)
    LSTM(16)
    Dense(30)

Loss: MSE\
Métrica: MAE\
EarlyStopping com restore_best_weights=True

Modelo salvo em:

    models/model.keras

------------------------------------------------------------------------

# 📊 Avaliação

Métricas utilizadas:

-   MAE
-   RMSE
-   MAPE
-   Baseline MAE (último valor)

Exemplo:

MAE (t+1): 0.64\
RMSE (t+1): 0.65\
MAPE (t+1): 36.8%\
Baseline MAE: 0.10

------------------------------------------------------------------------

# 🚀 API

Endpoint:

POST `/v1/predict`

Payload:

``` json
{
  "last_60_days": [ ... 60 valores ... ]
}
```

Resposta:

``` json
{
  "prediction_30_days": [...],
  "latency_seconds": 1.38
}
```

Monitoramento implementado: - Logging estruturado - Medição de
latência - Log de eventos e erros

------------------------------------------------------------------------

# 🛠️ Execução Local

Instalar dependências:

pip install -r requirements.txt

Treinar modelo:

python -m src.models.train_lstm

Subir API:

uvicorn src.api.main:app --reload

Acessar: http://127.0.0.1:8000/docs

------------------------------------------------------------------------

# 🌍 Deploy

Deploy realizado na plataforma Render.

https://petr4-lstm-api.onrender.com/docs

------------------------------------------------------------------------

# 👨‍💻 Autor

Plinio Coelho
