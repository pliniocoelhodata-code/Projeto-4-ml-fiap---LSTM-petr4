# PETR4 LSTM Forecast API

[![CI](https://github.com/pliniocoelhodata-code/Projeto-4-ml-fiap---LSTM-petr4/actions/workflows/ci.yml/badge.svg)](https://github.com/pliniocoelhodata-code/Projeto-4-ml-fiap---LSTM-petr4/actions/workflows/ci.yml)
[![CD](https://github.com/pliniocoelhodata-code/Projeto-4-ml-fiap---LSTM-petr4/actions/workflows/cd.yml/badge.svg)](https://github.com/pliniocoelhodata-code/Projeto-4-ml-fiap---LSTM-petr4/actions/workflows/cd.yml)

API de previsao para a acao `PETR4.SA` usando `FastAPI` e um modelo `LSTM` treinado com `TensorFlow`.

O projeto demonstra um fluxo completo de ML aplicado:

- coleta e preprocessamento de dados de series temporais
- treinamento e persistencia de artefatos do modelo
- serving de inferencia com `FastAPI`
- containerizacao com Docker
- execucao local com Docker Compose
- validacao automatizada com CI
- publicacao automatica da imagem no `ghcr.io`

## Destaques

- problema real de negocio: previsao de preco de acao
- separacao entre pipeline de treino e camada de inferencia
- modelo e scaler versionados no repositorio
- testes de API, integracao e smoke test em container
- pipeline de CI/CD simples e facil de explicar

## Arquitetura

Fluxo principal:

`yfinance -> CSV bruto -> preprocessamento -> treino LSTM -> models/saved_model + scaler.pkl -> FastAPI -> /v1/predict`

## Estrutura Do Projeto

```text
.
|-- .github/workflows/
|   |-- cd.yml
|   `-- ci.yml
|-- Dockerfile
|-- docker-compose.yaml
|-- models/
|   |-- metadata.json
|   |-- saved_model/
|   `-- scaler.pkl
|-- render.yaml
|-- requirements.txt
|-- tests/
|   `-- test_api.py
`-- src/
    |-- api/
    |   `-- main.py
    |-- data/
    |   |-- collect_data.py
    |   `-- preprocess.py
    `-- models/
        `-- train_lstm.py
```

## Como Rodar Localmente

### Opcao 1. Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opcao 2. Docker Compose

```bash
docker compose up --build
```

### Endpoints

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/`
- Predicao: `POST /v1/predict`

### Exemplo De Requisicao

```json
{
  "last_60_days": [
    32.1, 32.5, 32.4, 32.7, 32.6, 32.9, 33.0, 33.2, 33.1, 33.4,
    33.5, 33.6, 33.7, 33.8, 33.9, 34.0, 34.1, 34.2, 34.0, 33.9,
    34.1, 34.2, 34.4, 34.3, 34.5, 34.6, 34.8, 34.7, 34.9, 35.0,
    35.1, 35.2, 35.0, 35.3, 35.5, 35.4, 35.6, 35.7, 35.8, 35.9,
    36.0, 36.1, 36.2, 36.0, 36.3, 36.5, 36.4, 36.6, 36.8, 36.7,
    36.9, 37.0, 37.1, 37.2, 37.0, 37.3, 37.4, 37.5, 37.6, 37.7
  ]
}
```

Resposta esperada:

```json
{
  "prediction_30_days": [35.8, 35.9, 36.0],
  "latency_seconds": 0.42
}
```

## Pipeline De Treinamento

O treino e separado da API.

### 1. Coletar dados

```bash
python src/data/collect_data.py
```

Gera o arquivo `data/raw/petr4_raw.csv`.

### 2. Treinar o modelo

```bash
python -m src.models.train_lstm
```

Artefatos gerados:

- `models/saved_model/`
- `models/scaler.pkl`
- `models/metadata.json`

## Docker

### Build local

```bash
docker build -t petr4-lstm-api:latest .
```

### Executar localmente

```bash
docker run --rm -p 8000:8000 petr4-lstm-api:latest
```

## Imagem Publicada No GHCR

O workflow de CD publica a imagem em:

```text
ghcr.io/pliniocoelhodata-code/petr4-lstm-api
```

Exemplo de uso:

```bash
docker pull ghcr.io/pliniocoelhodata-code/petr4-lstm-api:latest
docker run --rm -p 8000:8000 ghcr.io/pliniocoelhodata-code/petr4-lstm-api:latest
```

Tags geradas pelo CD:

- `latest` na branch padrao
- nome da branch
- tag Git, como `v1.0.0`
- SHA do commit

## CI E CD

### CI

O workflow em `.github/workflows/ci.yml` executa:

- instalacao de dependencias Python
- compilacao do codigo
- testes da API com `pytest`
- teste de integracao com artefatos reais
- build da imagem Docker
- smoke test da API rodando em container

### CD

O workflow em `.github/workflows/cd.yml`:

- roda em push para `main` ou `master`
- aceita execucao manual
- gera tags automaticamente
- publica a imagem no GitHub Container Registry

Pre-requisito no GitHub:

- GitHub Actions com permissao de escrita em packages

## Deploy Publico

- `https://petr4-lstm-api.onrender.com/docs`

## Melhorias Futuras

- ampliar cobertura de testes da API
- adicionar monitoramento e metricas
- incluir badges adicionais de pacote ou deploy, se fizer sentido

## Autor

Plinio Coelho
