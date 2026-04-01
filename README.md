# PETR4 LSTM Forecast API

API de previsao para a acao `PETR4.SA` usando `FastAPI` e um modelo `LSTM` treinado com `TensorFlow`.

Este projeto foi montado para demonstrar um fluxo completo de machine learning aplicado:

- coleta de dados com `yfinance`
- preprocessamento para series temporais
- treinamento de uma rede `LSTM`
- persistencia de artefatos do modelo
- exposicao da inferencia por API
- empacotamento com Docker
- deploy local com Docker Compose
- validacao automatizada com CI
- publicacao automatica da imagem com CD

## Destaques Para Portfolio

- problema real de negocio: previsao de preco de acao
- stack moderna de API + ML serving
- separacao entre pipeline de treino e camada de inferencia
- artefatos prontos para containerizacao e demonstracao
- pipeline simples de CI para validar codigo e build
- entrega automatica da imagem no GitHub Container Registry

## Arquitetura

Fluxo principal:

`yfinance -> CSV bruto -> preprocessamento -> treino LSTM -> models/saved_model + scaler.pkl -> FastAPI -> endpoint /v1/predict`

## Estrutura Do Projeto

```text
.
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

### 1. Ambiente Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Subir A API

O projeto ja possui artefatos versionados em `models/`, entao a API pode ser iniciada sem novo treinamento:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/`

### 3. Exemplo De Requisicao

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

Isso gera o arquivo `data/raw/petr4_raw.csv`.

### 2. Treinar o modelo

```bash
python -m src.models.train_lstm
```

Ao final do processo, os artefatos relevantes sao:

- `models/saved_model/`
- `models/scaler.pkl`
- `models/metadata.json`

## Docker

### Build da imagem

```bash
docker build -t petr4-lstm-api:latest .
```

### Rodar o container

```bash
docker run --rm -p 8000:8000 petr4-lstm-api:latest
```

## Docker Compose

Para subir o servico localmente com um comando:

```bash
docker compose up --build
```

O Compose faz mais sentido aqui quando o objetivo e:

- demonstrar reproducao local rapida
- facilitar avaliacao por recrutadores e entrevistadores
- preparar terreno para adicionar mais servicos no futuro

## CI

O projeto agora inclui workflow de GitHub Actions em `.github/workflows/ci.yml` com:

- instalacao de dependencias
- compilacao do codigo Python
- teste basico da API
- build da imagem Docker

## CD

O projeto tambem inclui workflow de GitHub Actions em `.github/workflows/cd.yml` para publicar a imagem no `ghcr.io`.

Esse workflow:

- roda em push para `main` ou `master`
- aceita execucao manual
- gera tags de branch, tag Git e SHA do commit
- publica a imagem `ghcr.io/<usuario-ou-org>/petr4-lstm-api`

Pre-requisitos no GitHub:

- manter o repositrio hospedado no GitHub
- permitir GitHub Actions com permissao para escrever em packages

Esse fluxo e mais coerente com o tamanho atual do projeto do que manter um manifesto Kubernetes sem necessidade real de orquestracao.

## Endpoint Disponivel

- `GET /`
- `POST /v1/predict`

Deploy publico atual:

- `https://petr4-lstm-api.onrender.com/docs`

## Melhorias Futuras

- adicionar testes automatizados da API
- adicionar monitoramento e metricas

## Autor

Plinio Coelho
