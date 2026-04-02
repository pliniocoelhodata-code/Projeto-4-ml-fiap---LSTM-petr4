import numpy as np

from fastapi.testclient import TestClient

from src.api import main

app = main.app

client = TestClient(app)
VALID_INPUT = {"last_60_days": [30.0] * 60}


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_invalid_payload_length():
    response = client.post("/v1/predict", json={"last_60_days": [1.0, 2.0, 3.0]})

    assert response.status_code == 422


def test_docs_endpoint_is_available():
    response = client.get("/docs")

    assert response.status_code == 200
    assert "PETR4 LSTM Forecast API" in response.text


def test_predict_returns_500_when_model_is_not_loaded(monkeypatch):
    monkeypatch.setattr(main, "model", None)
    monkeypatch.setattr(main, "scaler", None)

    response = client.post("/v1/predict", json=VALID_INPUT)

    assert response.status_code == 500
    assert "detail" in response.json()


def test_predict_returns_expected_response_shape(monkeypatch):
    class DummyScaler:
        def transform(self, values):
            return values

        def inverse_transform(self, values):
            return values

    class DummyModel:
        def predict(self, values, verbose=0):
            return np.array([[float(index) for index in range(30)]])

    monkeypatch.setattr(main, "scaler", DummyScaler())
    monkeypatch.setattr(main, "model", DummyModel())

    response = client.post("/v1/predict", json=VALID_INPUT)

    assert response.status_code == 200

    body = response.json()
    assert "prediction_30_days" in body
    assert "latency_seconds" in body
    assert len(body["prediction_30_days"]) == 30
    assert isinstance(body["latency_seconds"], float)


def test_predict_with_real_artifacts():
    with TestClient(app) as integration_client:
        response = integration_client.post("/v1/predict", json=VALID_INPUT)

    assert response.status_code == 200

    body = response.json()
    assert len(body["prediction_30_days"]) == 30
    assert isinstance(body["latency_seconds"], float)
