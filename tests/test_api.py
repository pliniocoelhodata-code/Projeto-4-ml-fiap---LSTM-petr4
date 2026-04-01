from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_invalid_payload_length():
    response = client.post("/v1/predict", json={"last_60_days": [1.0, 2.0, 3.0]})

    assert response.status_code == 422
