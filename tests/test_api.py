from fastapi.testclient import TestClient

from src.api import app


def test_health_and_predict():
    client = TestClient(app)
    # check root route
    r = client.get("/")
    assert r.status_code == 200
    # check predict route
    r = client.post("/predict", json={"text": "I feel happy"})
    assert r.status_code == 200
    assert "label" in r.json()
