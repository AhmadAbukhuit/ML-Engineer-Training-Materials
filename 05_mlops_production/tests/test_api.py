import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Insert 05_mlops_production to sys.path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["model_loaded"] is True
    assert "version" in data


def test_model_info_endpoint(client):
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert data["n_features"] == 10
    assert "model_name" in data


def test_predict_endpoint_valid(client):
    sample_features = [0.5, -1.2, 3.4, 0.0, 1.1, -0.4, 2.2, -0.9, 0.1, 1.5]
    payload = {"features": sample_features}

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert "class_probabilities" in data
    assert data["latency_ms"] >= 0.0


def test_predict_endpoint_invalid_feature_count(client):
    # Only 5 features provided instead of required 10
    payload = {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_null_values(client):
    # Payload contains null
    payload = {"features": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict_endpoint(client):
    instances = [
        [0.5, -1.2, 3.4, 0.0, 1.1, -0.4, 2.2, -0.9, 0.1, 1.5],
        [-0.5, 1.0, -1.4, 0.2, -0.1, 0.8, -1.2, 0.4, -0.3, -1.0],
        [1.2, 0.3, 0.1, -0.5, 0.8, -0.2, 0.5, -0.1, 0.2, 0.4]
    ]
    payload = {"instances": instances}

    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert len(data["probabilities"]) == 3
    assert all(p in [0, 1] for p in data["predictions"])
