import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from student_performance.api import app
from student_performance.pipeline.train_pipeline import TrainPipeline


@pytest.fixture(scope="module")
def client():
    repo_root = Path(__file__).resolve().parents[1]
    original_cwd = Path.cwd()
    os.chdir(repo_root)
    TrainPipeline().run()

    try:
        with TestClient(app) as c:
            yield c
    finally:
        os.chdir(original_cwd)


def test_valid_prediction(client: TestClient):
    """Test valid prediction request."""
    response = client.post(
        "/predict",
        json={
            "gender": "female",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "risk_probability",
        "risk_tier",
        "performance_band",
        "score_range",
        "score_prediction",
    }
    assert isinstance(data["score_prediction"], (int, float))
    assert isinstance(data["score_range"], list)
    assert len(data["score_range"]) == 2
    assert data["risk_tier"] in {"low", "medium", "high"}
    assert data["performance_band"] in {"low", "medium", "high"}
    assert 0.0 <= float(data["risk_probability"]) <= 1.0


def test_empty_field_validation(client: TestClient):
    """Test rejection of empty fields."""
    response = client.post(
        "/predict",
        json={
            "gender": "",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
        },
    )

    assert response.status_code == 422
    assert "gender" in response.json()["detail"].lower()
    assert "empty" in response.json()["detail"].lower()


def test_invalid_category(client: TestClient):
    """Test rejection of invalid category values."""
    response = client.post(
        "/predict",
        json={
            "gender": "non-binary",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "invalid value" in detail
    assert "gender" in detail


def test_case_insensitive_validation(client: TestClient):
    """Test that validation is case-insensitive."""
    response = client.post(
        "/predict",
        json={
            "gender": "FEMALE",
            "race_ethnicity": "GROUP B",
            "parental_level_of_education": "BACHELOR'S DEGREE",
            "lunch": "STANDARD",
            "test_preparation_course": "NONE",
        },
    )

    assert response.status_code == 200


def test_whitespace_trimming(client: TestClient):
    """Test that whitespace is trimmed."""
    response = client.post(
        "/predict",
        json={
            "gender": "  female  ",
            "race_ethnicity": " group b ",
            "parental_level_of_education": "  bachelor's degree  ",
            "lunch": "  standard  ",
            "test_preparation_course": "  none  ",
        },
    )

    assert response.status_code == 200


def test_missing_field(client: TestClient):
    """Test rejection when required field is missing."""
    response = client.post(
        "/predict",
        json={
            "gender": "female",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            # Missing: test_preparation_course
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "missing" in detail
    assert "test_preparation_course" in detail


def test_extra_field(client: TestClient):
    """Test rejection when unexpected field is provided."""
    response = client.post(
        "/predict",
        json={
            "gender": "female",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
            "age": 18,  # Extra field
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "unexpected" in detail
    assert "age" in detail


def test_batch_prediction(client: TestClient):
    """Test batch prediction endpoint."""
    response = client.post(
        "/predict_batch",
        json=[
            {
                "gender": "female",
                "race_ethnicity": "group B",
                "parental_level_of_education": "bachelor's degree",
                "lunch": "standard",
                "test_preparation_course": "none",
            },
            {
                "gender": "male",
                "race_ethnicity": "group C",
                "parental_level_of_education": "some college",
                "lunch": "free/reduced",
                "test_preparation_course": "completed",
            },
        ],
    )

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 2

    for item in data["predictions"]:
        assert set(item.keys()) == {
            "risk_probability",
            "risk_tier",
            "performance_band",
            "score_range",
            "score_prediction",
        }
        assert isinstance(item["score_prediction"], (int, float))
        assert item["risk_tier"] in {"low", "medium", "high"}
        assert item["performance_band"] in {"low", "medium", "high"}
        assert 0.0 <= float(item["risk_probability"]) <= 1.0
        assert isinstance(item["score_range"], list)
        assert len(item["score_range"]) == 2


def test_health_endpoint(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["pipeline_loaded"] is True


def test_schema_endpoint(client: TestClient):
    """Test schema endpoint returns feature info."""
    response = client.get("/schema")

    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    # Should have 5 features
    assert len(data["features"]) == 5


def test_null_value(client: TestClient):
    """Test rejection of null values."""
    response = client.post(
        "/predict",
        json={
            "gender": "female",
            "race_ethnicity": None,  # Null value
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
        },
    )

    assert response.status_code == 422


def test_whitespace_only_field(client: TestClient):
    """Test rejection of whitespace-only values."""
    response = client.post(
        "/predict",
        json={
            "gender": "   ",  # Whitespace only
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "empty" in detail
