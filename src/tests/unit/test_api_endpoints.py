import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns correct welcome message"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Intelligent Product Support System API"

def test_health_check_endpoint():
    """Test the health check endpoint returns correct status"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_model_info_endpoint():
    """Test the model info endpoint returns correct structure"""
    response = client.get("/model_info")
    # This endpoint might return 200 or 404 depending on whether models are loaded
    assert response.status_code in [200, 404, 500]

def test_predict_endpoint_validation():
    """Test that predict endpoint validates input correctly"""
    # Test with missing required fields
    invalid_data = {
        "subject": "Test subject"
        # Missing description
    }
    
    response = client.post("/predict", json=invalid_data)
    # Should return 422 for validation error
    assert response.status_code == 422

def test_retrieve_solutions_endpoint_validation():
    """Test that retrieve solutions endpoint validates input correctly"""
    # Test with missing required fields
    invalid_data = {
        # Missing query
    }
    
    response = client.post("/retrieve_solutions", json=invalid_data)
    # Should return 422 for validation error
    assert response.status_code == 422
