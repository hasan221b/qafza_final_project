import pytest
from fastapi.testclient import TestClient
from src.api.finbert_api2 import app

client = TestClient(app)

def test_health_check():
    #Test if the health of check endpoint is working.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status":"healthy"}

def test_root():
    #Test root endpoint to check if API is running
    response = client.get("/")
    assert response.status_code==200
    assert response.json() == {"message":"Sentiment analysis API is running"}

def test_sentiment_prediction():
    #Test sentiment prediction endpoint.
    sample_text = {"text": "This is a fantastic product!"}
    response = client.post("/predict", json=sample_text)

    assert response.status_code == 200
    result = response.json()

    assert "sentiment" in result, "Sentiment key should be in the response"
    assert "confidence" in result, "Confidence score should be included"
    assert "probabilities" in result, "Probabilities should be included"
    
    assert isinstance(result["confidence"], float), "Confidence should be a float"
    assert isinstance(result["probabilities"], dict), "Probabilities should be a dictionary"

# Check if probabilities sum to ~1
    total_prob = sum(result["probabilities"].values())
    assert abs(total_prob - 1.0) < 1e-3, f"Probabilities sum should be close to 1, got {total_prob}"

def test_invalid_input():
    #Test API with invalid input.
    response = client.post("/predict", json={})
    assert response.status_code == 422, "Should return 422 for missing text field"