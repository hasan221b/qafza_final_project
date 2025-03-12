import pytest
from src.models.finbert_model import FinBertInference
from src.preprocessing.preprocess import clean_text
@pytest.fixture

def model():
    return FinBertInference()

def test_model_initialization(model):
    #Test if the model initiakizes correctly
    assert model is not None , "Model is not initialized. It should not be None."

def test_sentiment_prediction(model):
    #Test sentiment prediction for sample text.

    sample_text = "I love this product! It's amaizing"
    sample_text = clean_text(sample_text)
    result = model.predict_sentiment(sample_text)


    assert isinstance(result, dict),"Result should be a dictionary"
    assert 'sentiment' in result, "Sentiment key should be in the result"
    assert 'probabilities' in result, "Probabilities should be included"
    # Check if sentiment is one of the expected labels
    assert result["sentiment"] in ["positive", "negative", "neutral"], "Unexpected sentiment label"
    #Check if probabilities sum to 1
    total_prob = sum(result['probabilities'].values())
    assert abs(total_prob - 1.0) < 1e-3, f'Probabilities sum should be close to 1, got {total_prob}'


    