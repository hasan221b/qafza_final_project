import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.logger import logger

import math

class FinBertInference:
    def __init__(self, model_path='ProsusAI/finbert'):
        #Load pre-trained FinBERT model and tokenizer
        try:
            logger.info("Initializing FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.labels = ['positive','negative','neutral']
            logger.info("FinBERT model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise RuntimeError("Model initialization failed")

    def predict_sentiment(self, text:str):
        #Preform sentiment analysis on the input text

        logger.info(f"Predicting sentiment for the text {text[:50]}...") #log the first 50 chars only

        inputs = self.tokenizer(
            text, return_tensors='pt',
            truncation=True,
            max_length = 512,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1).numpy()[0]
        predicted_label = self.labels[probabilities.argmax()]

        logger.info(f"Prediction: {predicted_label} (Confidence: {max(probabilities):.4f})")

        return {
            'sentiment' : predicted_label,
            'confidence' : float(probabilities.max()),
            'probabilities' : {
                label: float(prob) for label, prob in zip(self.labels, probabilities)
            },          

            }               
    
    def compound_score(self,pos_prob:float,neg_prob:float,neu_prob:float):
        logger.info("Calculating compound score for the text..")
        numerator = pos_prob - neg_prob

        # Calculate the denominator (sqrt of sum of squares + alpha)
        denominator = math.sqrt((pos_prob + neg_prob + neu_prob) ** 2 + 15)

        # Compute the compound score
        compound = numerator / denominator
        logger.info(f"Compound score is: {compound}")
        
        return compound
    
    def ranking_method(self,text:str,pos_prob:float,neg_prob:float,neu_prob:float,comp:float):
        logger.info("Calculating rank score for the text..")
        rank_score = (pos_prob * 2)+(neu_prob * 1)-(neg_prob * 3)+(comp * 5)
    
        return {"news":text,
                "score":rank_score}