from fastapi import FastAPI, HTTPException
from src.models.finbert_model import FinBertInference
from src.schemas import TextInput, PredictionOutput2
from src.preprocessing.preprocess import clean_text
from src.utils.logger import logger
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd

class FinBertAPI:
    def __init__(self):
        self.app = FastAPI()
        self._setup_routes()

        Instrumentator().instrument(self.app).expose(self.app)  # Prometheus monitoring

        logger.info("Starting FastAPI server...")

        try:
            self.finbert_model = FinBertInference()
        except Exception as e:
            logger.error(f'Failed to initialize model: {str(e)}')
            raise RuntimeError('Model initialization failed')

    def _setup_routes(self):
        @self.app.get("/")
        def read_root():
            logger.info("Root endpoint accessed.")
            return {"message": "Sentiment analysis API is running"}

        @self.app.get("/health")
        async def health_check():
            logger.info("Health check endpoint accessed.")
            return {"status": "healthy"}

        @self.app.post("/predict", response_model=PredictionOutput2)
        async def sentiment_predict(input_data: TextInput):
            logger.info(f"Received prediction request: {input_data.text[:50]}...")  # Log first 50 chars
            try:
                text = clean_text(input_data.text)
                result = self.finbert_model.predict_sentiment(text)

                compound = self.finbert_model.compound_score(
                    result['probabilities']['positive'],
                    result['probabilities']['negative'],
                    result['probabilities']['neutral']
                )

                rank = self.finbert_model.ranking_method(
                    text,
                    result['probabilities']['positive'],
                    result['probabilities']['negative'],
                    result['probabilities']['neutral'],
                    compound
                )

                df = pd.DataFrame([rank])  # Convert dict to DataFrame
                df.to_csv('news.csv', mode='a', index=False, header=False)

                logger.info(f"Prediction successful: {result['sentiment']} with confidence {result['confidence']:.4f}")

                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])

                return {
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'compound': compound,
                    'rank_score': rank['score']
                }

            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail="Prediction failed")

        @self.app.get("/ranking")
        async def news_ranking():
            df = pd.read_csv('news.csv')
            sorted_df = df.sort_values(by='score', ascending=False)
            return sorted_df.to_dict(orient='records')

# Create an instance of the class and expose it as `app`
app = FinBertAPI().app
