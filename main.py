from src.api.finbert_api2 import FinBertAPI
import uvicorn

if __name__ == "__main__":
    uvicorn.run(FinBertAPI().app, host="0.0.0.0", port=8000)
