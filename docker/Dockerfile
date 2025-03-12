FROM python:3.11.4

WORKDIR /app

COPY requierments.txt .

COPY . .

RUN pip install -r requierments.txt

EXPOSE 8000

# Start the application using Gunicorn with Uvicorn workers
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "src.api.finbert_api2:app"]
