FROM python:3.11.4

WORKDIR /app

COPY requierments.txt .

COPY . .

RUN pip install -r requierments.txt

EXPOSE 8000

CMD ['uvicorn', 'src.api.finbert_api2:app',"--host","0.0.0.0","-port","8000"]

