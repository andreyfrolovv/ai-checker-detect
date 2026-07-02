FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Создаем директорию для динамически скачиваемых моделей
RUN mkdir -p /app/models
VOLUME /app/models

EXPOSE 8000

ENV MODELS_DIR=/app/models

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]