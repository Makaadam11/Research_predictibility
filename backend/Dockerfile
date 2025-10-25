FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    libopenjp2-7 \
    libtiff5 \
    libfreetype6-dev \
    liblcms2-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ustaw zmienne środowiskowe (możesz nadpisać przez docker-compose/.env)
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]