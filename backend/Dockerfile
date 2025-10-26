FROM python:3.13-slim-bookworm

WORKDIR /app

# Minimalne biblioteki systemowe pod Pillow/obrazy + BLAS (openblas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    libpng-dev \
    libopenjp2-7 \
    libtiff-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8000"]