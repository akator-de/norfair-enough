FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /reid
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install norfair-enough

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /demo/src && \
    chown -R appuser:appuser /demo /reid
USER appuser

WORKDIR /demo/src/
