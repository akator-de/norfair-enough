FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /demo/requirements.txt
RUN pip install --no-cache-dir -r /demo/requirements.txt

COPY src/ /demo/src/
RUN groupadd --system app && useradd --system --gid app --create-home app && \
    chown -R app:app /demo

WORKDIR /demo/src/
USER app
