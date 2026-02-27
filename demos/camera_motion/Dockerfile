FROM ultralytics/yolov5:latest

# Install Norfair
RUN pip install norfair-enough

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /demo/src && \
    chown -R appuser:appuser /demo
USER appuser

WORKDIR /demo/src/
