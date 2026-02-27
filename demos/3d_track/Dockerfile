#  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install mediapipe

RUN pip install norfair-enough

WORKDIR /demo/src/
