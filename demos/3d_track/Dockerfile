#  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && \
    apt-get install -y libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install mediapipe==0.8.11 && \
    pip install opencv-python==4.5.5.64

RUN pip install norfair-enough

WORKDIR /demo/src/
