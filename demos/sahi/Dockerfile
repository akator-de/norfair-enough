# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /sahi
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install norfair-enough

WORKDIR /demo/src/
