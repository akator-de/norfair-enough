# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /sahi
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /demo/src/
