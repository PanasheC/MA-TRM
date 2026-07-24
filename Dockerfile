FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    DISABLE_COMPILE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/MA-TRM
COPY . .

RUN python3.10 -m pip install --upgrade pip wheel setuptools && \
    python3.10 -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126 && \
    python3.10 -m pip install -r requirements-lock.txt && \
    python3.10 -m pip install --no-build-isolation adam-atan2==0.0.3

CMD ["bash"]
