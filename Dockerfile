
# ---- Base: CUDA runtime + cuDNN (GPU-ready) ----
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
    PORT=7860

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git build-essential ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python3 -m pip install --upgrade pip

# Install a CUDA 12.1 build of PyTorch + torchvision from the official index:
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0+cu121 torchvision==0.19.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# The rest of the deps
RUN pip install -r requirements.txt

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
