FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    TRITON_CACHE_DIR=/cache/triton \
    HF_HOME=/cache/huggingface \
    BREEZE_MODEL_ROOT=/models \
    BREEZE_MODEL_DIR=/models/Breeze-TTS-2 \
    BREEZE_WEIGHTS_FILE=Breeze-TTS-2-int8-hybrid.safetensors \
    BREEZE_PROFILE_DIR=/data/profiles

# gcc/g++ are NOT for FlashAttention.
# Breeze's --fast-depth-decoder uses torch.compile(), so keep a host compiler
# available for PyTorch/Inductor runtime compilation.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        gcc \
        g++ \
        libsndfile1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        sox \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade pip setuptools wheel packaging

WORKDIR /opt/breeze-infer

#
# Install CUDA PyTorch explicitly first.
#
RUN python -m pip install \
        torch==2.9.1 \
        torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128

#
# Install project dependencies, but explicitly handle packages for which
# we care about wheel selection ourselves.
#
COPY requirements.txt /tmp/requirements.txt

RUN grep -vE \
        '^(torch|torchaudio|flash-attn|comfy-kitchen)([[:space:]]|[<>=!~]|$)' \
        /tmp/requirements.txt \
        > /tmp/requirements-runtime.txt \
    && python -m pip install -r /tmp/requirements-runtime.txt \
    && python -m pip install \
        --only-binary=:all: \
        comfy-kitchen==0.2.31

#
# comfy-kitchen 0.2.31 dynamically probes cuBLASLt by looking for
# libcublasLt.so.13 and then libcublasLt.so.  The CUDA runtime image
# and PyTorch's nvidia-cublas-cu12 package provide libcublasLt.so.12
# but may omit the unversioned development symlink.
#
# Without this link, comfy-kitchen silently omits its native CUDA
# int8_linear backend and dispatches Breeze's ConvRot INT8 layers
# through the much slower Triton fallback.
#
RUN ln -sf \
    /opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12 \
    /usr/local/cuda/lib64/libcublasLt.so

#
# FlashAttention 2.8.3
#
# Install the known-good prebuilt CPython 3.12 / CUDA 12 / Torch 2.9 /
# CXX11-ABI=TRUE wheel. Preserve the complete wheel filename because pip
# uses it to validate Python/platform compatibility.
#
ARG FLASH_ATTN_WHEEL="flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
ARG FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
ARG FLASH_ATTN_SHA256="4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810"

RUN curl -fL "${FLASH_ATTN_URL}" -o "/tmp/${FLASH_ATTN_WHEEL}" \
    && echo "${FLASH_ATTN_SHA256}  /tmp/${FLASH_ATTN_WHEEL}" | sha256sum -c - \
    && python -m pip install --no-deps "/tmp/${FLASH_ATTN_WHEEL}" \
    && rm "/tmp/${FLASH_ATTN_WHEEL}"

RUN python - <<'PY'
import torch
import flash_attn

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CXX11 ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("FlashAttention:", flash_attn.__version__)
PY

#
# Application
#
COPY . .

#
# Keep model weights outside the image.
#
# At runtime mount one host directory at /models containing:
#
#   /models/Breeze-TTS-2/
#   /models/Breeze-TTS-2-int8-hybrid.safetensors
#
# Mount a second host directory at /data/profiles for persistent uploaded
# profiles, matching the OmniVoice deployment layout.
#
RUN mkdir -p /models /data/profiles /cache
RUN chmod +x /opt/breeze-infer/scripts/docker-entrypoint.sh

VOLUME ["/models", "/data/profiles", "/cache"]

#
# Build-time import/ABI verification.
# No GPU is required for these checks.
#
RUN python - <<'PY'
import torch
import flash_attn
import comfy_kitchen
import comfy_kitchen.backends.cuda as ck_cuda

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CXX11 ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("FlashAttention:", flash_attn.__version__)
print("comfy-kitchen CUDA extension:", ck_cuda._EXT_AVAILABLE)
print("comfy-kitchen cuBLASLt:", ck_cuda._CUBLASLT_AVAILABLE)

cuda_caps = comfy_kitchen.list_backends()["cuda"]["capabilities"]
print("comfy-kitchen CUDA int8_linear:", "int8_linear" in cuda_caps)

assert torch.__version__.startswith("2.9.1")
assert torch.version.cuda == "12.8"
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
assert flash_attn.__version__.startswith("2.8.3")

assert ck_cuda._EXT_AVAILABLE, "comfy-kitchen CUDA extension failed to load"
assert ck_cuda._CUBLASLT_AVAILABLE, "comfy-kitchen could not load cuBLASLt"
assert "int8_linear" in cuda_caps, \
    "comfy-kitchen CUDA int8_linear unavailable; would fall back to Triton"
PY

EXPOSE 7860

# Initial graph compilation/warmup can take a while, particularly on first
# launch of a persistent container.
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=300s \
    --retries=3 \
    CMD python -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3).read()"

ENTRYPOINT ["/opt/breeze-infer/scripts/docker-entrypoint.sh"]
