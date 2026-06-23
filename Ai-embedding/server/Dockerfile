FROM python:3.11-slim

WORKDIR /app

# System deps:
#   libgomp1     — OpenMP for faiss-cpu
#   build-essential + cmake — needed to compile llama-cpp-python from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install llama-cpp-python first (CPU-only, no GPU/Metal)
# GGML_NATIVE=OFF → portable binary, no host-specific AVX optimizations
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_BLAS=OFF" \
    pip install --no-cache-dir llama-cpp-python>=0.3.0

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV OMP_NUM_THREADS=1
# llama-cpp: use all available CPUs (Railway typically gives 2–4 vCPUs)
ENV LLAMA_N_THREADS=4

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}
