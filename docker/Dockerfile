# syntax=docker/dockerfile:1
# Linux x86_64 image — required for libcg.so (the game engine shared library)
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python deps. torch is required by every agent since v024 (card-embedding NN);
# without it their import guard degrades each decision to the greedy fallback and
# the gate silently measures the wrong pilot. CPU wheels — the gate has no GPU.
RUN pip install --no-cache-dir numpy pandas lightgbm
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Copy the repo (src/ contains cg/ with libcg.so, agents/, etc.)
COPY . /workspace/

# Make src importable so `import cg` and `import agents.*` resolve
ENV PYTHONPATH=/workspace/src

# Default: run the simulation harness
CMD ["python", "src/sim_harness.py"]
