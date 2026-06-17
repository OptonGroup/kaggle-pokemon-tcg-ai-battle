# syntax=docker/dockerfile:1
# Linux x86_64 image — required for libcg.so (the game engine shared library)
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir numpy pandas lightgbm

# Copy the repo (src/ contains cg/ with libcg.so, agents/, etc.)
COPY . /workspace/

# Make src importable so `import cg` and `import agents.*` resolve
ENV PYTHONPATH=/workspace/src

# Default: run the simulation harness
CMD ["python", "src/sim_harness.py"]
