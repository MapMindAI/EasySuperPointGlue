#!/usr/bin/env bash
set -e


MODEL_DIR=$(pwd)/model_repository

docker run -d --gpus all --rm --name 'tritonserver' \
-p 8001:8001 \
-v ${MODEL_DIR}:/models \
ghcr.io/mapmindai/tritonserver:latest \
tritonserver --model-repository=/models; exec bash
