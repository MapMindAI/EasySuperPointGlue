#!/usr/bin/env bash
set -e


MODEL_DIR=$(pwd)

docker run -d --gpus all --rm --name 'tritonserver' \
-p 8001:8001 \
-v ${MODEL_DIR}:/repo \
-v ${MODEL_DIR}/workspace:/workspace \
ghcr.io/mapmindai/tritonserver_amd64:latest \
tritonserver --model-repository=/repo/model_repository; exec bash


# docker exec -it tritonserver bash
