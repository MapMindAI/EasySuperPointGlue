#!/usr/bin/env bash
set -e


MODEL_DIR=$(pwd)

docker run --gpus all --rm --name 'tritonserver' \
-p 8001:8001 \
-v ${MODEL_DIR}:/repo \
-v ${MODEL_DIR}/workspace:/workspace \
ghcr.io/mapmindai/tritonserver_amd64:latest \
bash -c "
  echo '====> Starting Triton'
  tritonserver --model-repository=/repo/model_repository
"


# docker exec -it tritonserver bash
