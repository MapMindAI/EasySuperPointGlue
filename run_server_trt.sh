#!/usr/bin/env bash
set -e


MODEL_DIR=$(pwd)

docker run -d --gpus all --rm --name 'tritonserver_trt' \
-p 8001:8001 \
-v ${MODEL_DIR}:/repo \
-v ${MODEL_DIR}/workspace:/workspace \
ghcr.io/mapmindai/tritonserver_amd64:latest \
bash -c "
  echo '====> Step 1: Checking environment'
  nvidia-smi
  echo ''
  echo '====> Step 2: Check TRT plan files'
  bash /repo/model_trt/convert_models.sh
  echo ''
  echo '====> Step 3: Starting Triton'
  tritonserver --model-repository=/repo/model_repository_trt
"


# docker exec -it tritonserver bash
