FROM multiarch/qemu-user-static as qemu

FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime

COPY --from=qemu /usr/bin/qemu-aarch64-static /usr/bin/

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libb64-0d \
    libre2-5 \
    libssl1.1 \
    rapidjson-dev \
    libopenblas-dev \
    libarchive-dev \
    zlib1g \
    wget \
    python3 \
    python3-dev \
    python3-pip

WORKDIR /opt/tritonserver

RUN wget https://github.com/triton-inference-server/server/releases/download/v2.33.0/tritonserver2.33.0-jetpack5.1.tgz && \
    tar -xzf tritonserver2.33.0-jetpack5.1.tgz && \
    rm tritonserver2.33.0-jetpack5.1.tgz

ENV PATH=$PATH:/opt/tritonserver/bin
