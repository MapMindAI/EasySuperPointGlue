FROM nvcr.io/nvidia/tritonserver:23.12-py3

COPY installers/apt_install_clean.sh /usr/local/bin/

RUN apt_install_clean.sh \
    libgl1-mesa-dev \
    libglew-dev \
    unzip \
    cmake

# Compile opencv.
# COPY installers/opencv.sh /tmp/installers/
# RUN bash /tmp/installers/opencv.sh && rm /tmp/installers/opencv.sh

# cnmem, for triton
RUN cd /tmp && \
    git clone https://github.com/NVIDIA/cnmem.git && \
    cd cnmem &&\
    git checkout 37896cc9bfc6536a8c878a1e675835c22d827821 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install &&\
    rm -r /tmp/cnmem

# Install TensorRT based on architecture
RUN if [ "$ARCH" = "x86_64" ]; then \
        echo "Installing TensorRT 10.13.0.35-1+cuda12.9 for x86_64"; \
        apt-get update && apt-get install -y --no-install-recommends \
            libnvinfer10=10.13.0.35-1+cuda12.9 \
            libnvinfer-plugin10=10.13.0.35-1+cuda12.9 \
            libnvinfer-lean10=10.13.0.35-1+cuda12.9 \
            libnvinfer-vc-plugin10=10.13.0.35-1+cuda12.9 \
            libnvinfer-dispatch10=10.13.0.35-1+cuda12.9 \
            libnvonnxparsers10=10.13.0.35-1+cuda12.9 \
            libnvinfer-bin=10.13.0.35-1+cuda12.9 \
            libnvinfer-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-headers-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-plugin-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-headers-plugin-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-lean-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-vc-plugin-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-dispatch-dev=10.13.0.35-1+cuda12.9 \
            libnvonnxparsers-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-samples=10.13.0.35-1+cuda12.9 \
            libnvinfer-win-builder-resource10=10.13.0.35-1+cuda12.9 \
            python3-libnvinfer=10.13.0.35-1+cuda12.9 \
            python3-libnvinfer-dispatch=10.13.0.35-1+cuda12.9 \
            python3-libnvinfer-lean=10.13.0.35-1+cuda12.9 \
            python3-libnvinfer-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-headers-python-plugin-dev=10.13.0.35-1+cuda12.9 \
        && rm -rf /var/lib/apt/lists/*; \
    elif [ "$ARCH" = "aarch64" ]; then \
        echo "Installing TensorRT 10.13.2.6-1+cuda13.0 for aarch64"; \
        apt-get update && apt-get install -y --no-install-recommends \
            tensorrt=10.3.0.30-1+cuda12.5 \
        && rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Unsupported architecture: $arch"; exit 1; \
    fi


ENV LD_LIBRARY_PATH=/opt/tritonserver/backends/pytorch:$LD_LIBRARY_PATH
