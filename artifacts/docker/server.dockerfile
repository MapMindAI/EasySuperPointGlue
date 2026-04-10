FROM nvcr.io/nvidia/tritonserver:25.05-py3

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

ENV LD_LIBRARY_PATH=/opt/tritonserver/backends/pytorch:$LD_LIBRARY_PATH
