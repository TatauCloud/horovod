ARG PYTHON_VERSION

FROM tataucloud/python-cuda:${PYTHON_VERSION}

LABEL maintainer="tatau.io"

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=1.12.0
ENV PYTORCH_VERSION=1.0.0
ENV CUDNN_VERSION=7.4.1.5-1+cuda9.0
ENV NCCL_VERSION=2.3.7-1+cuda9.0
ENV KERAS_VERSION=2.2.4
ENV H5PY_VERSION=2.8.0
ENV TORCHVISION_VERSION=0.2.1
ENV HOROVOD_VERSION=0.15.2
ARG HOROVOD_GPU_ALLREDUCE


RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        libncurses5-dev \
        libncursesw5-dev \
        tk-dev \
        libbz2-dev \
        libssl-dev \
        libreadline-dev \
        libfreetype6-dev \
        libffi-dev \
        openssh-client openssh-server \
        mpich libmpich-dev \
        && rm -rf /var/lib/apt/lists/* && mkdir -p /var/run/sshd

# Install TensorFlow, Keras and PyTorch
RUN pip install  --no-cache-dir \
    tensorflow-gpu==${TENSORFLOW_VERSION} \
    keras==${KERAS_VERSION} \
    h5py==${H5PY_VERSION} \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION}

# Install Open MPI
#RUN mkdir /tmp/openmpi && \
#    cd /tmp/openmpi && \
#    wget https://www.open-mpi.org/software/ompi/v3.1/downloads/openmpi-3.1.2.tar.gz && \
#    tar zxf openmpi-3.1.2.tar.gz && \
#    cd openmpi-3.1.2 && \
#    ./configure --enable-orterun-prefix-by-default && \
#    make -j $(nproc) all && \
#    make install && \
#    ldconfig && \
#    rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=$HOROVOD_GPU_ALLREDUCE HOROVOD_WITH_TENSORFLOW=1 \
    HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig

# Create a wrapper for OpenMPI to allow running as root by default

#RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
#    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
#    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
#    chmod a+x /usr/local/bin/mpirun

# Replace /usr/local/bin by /usr/bin because mpich used from apt
RUN mv /usr/bin/mpirun /usr/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/bin/mpirun && \
    chmod a+x /usr/bin/mpirun

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf
    # Conflicted with option btl_tcp_if_include
    # echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf

# Set default NCCL parameters
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Download examples
#RUN apt-get install -y --no-install-recommends subversion && \
#    svn checkout https://github.com/uber/horovod/trunk/examples && \
#    rm -rf /examples/.svn

# WORKDIR "/examples"
