ARG PYTHON_VERSION


FROM tataucloud/python-cuda:${PYTHON_VERSION}
# FROM nvidia/cuda:9.0-devel-ubuntu16.04

ARG PYTHON_VERSION

LABEL maintainer="tatau.io"

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=1.12.0
ENV PYTORCH_VERSION=1.0.0
ENV CUDNN_VERSION=7.4.1.5-1+cuda9.0
ENV NCCL_VERSION=2.4.2-1+cuda9.0
ENV MXNET_URL=https://s3-us-west-2.amazonaws.com/mxnet-python-packages-gcc5/mxnet_cu90_gcc5-1.4.0-py2.py3-none-manylinux1_x86_64.whl
ENV DEBIAN_FRONTEND=noninteractive
ENV OFED_VER=4.5-1.0.1.0

#ENV http_proxy=http://172.17.0.1:3128 \
#    https_proxy=http://172.17.0.1:3128 \
#    ftp_proxy=http://172.17.0.1:3128
# RUN apt update && apt install -y


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
        lsb-core \
        unzip nano git \
        autotools-dev dkms debhelper \
        libelf-dev libnuma-dev \
        && rm -rf /var/lib/apt/lists/* && mkdir -p /var/run/sshd


RUN mkdir /tmp/nvidia-peer-memory && \
    cd /tmp/nvidia-peer-memory && \
    wget http://content.mellanox.com/ofed/nvidia-peer-memory_1.0-7.tar.gz && \
    mv nvidia-peer-memory_1.0-7.tar.gz nvidia-peer-memory_1.0.orig.tar.gz && \
    tar -xf nvidia-peer-memory_1.0.orig.tar.gz && cd /tmp/nvidia-peer-memory/nvidia-peer-memory-1.0 && \
    pwd && \
    ls -alh && \
    ./build_module.sh && \
    dpkg-buildpackage -us -uc && \
    dpkg -i ../nvidia-peer-memory_1.0-7_all.deb ../nvidia-peer-memory-dkms_1.0-7_all.deb && \
    cd / && \
    rm -rf /tmp/nvidia-peer-memory


RUN wget http://content.mellanox.com/ofed/MLNX_OFED-${OFED_VER}/MLNX_OFED_LINUX-${OFED_VER}-ubuntu16.04-x86_64.tgz && \
    tar -xzvf MLNX_OFED_LINUX-${OFED_VER}-ubuntu16.04-x86_64.tgz && \
    rm -rf MLNX_OFED_LINUX-${OFED_VER}-ubuntu16.04-x86_64.tgz && \
    MLNX_OFED_LINUX-${OFED_VER}-ubuntu16.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --all -q && \
    cd .. && \
    rm -rf *.tgz

RUN mkdir /tmp/ucx && \
    cd /tmp/ucx/ && \
    wget "https://github.com/openucx/ucx/releases/download/v1.5.0/ucx-1.5.0.tar.gz" && \
    tar xzf ucx-1.5.0.tar.gz && \
    cd ucx-1.5.0 && \
    ./contrib/configure-release --prefix=$PWD/install --with-verbs --with-mlx5-dv --with-dm --with-cuda= && \
    make -j8 && \
    make install && rm -rf /tmp/ucx


# http://www.openucx.org/wp-content/uploads/2015/03/UCX_install_guide_v1.5.0.pdf
# https://github.com/openucx/ucx/wiki/OpenMPI-and-OpenSHMEM-installation-with-UCX#running-open-mpi-with-ucx
# https://www.open-mpi.org/faq/?category=building#build-p2p
# https://www.open-mpi.org/faq/?category=openfabrics#run-ucx
# https://github.com/aws/aws-ofi-nccl

# RUN apt install  # libmlx4-dev #libmlx5-dev

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v3.1/downloads/openmpi-3.1.2.tar.gz && \
    tar zxf openmpi-3.1.2.tar.gz && \
    cd openmpi-3.1.2 && \
    ./configure --enable-orterun-prefix-by-default --with-verbs --with-ucx --with-cuda && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi


#RUN apt install -y --no-install-recommends nvidia-410-dev && \
#    mkdir /tmp/gdrcopy && cd /tmp/gdrcopy && wget https://github.com/NVIDIA/gdrcopy/archive/v1.3.tar.gz && \
#    tar -xf v1.3.tar.gz && cd gdrcopy-1.3 && make all install && ./insmod.sh && \
#    export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH && ./validate
#
#
#
#




RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip install 'numpy<1.15.0' tensorflow-gpu==${TENSORFLOW_VERSION} keras h5py torch==${PYTORCH_VERSION} torchvision ${MXNET_URL}


# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
    git clone --recurse-submodules https://github.com/horovod/horovod /horovod && \
    cd /horovod && git checkout v0.16.0 && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 python setup.py sdist && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install -v /horovod/dist/horovod-*.tar.gz && \
    ldconfig
#    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
#    ldconfig


# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
    chmod a+x /usr/local/bin/mpirun


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


# RUN service nv_peer_mem status
#
#
#apt install libelf-dev

#RUN apt-get install -y --no-install-recommends ibverbs-utils ibutils perftest libibverbs-dev


# https://community.mellanox.com/s/article/linux-sysctl-tuning


# MVAPICH: MPI over InfiniBand, Omni-Path, Ethernet/iWARP, and RoCE
# http://mvapich.cse.ohio-state.edu/benchmarks/
# http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-5.6.tar.gz