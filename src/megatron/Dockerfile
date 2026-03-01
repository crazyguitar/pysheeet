# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Modifications copyright (c) 2025 chang-ning
# Modifications licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
# See LICENSE or https://creativecommons.org/licenses/by/4.0/

ARG CUDA_VERSION=12.8.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ARG GDRCOPY_VERSION=v2.5.1
ARG EFA_INSTALLER_VERSION=1.47.0
ARG AWS_OFI_NCCL_VERSION=5f4202f11db1585d878196db4430aeda0e834a0c
ARG NCCL_VERSION=v2.29.3-1
ARG NCCL_TESTS_VERSION=v2.17.9
ARG NVSHMEM_VERSION=v3.5.19-1
ARG TORCH_VERSION=2.9.1
ARG MEGATRON_BRIDGE_VERSION=v0.2.2

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get remove -y --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1 \
    libnccl2 \
    libnccl-dev

RUN rm -rf /opt/hpcx \
    && rm -rf /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

ENV OPAL_PREFIX=

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    autoconf \
    automake \
    build-essential \
    check \
    cmake \
    ninja-build \
    curl \
    debhelper \
    devscripts \
    git \
    gcc \
    gdb \
    kmod \
    libsubunit-dev \
    libtool \
    openssh-client \
    openssh-server \
    pkg-config \
    vim \
    hwloc \
    libhwloc-dev \
    python3-dev \
    python3-venv \
    libomp-dev

RUN apt-get purge -y cuda-compat-*

RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# Set paths for both aarch64 and x86_64
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

RUN apt-get install -y python3-pip \
    && pip3 install --break-system-packages --no-cache-dir awscli nvidia-ml-py Cython

#################################################
## Install NVIDIA GDRCopy
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install \
    && rm -rf /tmp/gdrcopy

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:$LIBRARY_PATH
ENV CPATH=/opt/gdrcopy/include:$CPATH
ENV PATH=/opt/gdrcopy/bin:$PATH

#################################################
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify --skip-plugin \
    && rm -rf $HOME/aws-efa-installer $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz

###################################################
## Install aws-ofi-nccl from source (pinned commit)
RUN git clone https://github.com/aws/aws-ofi-nccl.git /tmp/aws-ofi-nccl \
    && cd /tmp/aws-ofi-nccl \
    && git checkout ${AWS_OFI_NCCL_VERSION} \
    && ./autogen.sh \
    && ./configure --prefix=/opt/amazon/ofi-nccl \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --with-nvtx=/usr/local/cuda \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/aws-ofi-nccl

###################################################
## Install NCCL
RUN git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git  /opt/nccl \
    && cd /opt/nccl \
    && make -j $(nproc) src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

###################################################
## Install NCCL-tests
RUN git clone -b ${NCCL_TESTS_VERSION} https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && make -j $(nproc) \
    MPI=1 \
    MPI_HOME=/opt/amazon/openmpi/ \
    CUDA_HOME=/usr/local/cuda \
    NCCL_HOME=/opt/nccl/build \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

###################################################
## Build NCCL Device API examples
RUN cd /opt/nccl/examples/06_device_api \
    && make -j $(nproc) NCCL_HOME=/opt/nccl/build CUDA_HOME=/usr/local/cuda MPI=1 MPI_HOME=/opt/amazon/openmpi

###################################################
## Install NVSHMEM
ENV NVSHMEM_DIR=/opt/nvshmem
ENV NVSHMEM_HOME=/opt/nvshmem
RUN git clone -b ${NVSHMEM_VERSION} https://github.com/NVIDIA/nvshmem.git \
  && cd nvshmem \
  && mkdir -p build \
  && cd build \
  && cmake -DNVSHMEM_PREFIX=/opt/nvshmem \
    -DCMAKE_CUDA_ARCHITECTURES="80;90" \
    -DNVSHMEM_MPI_SUPPORT=1 \
    -DNVSHMEM_PMIX_SUPPORT=1 \
    -DNVSHMEM_LIBFABRIC_SUPPORT=1 \
    -DNVSHMEM_IBRC_SUPPORT=1 \
    -DNVSHMEM_IBGDA_SUPPORT=1 \
    -DNVSHMEM_USE_GDRCOPY=1 \
    -DNVSHMEM_BUILD_TESTS=1 \
    -DNVSHMEM_BUILD_EXAMPLES=1 \
    -DNVSHMEM_BUILD_HYDRA_LAUNCHER=1 \
    -DNVSHMEM_BUILD_TXZ_PACKAGE=0 \
    -DNVSHMEM_BUILD_PYTHON_LIB=0 \
    -DMPI_HOME=/opt/amazon/openmpi \
    -DPMIX_HOME=/opt/amazon/pmix \
    -DGDRCOPY_HOME=/opt/gdrcopy \
    -DLIBFABRIC_HOME=/opt/amazon/efa \
    -G Ninja .. \
  && ninja -j $(nproc) \
  && ninja install \
  && rm -rf /root/nvshmem

RUN pip3 install --break-system-packages --no-cache-dir nvshmem4py-cu12

ENV LD_LIBRARY_PATH=/opt/amazon/pmix/lib:/opt/nvshmem/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/nvshmem/bin:$PATH
ENV NVSHMEM_REMOTE_TRANSPORT=libfabric
ENV NVSHMEM_LIBFABRIC_PROVIDER=efa

###################################################
## Install PyTorch
RUN pip3 install --break-system-packages --no-cache-dir torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu128

###################################################
## Install DeepEP with NCCL GIN backend
RUN unset NVSHMEM_DIR NVSHMEM_HOME \
    && export ENABLE_NCCL=1 \
    && export NCCL_DIR=/opt/nccl/build \
    && export LD_LIBRARY_PATH=/opt/nccl/build/lib:$LD_LIBRARY_PATH \
    && export LD_PRELOAD=/opt/nccl/build/lib/libnccl.so.2 \
    && git clone -b nccl https://github.com/aamirshafi/DeepEP.git /opt/DeepEP \
    && cd /opt/DeepEP \
    && git checkout 6d29f34 \
    && python3 setup.py build_ext --inplace \
    && pip install --break-system-packages --no-build-isolation .

RUN rm -rf /var/lib/apt/lists/*

## Set Open MPI variables
ENV OMPI_MCA_pml=^ucx            \
    OMPI_MCA_btl=tcp,self           \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent\
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth

ENV FI_EFA_USE_DEVICE_RDMA=1
ENV FI_PROVIDER=efa
ENV FI_EFA_FORK_SAFE=1
ENV NCCL_BUFFSIZE=8388608
ENV NCCL_P2P_NET_CHUNKSIZE=524288
ENV NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-tuner-ofi.so

## Turn off PMIx Error
ENV PMIX_MCA_gds=hash

## Set LD_PRELOAD for NCCL library
ENV LD_PRELOAD=/opt/nccl/build/lib/libnccl.so

# NVSHMEM additional settings
ENV NVSHMEM_DISABLE_CUDA_VMM=1

# Install Nsight Systems for profiling
RUN apt-get update -y && apt-get install -y --no-install-recommends gnupg \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
    && echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2404/$(dpkg --print-architecture) /" \
       > /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80 \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

###################################################
## Install cuDNN (headers + libs needed for TransformerEngine build)
RUN apt-get update -y && apt-get install -y --no-install-recommends libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

###################################################
## Build tools and pinned numpy (before TE build)
ENV TORCH_CUDA_ARCH_LIST="9.0;10.0"
ENV NVTE_FRAMEWORK=pytorch
ENV NVTE_CUDA_ARCHS="90;100"
ENV NCCL_DIR=/opt/nccl/build
ENV NCCL_INCLUDE_DIR=/opt/nccl/build/include
ENV NCCL_LIB_DIR=/opt/nccl/build/lib
ENV CPATH=/opt/nccl/build/include:${CPATH}
RUN pip3 install --break-system-packages --no-cache-dir \
    numpy ninja pybind11 packaging "setuptools<80" wheel

###################################################
## Pre-install CUDA extensions against current torch
RUN pip3 install --break-system-packages --no-cache-dir --no-build-isolation causal-conv1d==1.6.0 mamba-ssm==2.3.0

###################################################
## Install Megatron-Bridge
RUN apt-get update && apt-get remove -y python3-blinker && rm -rf /var/lib/apt/lists/*
RUN git clone -b ${MEGATRON_BRIDGE_VERSION} --depth 1 https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /tmp/Megatron-Bridge \
    && cd /tmp/Megatron-Bridge \
    && pip install --break-system-packages --no-cache-dir --no-build-isolation "torch==${TORCH_VERSION}" . \
    && rm -rf /tmp/Megatron-Bridge

RUN pip install --break-system-packages --no-cache-dir viztracer

# Sanity check
RUN python3 -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.version.cuda}')" \
    && python3 -c "import transformer_engine; print(f'TE={transformer_engine.__version__}')" \
    && python3 -c "import megatron.core; print('megatron-core OK')" \
    && python3 -c "from megatron.bridge import AutoBridge; print('megatron-bridge OK')"

WORKDIR /workspace
