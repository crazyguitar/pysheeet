ARG VLLM_VERSION=0.15.1
ARG CUDA_VERSION=12.8.1
ARG GDRCOPY_VERSION=v2.5.1
ARG EFA_INSTALLER_VERSION=1.46.0
ARG NCCL_VERSION=v2.29.2-1
ARG NVSHMEM_VERSION=v3.5.19-1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ARG GDRCOPY_VERSION
ARG EFA_INSTALLER_VERSION
ARG NCCL_VERSION
ARG NVSHMEM_VERSION
ARG VLLM_VERSION

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Update and remove conflicting packages
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get remove -y --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1 \
    libnccl2 \
    libnccl-dev

# Clean up existing MPI installations
RUN rm -rf /opt/hpcx \
    && rm -rf /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

ENV OPAL_PREFIX=

# Install build dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    autoconf \
    automake \
    build-essential \
    check \
    cmake \
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
    python3 \
    python3-dev \
    python3-pip \
    vim \
    wget \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Remove cuda-compat if present
RUN apt-get purge -y cuda-compat-* || true

# Configure SSH
RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo " UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# Set library paths
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/gdrcopy/lib:/opt/nvshmem/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/opt/gdrcopy/bin:/usr/bin:/usr/local/bin:$PATH

# Remove PEP 668 restriction and install packages
RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED \
    && pip3 install --no-cache-dir awscli nvidia-ml-py Cython

# Install GDRCopy
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install \
    && rm -rf /tmp/gdrcopy

ENV LIBRARY_PATH=/opt/gdrcopy/lib:${LIBRARY_PATH:-}
ENV CPATH=/opt/gdrcopy/include

# Install EFA dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    pciutils \
    environment-modules \
    tcl \
    libnl-3-200 \
    libnl-3-dev \
    libnl-route-3-200 \
    libnl-route-3-dev \
    udev \
    dmidecode \
    ethtool \
    iproute2 \
    libevent-core-2.1-7t64 \
    libevent-pthreads-2.1-7t64 \
    libhwloc15 \
    && rm -rf /var/lib/apt/lists/*

# Install EFA
RUN cd /tmp \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf /tmp/aws-efa-installer*

# Install NCCL
RUN git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git /tmp/nccl \
    && cd /tmp/nccl \
    && make -j $(nproc) src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" \
    && mkdir -p /opt/nccl/build/lib \
    && cp -r build/lib/* /opt/nccl/build/lib/ \
    && cp -r build/include /opt/nccl/build/ \
    && rm -rf /tmp/nccl

# Install NVSHMEM
ENV NVSHMEM_DIR=/opt/nvshmem
ENV NVSHMEM_HOME=/opt/nvshmem
RUN git clone https://github.com/NVIDIA/nvshmem.git /tmp/nvshmem \
    && cd /tmp/nvshmem \
    && git checkout ${NVSHMEM_VERSION} \
    && mkdir -p build \
    && cd build \
    && cmake -DNVSHMEM_PREFIX=/opt/nvshmem \
        -DCMAKE_CUDA_ARCHITECTURES="90" \
        -DNVSHMEM_MPI_SUPPORT=1 \
        -DNVSHMEM_PMIX_SUPPORT=1 \
        -DNVSHMEM_LIBFABRIC_SUPPORT=1 \
        -DNVSHMEM_IBRC_SUPPORT=1 \
        -DNVSHMEM_IBGDA_SUPPORT=1 \
        -DNVSHMEM_USE_GDRCOPY=1 \
        -DNVSHMEM_BUILD_TESTS=0 \
        -DNVSHMEM_BUILD_EXAMPLES=0 \
        -DNVSHMEM_BUILD_HYDRA_LAUNCHER=0 \
        -DNVSHMEM_BUILD_TXZ_PACKAGE=0 \
        -DNVSHMEM_BUILD_PYTHON_LIB=0 \
        -DMPI_HOME=/opt/amazon/openmpi \
        -DPMIX_HOME=/opt/amazon/pmix \
        -DGDRCOPY_HOME=/opt/gdrcopy \
        -DLIBFABRIC_HOME=/opt/amazon/efa \
        -G Ninja .. \
    && ninja -j $(nproc) \
    && ninja install \
    && rm -rf /tmp/nvshmem

# OpenMPI settings
ENV OMPI_MCA_pml=^ucx
ENV OMPI_MCA_btl=tcp,self
ENV OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent
ENV OPAL_PREFIX=/opt/amazon/openmpi
ENV PMIX_MCA_gds=hash

# NCCL settings
ENV NCCL_DEBUG=INFO
ENV NCCL_SOCKET_IFNAME=^docker,lo,veth
ENV NCCL_P2P_NET_CHUNKSIZE=524288
ENV NCCL_BUFFSIZE=8388608
ENV NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-tuner-ofi.so
ENV LD_PRELOAD=/opt/nccl/build/lib/libnccl.so

# EFA settings
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV FI_EFA_FORK_SAFE=1
ENV RDMAV_FORK_SAFE=1

# vLLM settings
ENV VLLM_RPC_TIMEOUT=3600000
ENV VLLM_ENGINE_READY_TIMEOUT_S=3600
ENV VLLM_USE_DEEP_GEMM=1
ENV DG_JIT_CACHE_DIR=/tmp

# NVSHMEM settings
ENV NVSHMEM_REMOTE_TRANSPORT=libfabric
ENV NVSHMEM_LIBFABRIC_PROVIDER=efa
ENV NVSHMEM_DISABLE_CUDA_VMM=1

# Install vLLM
RUN pip3 install --no-cache-dir vllm==${VLLM_VERSION}

# Install Nsight Systems for profiling
RUN apt-get update -y && apt-get install -y --no-install-recommends gnupg \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
    && echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2404/$(dpkg --print-architecture) /" \
       > /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80 \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

# Install DeepGEMM (requires torch from vLLM)
RUN git clone --recursive -b v2.1.1.post3 https://github.com/deepseek-ai/DeepGEMM.git /tmp/deepgemm \
    && cd /tmp/deepgemm \
    && python3 setup.py bdist_wheel \
    && pip3 install dist/*.whl \
    && rm -rf /tmp/deepgemm

# Copy run.sh script
COPY run.sh /workspace/run.sh
RUN chmod +x /workspace/run.sh

WORKDIR /workspace
