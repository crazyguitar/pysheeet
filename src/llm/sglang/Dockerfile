ARG SGLANG_VERSION=0.5.8
ARG CUDA_VERSION=12.8.1
ARG GDRCOPY_VERSION=v2.5.1
ARG EFA_INSTALLER_VERSION=1.46.0
ARG NCCL_VERSION=v2.29.2-1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ARG GDRCOPY_VERSION
ARG EFA_INSTALLER_VERSION
ARG NCCL_VERSION
ARG SGLANG_VERSION

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
    libnuma-dev \
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
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/gdrcopy/lib:/usr/local/lib:$LD_LIBRARY_PATH
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

# Install SGLang (with CUDA 12.8 wheels)
RUN pip3 install --no-cache-dir "sglang[all]==${SGLANG_VERSION}" \
    --find-links https://docs.sglang.ai/whl/cu128/

# Install Nsight Systems for profiling
RUN apt-get update -y && apt-get install -y --no-install-recommends gnupg \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
    && echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2404/$(dpkg --print-architecture) /" \
       > /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80 \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
