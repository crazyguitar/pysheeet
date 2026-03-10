ARG CUDA_VERSION=12.8.1
ARG GDRCOPY_VERSION=v2.5.1
ARG EFA_INSTALLER_VERSION=1.47.0
ARG UCX_VERSION=v1.20.0
ARG NIXL_VERSION=0.10.1
ARG CUDA_ARCH=90

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ARG GDRCOPY_VERSION
ARG EFA_INSTALLER_VERSION
ARG UCX_VERSION
ARG NIXL_VERSION
ARG CUDA_ARCH

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get remove -y --allow-change-held-packages \
    ibverbs-utils libibverbs-dev libibverbs1 libmlx5-1

RUN rm -rf /opt/hpcx /usr/local/mpi /etc/ld.so.conf.d/hpcx.conf && ldconfig

ENV OPAL_PREFIX=

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    dmidecode \
    environment-modules \
    etcd-server \
    ethtool \
    git \
    ibverbs-providers \
    ibverbs-utils \
    iproute2 \
    jq \
    libcpprest-dev \
    libcurl4-openssl-dev \
    libelf-dev \
    libevent-core-2.1-7t64 \
    libevent-pthreads-2.1-7t64 \
    libgflags-dev \
    libgrpc++-dev \
    libgrpc-dev \
    libhwloc-dev \
    libhwloc15 \
    libibumad-dev \
    libibverbs-dev \
    libnl-3-200 \
    libnl-3-dev \
    libnl-route-3-200 \
    libnl-route-3-dev \
    libnuma-dev \
    libprotobuf-dev \
    librdmacm-dev \
    libssl-dev \
    libtool \
    meson \
    ninja-build \
    openssh-client \
    openssh-server \
    pciutils \
    pkg-config \
    protobuf-compiler-grpc \
    pybind11-dev \
    python3 \
    python3-dev \
    python3-pip \
    rdma-core \
    tcl \
    udev \
    uuid-dev \
    vim \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get purge -y cuda-compat-* || true

RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/gdrcopy/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED \
    && pip3 install --no-cache-dir awscli nvidia-ml-py

# --- GDRCopy ---
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install \
    && rm -rf /tmp/gdrcopy

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:$LIBRARY_PATH
ENV CPATH=/opt/gdrcopy/include
ENV PATH=/opt/gdrcopy/bin:$PATH

# --- EFA ---
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz

# --- UCX (verbs + rdmacm + dm + efa) ---
ENV UCX_PREFIX=/usr/local/ucx

RUN git clone --depth 1 --branch ${UCX_VERSION} https://github.com/openucx/ucx.git /tmp/ucx \
    && cd /tmp/ucx && ./autogen.sh \
    && ./contrib/configure-release-mt \
       --prefix=${UCX_PREFIX} \
       --enable-shared --disable-static \
       --enable-optimizations --enable-cma --enable-mt \
       --enable-devel-headers \
       --with-cuda=/usr/local/cuda \
       --with-gdrcopy=/opt/gdrcopy \
       --with-verbs --with-rdmacm --with-dm --with-efa \
    && make -j$(nproc) && make install \
    && echo "${UCX_PREFIX}/lib" > /etc/ld.so.conf.d/ucx.conf \
    && echo "${UCX_PREFIX}/lib/ucx" >> /etc/ld.so.conf.d/ucx.conf \
    && ldconfig && rm -rf /tmp/ucx

ENV PATH="${UCX_PREFIX}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${UCX_PREFIX}/lib:${UCX_PREFIX}/lib/ucx:${LD_LIBRARY_PATH}"

# --- PyTorch (install before NIXL, needed for Python bindings) ---
RUN pip3 install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# --- NIXL ---
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN git clone --depth 1 --branch ${NIXL_VERSION} \
      https://github.com/ai-dynamo/nixl.git /opt/nixl \
    && cd /opt/nixl \
    && pip3 install --no-cache-dir tomlkit \
    && export PKG_CONFIG_PATH="/opt/amazon/efa/lib/pkgconfig:$PKG_CONFIG_PATH" \
    && export CPATH="/opt/amazon/efa/include:$CPATH" \
    && export LIBRARY_PATH="/opt/amazon/efa/lib:/usr/local/cuda/lib64/stubs" \
    && meson setup build \
       --prefix=/usr/local \
       --buildtype=release \
       -Ducx_path=${UCX_PREFIX} \
       -Dlibfabric_path=/opt/amazon/efa \
    && ninja -C build -j$(nproc) \
    && ninja -C build install \
    && pip3 install --no-cache-dir build/src/bindings/python/nixl-meta/nixl-${NIXL_VERSION}-py3-none-any.whl \
    && ldconfig

# --- nixlbench ---

# etcd-cpp-api (nixlbench dep)
RUN git clone --depth 1 https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git /tmp/etcd-cpp \
    && cd /tmp/etcd-cpp && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) && make install && ldconfig \
    && rm -rf /tmp/etcd-cpp

RUN cd /opt/nixl/benchmark/nixlbench \
    && meson setup build --prefix=/usr/local --buildtype=release -Dnixl_path=/usr/local \
    && ninja -C build -j$(nproc) && ninja -C build install

# --- kvbench (Python-based, lives in /opt/nixl/benchmark/kvbench) ---
RUN pip3 install --no-cache-dir pyYAML click tabulate

# --- Environment ---
ENV OPAL_PREFIX=/opt/amazon/openmpi \
    OMPI_MCA_pml=ob1 \
    OMPI_MCA_btl="tcp,self" \
    FI_PROVIDER=efa \
    FI_EFA_FORK_SAFE=1 \
    FI_EFA_USE_DEVICE_RDMA=1 \
    FI_LOG_LEVEL=warn \
    UCX_TLS="^cuda_ipc" \
    UCX_NET_DEVICES=all \
    PYTHONPATH="/opt/nixl/benchmark"

# --- Validation ---
RUN echo "=== NIXL Image Validation ===" \
    && ucx_info -v | head -3 \
    && fi_info --version \
    && ${OPAL_PREFIX}/bin/mpirun --version | head -1 \
    && which ucx_perftest \
    && which nixlbench \
    && python3 -c "import nixl; print('nixl OK')"

# --- flash-attn + DeepGEMM ---
RUN apt-get update && apt-get remove -y python3-blinker && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir packaging numpy ninja pybind11 "setuptools<80" wheel
RUN pip3 install --no-cache-dir flash-attn==2.8.1 --no-build-isolation
RUN git clone --recursive -b v2.1.1.post3 https://github.com/deepseek-ai/DeepGEMM.git /tmp/deepgemm \
    && cd /tmp/deepgemm \
    && python3 setup.py bdist_wheel \
    && pip3 install dist/*.whl \
    && rm -rf /tmp/deepgemm

# --- vLLM + SGLang + Ray ---
ENV VLLM_USE_DEEP_GEMM=1
ENV DG_JIT_CACHE_DIR=/tmp
RUN pip3 install --no-cache-dir vllm==0.15.1
RUN pip3 install --no-cache-dir "sglang[all]==0.5.9"
RUN pip3 install --no-cache-dir ray

# --- vllm-router (PD disaggregation proxy) ---
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip3 install --no-cache-dir vllm-router

WORKDIR /workspace
CMD ["/bin/bash"]
