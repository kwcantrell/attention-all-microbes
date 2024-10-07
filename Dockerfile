# Stage 1: Base Image for building environment
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, Miniconda, and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget bzip2 git make gcc build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -a -y

# Set PATH to include conda
ENV PATH="/opt/conda/bin:${PATH}"

# Create conda environment and install dependencies
RUN conda create --name aam -c conda-forge -c bioconda \
    h5py=3.3.0 \
    gxx_linux-64 \
    unifrac \
    python=3.9 \
    cython --yes && \
    conda clean --packages --tarballs --yes

RUN ln -s /opt/conda/envs/aam/lib/libhdf5_cpp.so.200 /opt/conda/envs/aam/lib/libhdf5_cpp.so.103 && \ 
    ln -s /opt/conda/envs/aam/lib/libhdf5_hl_cpp.so.200 /opt/conda/envs/aam/lib/libhdf5_hl_cpp.so.100 && \ 
    ln -s /opt/conda/envs/aam/lib/libhdf5.so.200 /opt/conda/envs/aam/lib/libhdf5.so.100 && \
    ln -s /opt/conda/envs/aam/lib/libhdf5_hl.so.200 /opt/conda/envs/aam/lib/libhdf5_hl.so.100 && \
    ln -s /opt/conda/envs/aam/lib/libhdf5.so.200 /opt/conda/envs/aam/lib/libhdf5.so.103

# Install TensorFlow and other Python dependencies in the conda environment using pip
RUN conda run -n aam pip install \
    tensorflow==2.14.0 \
    iow \
    tf-models-official==2.14.2 \
    tensorboard

# Stage 2: Final Image for running the application
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime

# Copy Miniconda from base
COPY --from=base /opt/conda /opt/conda

# Set PATH to include conda and set LD_LIBRARY_PATH
ENV PATH="/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Clone the repository and install it
ARG BRANCH=dockerize
RUN apt-get update && apt-get install -y --no-install-recommends git && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    git clone --depth=1 --branch $BRANCH https://github.com/kwcantrell/attention-all-microbes.git /app/attention-all-microbes

WORKDIR /app/attention-all-microbes

# Install the repository in editable mode within the conda environment
RUN conda run -n aam pip install --no-cache-dir -e .

# Default command to run the 'attention' CLI using full path
CMD ["/opt/conda/bin/conda", "run", "-n", "aam", "attention"]

