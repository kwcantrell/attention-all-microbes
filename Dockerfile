# Use Python 3.9 base image with CUDA 11.8 and cuDNN 8 support for GPU
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 AS base

# Miniconda installation
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init bash

# Set PATH to include conda
ENV PATH="/opt/conda/bin:${PATH}"

# Install Unifrac, TensorFlow, tf-models-official, TensorBoard, and other dependencies via conda
RUN /opt/conda/bin/conda create --name aam -c conda-forge -c bioconda unifrac=1.1.0 python=3.9 cython && \
    /opt/conda/bin/conda install -n aam -c conda-forge tensorflow-gpu tf-models-official>=2.14.2 tensorboard && \
    conda clean --all && \
    echo "source activate aam" > ~/.bashrc

# Stage 2: Application Layer
FROM base AS application

# Install git
RUN apt-get update && apt-get install -y git

# Clone the specific branch of the repository
ARG BRANCH=main
RUN git clone --depth=1 --branch $BRANCH https://github.com/kwcantrell/attention-all-microbes.git /app/attention-all-microbes

# Install the repository in editable mode, but skip installing dependencies already handled by conda
WORKDIR /app/attention-all-microbes
RUN pip install -e .

# Cleanup after installation
RUN rm -rf /root/.cache/pip && \
    apt-get remove --purge -y git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /app/.git

# Default command to run the 'attention' CLI
CMD ["/bin/bash", "-c", "source activate aam && attention"]

