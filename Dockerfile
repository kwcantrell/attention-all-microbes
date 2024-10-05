# Use the official TensorFlow GPU image with TensorFlow 2.14.0 and CUDA
FROM tensorflow/tensorflow:2.14.0-gpu as base

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Miniconda and necessary dependencies in a single step, and clean up apt cache
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    /opt/conda/bin/conda init bash

# Set PATH to include conda
ENV PATH="/opt/conda/bin:${PATH}"

# Install Unifrac and other dependencies via conda in one step and clean up conda cache
RUN /opt/conda/bin/conda create --name aam -c conda-forge -c bioconda unifrac python=3.9 cython && \
    conda clean --all

# Install tf-models-official and tensorboard using pip in the conda environment, and clean pip cache
RUN /opt/conda/bin/conda run -n aam pip install --no-cache-dir tf-models-official==2.14.2 tensorboard

# Stage 2: Application Layer
FROM base AS application

# Install git and clean up apt cache in one step
RUN apt-get update && apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone the specific branch of the repository and install it in editable mode
ARG BRANCH=dockerize
RUN git clone --depth=1 --branch $BRANCH https://github.com/kwcantrell/attention-all-microbes.git /app/attention-all-microbes

# Install the repository in editable mode using pip within the conda environment
WORKDIR /app/attention-all-microbes
RUN /opt/conda/bin/conda run -n aam pip install --no-cache-dir -e .

# Cleanup unnecessary files after installation
RUN rm -rf /root/.cache/pip /app/.git

# Default command to run the 'attention' CLI
CMD ["/opt/conda/bin/conda", "run", "-n", "aam", "/opt/conda/envs/aam/bin/attention"]

