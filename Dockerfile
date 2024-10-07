# Use the official TensorFlow GPU image with TensorFlow 2.14.0 and CUDA
FROM tensorflow/tensorflow:2.14.0-gpu as base

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Miniconda
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set PATH to include conda
ENV PATH="/opt/conda/bin:${PATH}"

# Set the LD_LIBRARY_PATH to ensure the HDF5 libraries from the system are available
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Create conda environment and install Unifrac with specific versions
RUN conda create --name aam -c conda-forge -c bioconda \
    h5py=3.3.0 \
    gxx_linux-64 \
    unifrac \
    python=3.9 \
    cython --yes && \
    conda clean --packages --tarballs --yes

# Install iow, tf-models-official, and tensorboard in the conda environment using pip
RUN conda run -n aam pip install \
    iow \
    tf-models-official==2.14.2 \
    tensorboard

# Stage 2: Application Layer
FROM base AS application

# Clone the specific branch of the repository and install it in editable mode
ARG BRANCH=dockerize
RUN git clone --depth=1 --branch $BRANCH https://github.com/kwcantrell/attention-all-microbes.git /app/attention-all-microbes

# Install the repository in editable mode using pip within the conda environment
WORKDIR /app/attention-all-microbes
RUN conda run -n aam pip install --no-cache-dir -e .

# Set the default command to run the 'attention' CLI using full path
CMD ["/opt/conda/bin/conda", "run", "-n", "aam", "/opt/conda/envs/aam/bin/attention"]

