# Attention All Microbes (AAM)

Attention-based network for microbial sequencing data. 

# Installation Instructions
IMPORTANT: If installing on a server cluster, spawn an instance with a GPU before proceeding with environment setup.
First create a new conda environment with unifrac

`conda create --name aam -c conda-forge -c bioconda unifrac python=3.9 cython`

`conda activate aam`

`conda install -c conda-forge gxx_linux-64 hdf5 mkl-include lz4 hdf5-static libcblas liblapacke make`


## Install AAM


For the latest version

`pip install git+https://github.com/kwcantrell/attention-all-microbes.git`

or install a specific version

`pip install git+https://github.com/kwcantrell/attention-all-microbes.git@v0.1.1`

## Developers

`git clone git@github.com:kwcantrell/attention-all-microbes.git`

`cd attention-all-microbes`

`pip install -e .`

# Training

Classifiers and Regressors are trained use cross-validation 

`attention --help`







