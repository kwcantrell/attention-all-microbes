# Attention All Microbes (AAM)

Attention-based network for microbial sequencing data. 

# Installation Instructions
IMPORTANT: If installing on a server cluster, spawn an instance with a GPU before proceeding with environment setup.
First create a new conda environment with unifrac

`conda create --name aam -c conda-forge -c bioconda unifrac`
`conda activate aam`

## GPU Support 

Install CUDA 11.8

`conda install nvidia/label/cuda-11.8.0::cuda-toolkit`

Verify the NVIDIA GPU drives are on your path

`nvidia-smi`

`pip install tensorflow==2.14 tf-models-official==2.14.2`
Please see [Tensorflow](https://www.tensorflow.org/install) for more information

## Install AAM


For the latest version

`pip install git+https://github.com/kwcantrell/attention-all-microbes.git`

or install a specific version

`pip install git+https://github.com/kwcantrell/attention-all-microbes.git@v0.1.0`


# Training

Classifiers and Regressors are trained use cross-validation 

`attention-all-microbes --help`







