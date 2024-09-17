# Attention All Microbes (AAM)

Attention-based network for microbial sequencing data. 

# Installation

## Install Requirements
Requires tensorflow==2.14 and tf-models-official==2.14.2

`pip install tensorflow==2.14 tf-models-official==2.14.2` 

or

 `pip install tensorflow[and-cuda]==2.14 tf-models-official==2.14.2` 

for GPU support.

Tensorboard is an optional dependency to visualize training losses/metrics.

`pip install tensorboard`

## Install AAM

For the latest version

`pip install git+https://github.com/kwcantrell/attention-all-microbes.git`

or

`pip install git+https://github.com/kwcantrell/attention-all-microbes.git@v0.1.0`

for a specific tagged version.


# Training

Classifiers and Regressors are trained use cross-validation 

`python attention_cli.py --help`







