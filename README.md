# Attention All Microbes (AAM)

Attention-based network for microbial sequencing data. 

# Installing the Repo and Environment 

1. Clone repo using `git clone git@github.com:kwcantrell/attention-all-microbes.git`
2. Run `conda env create -f environment.yml` (name defaults to attention-all-microbes)
3. `conda activate attention-all-microbes`
4. Install Tensorflow - See below:
   
## Tensorflow Install Requirements
IMPORTANT: If installing onto a cluster based system, make sure you install tensorflow while on an instance that has an NVIDIA gpu.

AAM Requires tensorflow==2.14 and tf-models-official==2.14.2

`pip install tensorflow==2.14 tf-models-official==2.14.2` 

or

 `pip install tensorflow[and-cuda]==2.14 tf-models-official==2.14.2` 

for GPU support.

Tensorboard is an optional dependency to visualize training losses/metrics.

`pip install tensorboard`



# Training

Classifiers and Regressors are trained use cross-validation 

`python attention_cli.py --help`







