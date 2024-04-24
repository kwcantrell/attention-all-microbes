# Attention All Microbes (AAM)
<b>Alpha release</b> : Bugs may be present

Attention-based network for microbial sequencing data. 

# Installation

Tensorflow version = 2.14
Python version = 3.9
## General Usage

TBA

## For Knight Lab Users
1. Start GPU instance on Barnacle
2. `module load tensorflow=2.14_3.9`
3. `python -m venv path/to/env`
4. `source path/to/env/bin/activate`
5. `pip install -r requirements.txt`

# Training

Currently only supports  `python cli.py fit-regressor`


# Results

Currently only supports  `python cli.py scatter-plot`

<h2>Significant improvement over random forest</h2>

Compare results from [age regression](https://journals.asm.org/doi/10.1128/msystems.00630-19)

| AAM | Random Forest |
|-----|---------------|
|![Scatter](docs/scatter-plot.png)|![Random](docs/random_forest.png)|
|![Residual](docs/residual-plot.png)| |







