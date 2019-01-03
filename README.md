# EPFL CS-433 Machine Learning: Project 2
## ICLR 2019 Reproducibility Challenge: Learning Neural PDE Solvers with Convergence Guarantees

This is the repository for the second project in the CS-433 class at EPFL.

We tried to reproduce the results of a paper handed for the ICLR conference: https://openreview.net/forum?id=rklaWn0qK7

## Setup

Execute:

```
conda env create -f environment.yml --name <your chosen name>
```

Followed by:

```
source activate <your chosen name>
```

## Structure

```
├── environment.yml                          # environment file
├── nnpde
│   ├── main.ipynb                           # main notebook, entry point
│   └── nnpde
│       ├── __init__.py
│       ├── geometries.py                    # geometries: shapes and boundaries
│       ├── helpers.py                       # more project based helpers
│       ├── iterative_methods.py             # definition of iterative solver
│       ├── metrics.py
│       ├── model.py                         # model definition
│       ├── model_testing.py
│       ├── problems.py                      # definition problems
│       └── utils                            # various helpers
│           ├── __init__.py
│           ├── jupyter.ipynb
│           ├── jupyter.py
│           ├── logs.py
│           ├── misc.py
│           └── plots.py
├── README.md                                # this file
├── report                                   # latex script, plots, etc.
└── references
    └── paper.pdf                            # paper on which this is based
```

The notebook files were converted using this
[script](https://gist.github.com/samuelsmal/144e1204d646cd65ff8864d4b483f948),
but should be viewed as a notebook.

## General comments about the code

The deep learning part is implemented in PyTorch, therefore when in doubt it's
a PyTorch tensor.

## Authors (in alphabetical order)

Francesco Bardi, Samuel Edler von Baussnern, Emiljano Gjiriti

fransesco.bardi@epfl.ch, samuel.edervonbaussnern@epfl.ch, emiljano.gjiriti@epfl.ch

