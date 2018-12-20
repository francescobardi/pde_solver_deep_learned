# mL18

## Setup

Execute:

```
conda env create -f environment.yml --name <your chosen name>
```

Followed by:

```
source activate <your chosen name>
```

## File Layout


```
.
.
├── environment.yml
├── environment.yml                          # environment file
├── nnpde
│   ├── main.ipynb                           # main notebook, entry point
│   ├── main.py                              # copy of main notebook as a python module
│   ├── nnpde
│   │   ├── __init__.py
│   │   ├── geometries.py
│   │   ├── helpers.py
│   │   ├── iterative_methods.py
│   │   ├── metrics.py
│   │   ├── model.py                         # model definition
│   │   ├── model_testing.py
│   │   ├── problems.py                      # problems
│   │   └── utils                            # various helpers
│   │       ├── __init__.py
│   │       ├── jupyter.ipynb
│   │       ├── jupyter.py
│   │       ├── logs.py
│   │       ├── misc.py
│   │       └── plots.py
│   ├── _graveyard_
│   │   ├── pde_with_conv.ipynb
│   │   ├── test_H.ipynb
│   │   ├── test_H.py
│   │   ├── test_jacobi.ipynb
│   │   ├── test_jacobi.py
│   │   ├── test-model.ipynb
│   │   ├── _conv_to_mat_.py
│   │   └── __init__.py
├── README.md                                # this file
└── references
    ├── convergenceJacobi.pdf
    └── paper.pdf                            # paper on which this is based
```
