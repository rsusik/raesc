# Recurrent AutoEncoder with Sequence-aware encoding

## About

This source code was written for research purpose (https://arxiv.org/abs/2009.07349) and has a minimal error checking. The code may be not very readable and comments may not be adequate. There is no warranty, your use of this code is at your own risk.

Cite (paper preprint):

```
@misc{susik2020recurrent,
    title={Recurrent autoencoder with sequence-aware encoding},
    author={Robert Susik},
    year={2020},
    eprint={2009.07349},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Requirements

- Python 3
- Tensorflow==2.3.0
- Matplotlib
- Numpy

See more details in `requirements.txt` file.

_NOTE: The code was tested on Fedora 28 64-bit, and never tested on other OS._

## Dataset
The all dataset files are "pickled" and placed in folder `datasets`.

## Running

* The training and evaluation may be executed from jupyter notebook `main.ipynb`.
* The source code of models and other utilities is in `main.py` file.