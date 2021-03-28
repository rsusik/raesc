# Recurrent AutoEncoder with Sequence-aware encoding

## About

This source code was written for research purpose (https://arxiv.org/abs/2009.07349) and has a minimal error checking. The code may be not very readable and comments may not be adequate. There is no warranty, your use of this code is at your own risk.

> The paper has been accepted to ICCS 2021 conference (https://www.iccs-meeting.org/iccs2021/). The citation will be updated once the proceedings appear.

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

* The training may be executed by running `run.py`, example:
    ```shell
    python run.py rae,raes,raesc 1,2,4,8 1,2,4
    ```
* All charts used in the research can be generated from the jupyter notebook `main.ipynb`.
* The source codes and other utilities are in `main.py` file.