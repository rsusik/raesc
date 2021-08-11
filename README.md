# Recurrent AutoEncoder with Sequence-aware encoding

## About

This source code was written for research purpose (https://arxiv.org/abs/2009.07349) and has a minimal error checking. The code may be not very readable and comments may not be adequate. There is no warranty, your use of this code is at your own risk.

> The paper has been accepted to ICCS 2021 conference (https://www.iccs-meeting.org/iccs2021/).

## Cite:

```
@InProceedings{10.1007/978-3-030-77964-1_4,
    author="Susik, Robert",
    editor="Paszynski, Maciej and Kranzlm{\"u}ller, Dieter and Krzhizhanovskaya, Valeria V. and Dongarra, Jack J. and Sloot, Peter M. A.",
    title="Recurrent Autoencoder with Sequence-Aware Encoding",
    booktitle="Computational Science -- ICCS 2021",
    year="2021",
    publisher="Springer International Publishing",
    address="Cham",
    pages="47--57",
    abstract="Recurrent Neural Networks (RNN) received a vast amount of attention last decade. Recently, the architectures of Recurrent AutoEncoders (RAE) found many applications in practice. RAE can extract the semantically valuable information, called context that represents a latent space useful for further processing. Nevertheless, recurrent autoencoders are hard to train, and the training process takes much time. This paper proposes a new recurrent autoencoder architecture with sequence-aware encoding (RAES), and its second variant which employs a 1D Convolutional layer (RAESC) to improve its performance and flexibility. We discuss the advantages and disadvantages of the solution and prove that the recurrent autoencoder with sequence-aware encoding outperforms a standard RAE in terms of model training time in most cases. The extensive experiments performed on a dataset of generated sequences of signals shows the advantages of RAES(C). The results show that the proposed solution dominates over the standard RAE, and the training process is the order of magnitude faster.",
    isbn="978-3-030-77964-1"
}
```

## Requirements

- Python 3
- Tensorflow==2.3.0
- Matplotlib
- Numpy

See more details in `requirements.txt` file.

_NOTE: The code was tested on Linux 64-bit OS._

## Dataset
The all dataset files are "pickled" and placed in folder `datasets`.

## Running

* The training may be executed by running `run.py`, example:
    ```shell
    python run.py rae,raes,raesc 1,2,4,8 1,2,4
    ```
* All charts used in the research can be generated from the jupyter notebook `main.ipynb`.
* The source codes and other utilities are in `main.py` file.
