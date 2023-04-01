# Learning Multiscale Convolutional Dictionaries for Image Reconstruction

**Authors:** Tianlin Liu, Anadi Chaman, David Belius, and Ivan Dokmanić

This repository hosts the code for our paper titled [Learning Multiscale Convolutional Dictionaries for Image Reconstruction](https://arxiv.org/abs/2011.12815).


```BibTeX
@ARTICLE{Liu2022learning,
  author={Liu, Tianlin and Chaman, Anadi and Belius, David and Dokmanić, Ivan},
  journal={IEEE Transactions on Computational Imaging},
  title={Learning Multiscale Convolutional Dictionaries for Image Reconstruction},
  year={2022},
  volume={8},
  pages={425-437}}
```

In summary (TL;DR), we propose a multiscale dictionary model. We found that when this model is trained with a standard sparse-coding approach, it performs comparably to the widely recognized U-Net.


## Installation
Create the environment from the environment.yml file

```sh
conda env create -f environment.yml
```

## Usage
To train the MUSC model, use python scripts in the folder `train_src`.
