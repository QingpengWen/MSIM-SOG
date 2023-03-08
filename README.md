# MSIM-SOG

This repository contains the PyTorch implementation of the paper: 

**A deep joint model of Multi-Slot Intent interaction with Second-Order Gate for Spoken Language Understanding**.

Qingpeng Wen, Bi Zeng, Pengfei Wei, Huiting Hu. 

## Architecture

![]<img src="Figures\MSIM-SOG.png">



Our code is based on Python 3.7.6 and PyTorch 1.1. Requirements are listed as follows:
> - torch==1.1.0
> - transformers==2.4.1
> - numpy==1.18.1
> - tqdm==4.42.1
> - seqeval==0.0.12

We highly suggest you using [Anaconda](https://www.anaconda.com) to manage your python environment.

## How to Run it

### Quick start
The script **train.py** acts as a main function to the project, you can run the experiments by replacing the unspecified options in the following command with the corresponding values.

```shell
    CUDA_VISIBLE_DEVICES=$1 python train.py -dd ${dataDir} -sd ${saveDir} -u -bs 16 -dr 0.3 \ 
        -ced 128 -wed 128 -ehd 512 -aod 128 -sed 32 -sdhd 64
```

## Acknowledgement

This work was supported in part by the National Science Foundation of China under Grant 62172111, in part by the Natural Science Foundation of Guangdong Province under Grant 2019A1515011056, in part by the Key technology project of Shunde District under Grant 2130218003002.

