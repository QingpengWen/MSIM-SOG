# MSIM-SOG

This repository contains the `PyTorch` implementation of the paper in the 2023 International Conference on Neural Information Processing (***[ICONIP 2023](http://www.iconip2023.org/) , Vol. 12***): 

**[A deep joint model of Multi-Slot Intent interaction with Second-Order Gate for Spoken Language Understanding](https://link.springer.com/chapter/10.1007/978-981-99-8148-9_4)**.

[Qingpeng Wen](mailto:wqp@mail2.gdut.edu.cn), [Bi Zeng](mailto:zb9215@gdut.edu.cn), [Pengfei Wei](mailto:wpf@gdut.edu.cn), [Huiting Hu](mailto:huhuiting@zhku.edu.cn)

In the following, we will guide you how to use this repository step by step.

## Architecture

<img src="Figures\MSIM-SOG.png">



Our code is based on Python 3.7.6 and PyTorch 1.13.1. Requirements are listed as follows:
> - torch==1.13.1
> - transformers==2.4.1
> - numpy==1.18.1
> - tqdm==4.42.1
> - seqeval==0.0.12

We highly suggest you using [Anaconda](https://www.anaconda.com) to manage your python environment.

## How to Run it

### Quick start
The script **train.py** acts as a main function to the project, you can run the experiments by replacing the unspecified options in the following command with the corresponding values:

```shell
    CUDA_VISIBLE_DEVICES=$1 python train.py -dd ${dataDir} -sd ${saveDir}
```

or run the script **train.py** directly via pycharm.

## Acknowledgement

This work was supported in part by the National Science Foundation of China under Grant 62172111, in part by the Natural Science Foundation of Guangdong Province under Grant 2019A1515011056, in part by the Key technology project of Shunde District under Grant 2130218003002.

## Cite this paper
@InProceedings{[10.1007/978-981-99-8148-9_4](https://doi.org/10.1007/978-981-99-8148-9_4),

author="Wen, Qingpeng and Zeng, Bi and Wei, Pengfei and Hu, Huiting",

title="A Deep Joint Model of Multi-scale Intent-Slots Interaction with Second-Order Gate for SLU",

booktitle="Neural Information Processing",

year="2024",

publisher="Springer Nature Singapore",

address="Singapore",

pages="42--54",

isbn="978-981-99-8148-9"

}
