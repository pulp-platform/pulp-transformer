# A Microcontroller is All You Need: Enabling Transformer Execution on Low-Power IoT Endnodes

*by Alessio Burrello,*
*Moritz Scherer,*
*Marcello Zanghieri,*
*Francesco Conti,*
*Luca Benini*

[[`paper`](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9524173)] [[`slides`](https://pulp-platform.org/docs/IEEECOINS2021.pdf)]

This repository contains the code of the kernels used in the *MCU is All You Need*

## Abstract

Transformer networks have become state-of-the-art for many tasks such as NLP and are closing the gap on other tasks like image recognition. Similarly, Transformers and Attention methods are starting to attract attention on smaller-scale tasks, which fit the typical memory envelope of MCUs. In this work, we propose a new set of execution kernels tuned for efficient execution on MCU-class RISC-V and ARM Cortex-M cores. We focus on minimizing memory movements while maximizing data reuse in the Attention layers. With our library, we obtain 3.4×, 1.8×, and 2.1× lower latency and energy on 8-bit Attention layers, compared to previous state-of-the-art (SoA) linear and matrix multiplication kernels in the CMSIS-NN and PULP-NN libraries on the STM32H7 (Cortex M7), STM32L4 (Cortex M4), and GAP8 (RISC-V IMC-Xpulp) platforms, respectively. As a use case for our TinyTransformer library, we also demonstrate that we can fit a 263 kB Transformer on the GAP8 platform, outperforming the previous SoA convolutional architecture on the TinyRadarNN dataset, with a latency of 9.24 ms and 0.47 mJ energy consumption and an accuracy improvement of 3.5%.

## Citation

If you use our work or find it valuable, please cite us with:
```
@INPROCEEDINGS{9524173,
  author={Burrello, Alessio and Scherer, Moritz and Zanghieri, Marcello and Conti, Francesco and Benini, Luca},
  booktitle={2021 IEEE International Conference on Omni-Layer Intelligent Systems (COINS)}, 
  title={A Microcontroller is All You Need: Enabling Transformer Execution on Low-Power IoT Endnodes}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  keywords={Energy consumption;Image recognition;Microcontrollers;Conferences;Transformer cores;Libraries;Classification algorithms;TinyML;Transformers;Deep Learning;Internet of Things},
  doi={10.1109/COINS51742.2021.9524173}}
```
