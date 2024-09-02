# Optimizing the Deployment of Tiny Transformers on Low-Power MCUs
*by Victor J.B. Jung,*
*Alessio Burrello,*
*Moritz Scherer,*
*Francesco Conti,*
*Luca Benini*

[[`paper (arxiv)`](https://arxiv.org/abs/2404.02945)]

This repository contains code used in *Optimizing the Deployment of Tiny Transformers on Low-Power MCUs*.

## Abstract

Transformer networks are rapidly becoming SotA in many fields, such as NLP and CV. Similarly to CNNs, there is a strong push for deploying Transformer models at the extreme edge, ultimately fitting the tiny power budget and memory footprint of MCUs. However, the early approaches in this direction are mostly ad-hoc, platform, and model-specific. This work aims to enable and optimize the flexible, multi-platform deployment of encoder Tiny Transformers on commercial MCUs. We propose a complete framework to perf orm end-to-end deployment of Transformer models onto single and multi-core MCUs. Our framework provides an optimized library of kernels to maximize data reuse and avoid unnecessary data marshaling operations into the crucial attention block. A novel MHSA inference schedule, named FWSA, is introduced, fusing the linear projection weights offline to further reduce the number of operations and parameters. Furthermore, to mitigate the memory peak reached by the computation of the attention map, we present a DFT scheme for MHSA tailored for cache-less MCU devices that allows splitting the computation of the attention map into successive steps, never materializing the whole matrix in memory. We evaluate our framework on three different MCU classes exploiting ARM and RISC-V ISA, namely the STM32H7 (ARM Cortex M7), the STM32L4 (ARM Cortex M4), and GAP9 (RV32IMC-XpulpV2). We reach an average of 4.79x and 2.0x lower latency compared to SotA libraries CMSIS-NN (ARM) and PULP-NN (RISC-V), respectively. Moreover, we show that our MHSA depth-first tiling scheme reduces the memory peak by up to 6.19x, while the fused-weight attention can reduce the runtime by 1.53x, and number of parameters by 25%. Leveraging the optimizations proposed in this work, we run end-to-end inference of three SotA Tiny Transformers for three applications characterized by different input dimensions and network hyperparameters. We report significant improvements across the networks: for instance, when executing a transformer block for the task of radar-based hand-gesture recognition on GAP9, we achieve a latency of 0.14ms and energy consumption of 4.92 micro-joules, 2.32x lower than the SotA PULP-NN library on the same platform. 

## Kernel test harness

Start by creating a fresh python environement with `Python 3.10` and install required packages with:
```
pip install -r ./Test/testEnv.txt
```
You will also need to install the latest version of the [GAP SDK](https://github.com/GreenWaves-Technologies/gap_sdk).

To run the test harness, it is as simple as running the `kernelTest.sh` script.

You can configure which test you want to run and select the hyperparameters of the tests by modifying the `testConfig.yaml` file.

For instance this is the config file to run the Q linear projection kernel with the parameters of the EEGFormer:
```
cores: 8
seed: 42

### MHSA Parameters ###
S:
  - 81
E: 
  - 32
P:
  - 32
H: 
  - 8

testToRun:
  - projQK

# Projection QK
projQK:
  kernelName: linearQK_4x2_H
  appFolder: ./Application/GAP9LinProjQK
  inputGen: generateInputsQKV
  templateGen: generateTemplateQKV
  goldenKernel: linearProjectionQK
  platform: gvsoc
```

If you want to run more than one test a the time you can simply add more test to the `testToRun` list in the config file.

## Citation

If you use our work or find it valuable, please cite us with:

```
@misc{jung2024optimizingdeploymenttinytransformers,
      title={Optimizing the Deployment of Tiny Transformers on Low-Power MCUs}, 
      author={Victor J. B. Jung and Alessio Burrello and Moritz Scherer and Francesco Conti and Luca Benini},
      year={2024},
      eprint={2404.02945},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.02945}, 
}
```