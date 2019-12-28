# Reproducibility Challenge @ NeurIPS 2019

## Introduction

This code repository hosts the PyTorch implementation of "Learning Robust Global Representations by Penalizing Local Predictive Power" ([arXiv](https://arxiv.org/abs/1905.13549)). It tries to reproduce some of the results presented in the paper.

The original TensorFlow implementation by the authors could be found [here](https://github.com/HaohanWang/PAR_experiments) (executable codes) and [here](https://github.com/HaohanWang/PAR) (clear code for illustration).

This code is this repository is **not** organized as `main.py`, models, utility functions and other components, but rather each individual experiment. This is **intended** for quick and easy replication.

## Setup

The following demonstrate how to run the codes. Note the "Section 4.x" corresponds the section number in the [original paper](https://arxiv.org/abs/1905.13549). The directory structure ready to reproduce experiment should look like

```
├── data
│   ├── CIFAR10
│   │   ├── testData_greyscale.npy
│   │   ├── testData_negative.npy
│   │   ├── testData.npy
│   │   ├── testData_radiokernel.npy
│   │   ├── testData_randomkernel.npy
│   │   ├── testLabel.npy
│   │   ├── trainData.npy
│   │   └── trainLabel.npy
│   └── MNIST
│       └── mnist.pkl.gz
├── exp1.py
├── exp2.py
├── generate_cifar10.py
└── README.md
```

## Hyperparameters

The hyperparameter is set as follows. If you would like to customize the choice of hyperparameters. You could directly modify the `exp1.py`  and `exp2.py` scripts. This fixed choice is **intentional** and aims to be consistent with the choice in the original paper (except number of epochs).

| Hyperparameter   | Value                     |
| ---------------- | ------------------------- |
| Learning rate    | 1e-4                      |
| Batch size       | 64                        |
| Number of epochs | 100 (MNIST), 80 (CIFAR10) |

### Perturbed MNIST  (Section 4.1)

1. Download the MNIST data from [here](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz) and put it into directory `./data/MNIST/`

2. Run the code in the `.` directory. 

   ```bash
   python exp1.py --test_case [0-3] --superficial_mode [dependent, independent] --par_type [PAR, PARH, PARB, PARM] --adv_strength [any nonnegative number]
   ```

   The `--par_type` and `--adv_strength` are self-explanatory. `--test-case` and `--superficial_mode` refers to different superficial patterns attached to the dataset, where

   - 0, 1 and 2 correspond to making original data, randomly filtered data (in frequency domain), and radially filtered data (in frequency domain) as test set while the other two as training and validation set. Since in the training time, the classifier has access to perturbed data, this is known as a domain adaptation (DA) task.
   - "dependent" and "independent" correspond to two ways to attach these patterns. "dependent" means the digits 0-4 have one pattern while 5-9 have other patterns. "Independent" means digit is independent of pattern.
   
3. After the execution, a `.pickle` file that records training/validation accuracy/loss, test accuracy in the form of Python dictionary is generated.

### Perturbed CIFAR10 (Section 4.2)

1. Download CIFAR10 Python version from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and unwrap the dataset in some directory. This will generate a folder named `cifar-10-python`.

   ```bash
   tar -xvf cifar-10-python.tar.gz
   ```

2. Move the `generate_cifar10.py` into the same directory of `cifar-10-python`, generate the experiment dataset by executing `generate_cifar10.py` . This will create 8 `.npy` files.

   ```bash
   python generate_cifar10.py
   ```

3. Create `./data/CIFAR/` directory and move 8 `.npy` files into it.

4. Run the code in the `.` directory.

```bash
python exp2.py --test_case [0-4] --par_type [PAR, PARH, PARB, PARM] --adv_strength [any nonnegative number]
```

The options `--par_type` and `adv_strength` is similar to the first experiment. However, the `--test_case`here have different meanings here. Specifically, 0, 1, through 4 here refer to using original test data, grayscale data, negative color data, randomly filtered data (in frequency domain), and radially filtered data (in frequency domain) as test set. In this case, the classifier does **not** access perturbed data, it is therefore a domain generalization (DG) task.

4. After the execution, a `.pickle` file that records training/validation accuracy/loss, test accuracy in the form of Python dictionary is generated.

### Batch Experiment

One could use shell scripts to run a series of experiments at one command, one such script may look like

```shell
#!/bin/sh
for type in "PAR" "PARH" "PARM" "PARB" "vanilla"
do
    for test_data in 0 1 2 3 4
    do
        python exp2_arg.py --test_case ${test_data} --par_type ${type}
    done
done
```

## Author

Guanqun Yang (guanqun.yang@engineering.ucla.edu or guanqun.yang@outlook.com)





