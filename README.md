# Convolutional Neural Network for CIFAR-10 in C

A C-based Convolutional Neural Network (CNN) implementation to classify the CIFAR-10 dataset.

**This project was developed as part of my undergraduate thesis in Electrical and Computer Engineering at the University of Patras.**  
The implementation includes a serial version and a GPU-accelerated version using OpenACC. The goal was to evaluate inference performance and explore parallelization strategies on GPUs. Multiple code versions are provided to compare performance, including an optimized version using zero-padding before convolution.

### Thesis Information

> **Title:** "Implementation of Convolutional Neural Networks Using GPUs"  
> *(Greek: Υλοποίηση συνελικτικών νευρωνικών δικτύων σε κάρτες γραφικών)*  
> [University of Patras Institutional Repository link](https://hdl.handle.net/10889/28866)

---

## Project Structure

There are three versions of the code, each in own directory:

- `serial/` – Basic serial implementation of CNN
- `openacc/` – GPU-accelerated version using OpenACC
- `padding/` – Optimized version with zero-padding to simplify convolution


## Requirements 

### Compiler

This project uses the `nvc` compiler of the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk), which needs an NVIDIA GPU.

To run the serial code with the `gcc` compiler or other, change the **Makefile**:
```
CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -march=native
```

### Dataset

To run the code, you need to download the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) in binary format.

After extracting the archive, set the dataset path in `main.c` using the `DATA_FOLDER` constant:
```
const char* DATA_FOLDER = "../../cifar-10-batches-bin";
```
Make sure the path points to the folder containing the binary files (data_batch_1.bin, data_batch_2.bin, etc.). You can modify the path to match your local directory structure.

## Compile and Run

Each version directory contains its own `Makefile`. 

Example (from the parallel version):

```bash
cd openacc
make
./cnn-cifar10
```

### Debugging

Compile with `-g` flag for debugging:

```bash
make debug
./cnn-cifar10-debug
```

### Profiling

Compile with `-pg` flag for profiling:

```bash
make profile
./cnn-cifar10-profile
```

### Clean Up

To remove all compiled files:

```
make clean
```