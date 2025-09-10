# GPU-Accelerated Convolutional Neural Network in C

A C-based Convolutional Neural Network (CNN) implementation to classify the CIFAR-10 dataset.

**This project was developed as part of my undergraduate thesis in Electrical and Computer Engineering at the University of Patras.**  
The implementation includes a serial version and a GPU-accelerated version using *OpenACC*. The goal was to evaluate inference performance and explore parallelization strategies on GPUs. Multiple code versions are provided to compare performance, including an optimized version using zero-padding before convolution.

### Thesis Information

> **Title:** "Implementation of Convolutional Neural Networks Using GPUs"  
> *(Greek: Υλοποίηση συνελικτικών νευρωνικών δικτύων σε κάρτες γραφικών)*  
> *University of Patras Institutional Repository link: [https://hdl.handle.net/10889/28866](https://hdl.handle.net/10889/28866)*

---

## Project Structure

There are three versions of the code, each in its own directory:

- `serial/` – Basic serial implementation of CNN
- `openacc/` – GPU-accelerated version using OpenACC
- `padding/` – Optimized version with zero-padding to simplify convolution

- `weights/` - Contains the pre-trained weights of the network


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

By default, the code uses the 50.000 images of the training dataset. You can change the number of images in `main.c`:

```
#define NUM_IMAGES 50000
```

## Compile and Run

Each version directory contains its own `Makefile`. 

Example:

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

```bash
make clean
```

## Results Example

For 50.000 images the CNN achieves 78.84 % accuracy. 
The execution time of each major part of the program is measured and printed, including data loading, network initialization, and inference (forward propagation).  

Output:

```
$ ./cnn-cifar10 
Serial Code
CNN for 50000 images
Loading input batch 1...
Loading input batch 2...
Loading input batch 3...
Loading input batch 4...
Loading input batch 5...
Load Data time:0.518275 seconds
Create Network time:0.000006 seconds
Load Network Parameters time:0.003319 seconds
Create Ouputs time:0.000235 seconds

Net Forward total time:724.833147 seconds
    Time for conv1: 235.903980 seconds
    Time for relu1: 3.219701 seconds
    Time for pool1: 3.214649 seconds
    Time for conv2: 371.571337 seconds
    Time for relu2: 0.921022 seconds
    Time for pool2: 0.946631 seconds
    Time for conv3: 108.097672 seconds
    Time for relu3: 0.246505 seconds
    Time for pool3: 0.248356 seconds
    Time for fc: 0.442517 seconds
    Time for softmax: 0.008063 seconds

  Conv: 715.572989 seconds
  ReLU: 4.387228 seconds
  Pool: 4.409636 seconds
  FC:   0.442517 seconds
  Softmax: 0.008063 seconds

Net Accuracy: 78.84 % 
Net Accuracy time:0.001737 seconds
Free memory time:0.028954 seconds
Total time:725.385673 seconds
END!

```
