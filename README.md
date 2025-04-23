# Convolutional Neural Network for CIFAR-10 in C

This project implements a Convolutional Neural Network (CNN) in C to classify images from the CIFAR-10 dataset. It was developed as part of my Diploma Thesis at the University of Patras.

## Thesis Information

> **Title:** "Implementation of Convolutional Neural Networks Using GPUs"  
> *(Greek: Υλοποίηση συνελικτικών νευρωνικών δικτύων σε κάρτες γραφικών)*  
> link: [https://hdl.handle.net/10889/28866](https://hdl.handle.net/10889/28866)

---

### **Abstract (EN)**

This project implements a convolutional neural network (CNN) in the C programming language to classify images from the CIFAR-10 dataset. The implementation includes a serial version and a GPU-accelerated version using OpenACC. The goal was to evaluate inference performance and explore parallelization strategies on NVIDIA GPUs. Multiple code versions are provided to compare performance, including an optimized version using zero-padding before convolution.

---

## Project Structure

- `serial/` – Basic serial implementation of CNN
- `openacc/` – GPU-accelerated version using OpenACC
- `padding/` – Optimized version with zero-padding to simplify convolution
- `common/` – Utility functions for memory allocation and timing

## How to Compile

Each version folder contains its own `Makefile`. 
Μake sure you have the NVIDIA HPC compiler (nvc) installed.

Example (from the serial version):

```bash
cd serial
make
./cnn
```
