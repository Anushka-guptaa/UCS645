# Assignment 8: GPU Accelerated Machine Learning


## Files Included

| File                        | Exercise | Description |
|-----------------------------|----------|-------------|
| `ex01_cuda_basics.cu`       | Problem 1 | Vector operations, bandwidth benchmark, launch config analysis |
| `ex02_memory_hierarchy.cu`  | Problem 2 | Shared memory reduction, bank conflicts, warp shuffle |
| `ex03_ml_primitives.cu`     | Problem 3 | Activations (Sigmoid, Tanh, Leaky ReLU), ReLU backward, BCE & Cross-Entropy loss |
| `ex04_cnn_layers.cu`        | Problem 4 | Tiled GEMM, MaxPool2x2, BatchNorm inference |
| `ex05_mnist_cnn.cu`         | Problem 5 | MNIST CNN training pipeline simulation |
And a detailed report with outputs.

Summary of Work Done

-Problem 1: Full bandwidth benchmark + launch configuration analysis (block sizes 64–1024)
-Problem 2: Naive, Shared Memory Tree, and Warp Shuffle reductions + bank conflict timing experiment
-Problem 3: Complete activation suite + BCE & Cross-Entropy loss kernels with verification
-Problem 4: Tiled MatMul, MaxPool2x2, and BatchNorm inference kernels
-Problem 5: Simulated full MNIST CNN training loop showing decreasing loss and increasing accuracy

## How to Compile and Run

```bash
nvcc -arch=sm_75 ex01_cuda_basics.cu      -o ex01
nvcc -arch=sm_75 ex02_memory_hierarchy.cu -o ex02
nvcc -arch=sm_75 ex03_ml_primitives.cu    -o ex03
nvcc -arch=sm_75 ex04_cnn_layers.cu       -o ex04 -lcublas
nvcc -arch=sm_75 ex05_mnist_cnn.cu        -o ex05 -lcudnn -lcublas

./ex01
./ex02
./ex03
./ex04
./ex05
