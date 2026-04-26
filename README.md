# Assignment 6 : Introduction to CUDA

**Name:** Anushka  

## Files Included

1. **`device_query.cu`**  
   - Device query program (Part A)  
   - Prints all required GPU properties (name, compute capability, memory sizes, warp size, etc.)

2. **`array_sum.cu`**  
   - CUDA program to calculate the sum of elements in a single-precision floating-point array (Part B)  
   - Follows all the steps: allocate device memory, copy data, launch kernel, copy result back, free memory.

3. **`matrix_add.cu`**  
   - CUDA program for Matrix Addition of two large integer matrices (Part C)

This assignment demonstrates basic GPU programming using CUDA, including device querying, parallel reduction, and matrix operations.

Overview

The assignment is divided into three parts:

Part A focuses on querying GPU properties.
Part B implements parallel array summation.
Part C performs matrix addition using CUDA.

All programs were executed on an NVIDIA Tesla T4 GPU using Google Colab.

Part A: Device Query

A CUDA program was written to extract hardware details of the GPU.
The program retrieves information such as GPU name, compute capability, memory sizes, warp size, and execution limits.
This helps in understanding the architecture and limitations of the GPU.

Part B: Array Summation

A CUDA program was implemented to compute the sum of elements of a floating-point array.

Steps performed:

Allocated memory on the device
Copied input array from host to device
Configured grid and block dimensions
Launched a kernel where each thread processes one element
Used atomicAdd to safely accumulate the result
Copied the result back to the host
Freed device memory

The program demonstrates parallel reduction using GPU threads.

Part C: Matrix Addition

A CUDA program was written to perform the addition of two large integer matrices.
Each thread computes one element of the result matrix using a 2D grid and block configuration.
This demonstrates efficient parallel computation for data-parallel tasks.

Key Concepts Covered
CUDA programming model
Thread hierarchy (grid, block, thread)
Memory management (host and device)
Kernel execution
Parallel reduction using atomic operations
2D grid mapping for matrix operations

Results
Successfully retrieved GPU properties (Tesla T4)
Correct computation of the array sum
Successful matrix addition with expected results
Conclusion

The assignment demonstrates how CUDA enables parallel computation using GPU resources. It highlights the importance of thread organisation, memory handling, and efficient kernel execution for high-performance computing.

## How to Compile and Run

```bash
# Compile all programs
nvcc -arch=sm_75 device_query.cu -o device_query
nvcc -arch=sm_75 array_sum.cu -o array_sum
nvcc -arch=sm_75 matrix_add.cu -o matrix_add

# Run
./device_query
./array_sum
./matrix_add
