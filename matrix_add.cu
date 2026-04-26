#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024   // Large matrix size (1024 x 1024)

__global__ void matrixAddKernel(int *A, int *B, int *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];     // one integer addition per element
    }
}

int main() {
    size_t size = N * N * sizeof(int);

    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i % 100;
        h_B[i] = i % 50;
    }

    int *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Thread and grid configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15)/16, (N + 15)/16);

    // Launch kernel
    matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("Matrix addition completed successfully!\n");
    printf("Sample result C[0][0] = %d\n", h_C[0]);

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
