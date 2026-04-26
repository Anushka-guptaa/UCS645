#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 7. CUDA Kernel that computes the sum
__global__ void sumKernel(float *d_input, float *d_output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(d_output, d_input[idx]);   // Simple reduction using atomicAdd
    }
}

int main() {
    int N = 1 << 20;                    // 1,048,576 elements (you can change this)
    size_t size = N * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(sizeof(float));

    // Generate input array (all 1.0f → expected sum = N)
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_output;

    // 1. Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, sizeof(float));

    // 2. Copy host memory to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));   // initialize output to 0

    // 3. Initialize thread block and kernel grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 4. Invoke CUDA kernel
    sumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    cudaDeviceSynchronize();


    // 5. Copy results from device to host
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    printf("Sum of array = %.2f (Expected: %d)\n", *h_output, N);

    free(h_input);
    free(h_output);
    return 0;
}
