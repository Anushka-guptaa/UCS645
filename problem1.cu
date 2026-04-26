#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// Kernel a: Iterative approach - sum 1 to N using loop (no direct formula)
__global__ void iterativeSumKernel(int *d_input, int *d_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    atomicAdd(&d_output[0], d_input[idx]);
}
}

// Kernel b: Direct formula - sum = n*(n+1)/2
__global__ void formulaSumKernel(int *d_input, int *d_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        int sum = N * (N + 1) / 2;
        d_output[1] = sum;
    }
}

int main() {
    int h_input[N];
    int h_output[2] = {0};

    // Step 2 & 4: Create and fill input array with first N integers
    for (int i = 0; i < N; i++) {
        h_input[i] = i + 1;
    }

    int *d_input, *d_output;

    // Step 3: Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, 2 * sizeof(int));

    // Step 5: Copy host to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, 2 * sizeof(int));

    // Step 6: Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Step 7: Launch kernels
    iterativeSumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    formulaSumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);

    // Copy result back
    cudaMemcpy(h_output, d_output, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("a. Iterative Sum (1 to %d)   = %d\n", N, h_output[0]);
    printf("b. Formula Sum (1 to %d)     = %d\n", N, h_output[1]);
    printf("Both results should match!\n");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
