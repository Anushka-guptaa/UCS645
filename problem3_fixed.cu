#include <stdio.h>
#include <cuda_runtime.h>

#define N 262144

__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

__global__ void vectorAddKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

int main() {
    float h_A[N], h_B[N], h_C[N];

    // Initialize
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    cudaMemcpyToSymbol(d_A, h_A, sizeof(h_A));
    cudaMemcpyToSymbol(d_B, h_B, sizeof(h_B));

    // Timing with CUDA Events (as required)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpyFromSymbol(h_C, d_C, sizeof(h_C));

    printf("Kernel execution time: %.3f ms\n\n", milliseconds);

    // 1.3 Theoretical Bandwidth
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float clockRateGHz = prop.memoryClockRate / 1000000.0f;
    float theoreticalBW = (clockRateGHz * prop.memoryBusWidth * 2) / 8.0f;  // GB/s

    printf("=== Theoretical Memory Bandwidth ===\n");
    printf("Memory Clock Rate : %.2f GHz\n", clockRateGHz);
    printf("Memory Bus Width  : %d bits\n", prop.memoryBusWidth);
    printf("Theoretical Bandwidth : %.2f GB/s\n\n", theoreticalBW);

    // 1.4 Measured Bandwidth
    size_t bytesRead    = 2 * N * sizeof(float);
    size_t bytesWritten = N * sizeof(float);
    size_t totalBytes   = bytesRead + bytesWritten;
    float seconds = milliseconds / 1000.0f;
    float measuredBW = totalBytes / (seconds * 1e9f);

    printf("=== Measured Memory Bandwidth ===\n");
    printf("Measured Bandwidth    : %.2f GB/s\n", measuredBW);
    printf("Efficiency            : %.1f%%\n\n", (measuredBW / theoreticalBW) * 100);

    printf("Vector addition completed successfully!\n");
    printf("Sample check: C[0] = %.1f (should be 3.0)\n", h_C[0]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
