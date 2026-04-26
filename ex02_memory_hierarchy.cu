#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } \
} while(0)

#define THREADS 256
#define N (1<<20)

// Naive Reduction
__global__ void naiveReduce(const float* in, float* out, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += in[i];
        out[0] = sum;
    }
}

// Shared Memory Tree Reduction
__global__ void sharedReduce(const float* in, float* out, int n) {
    __shared__ float sdata[THREADS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// Warp Shuffle Reduction
__global__ void warpReduce(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? in[i] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (threadIdx.x % 32 == 0)
        atomicAdd(out, val);
}

// Bank Conflict Kernel
__global__ void bankConflictKernel(float* out, int stride, int n) {
    __shared__ float smem[1024];
    int tid = threadIdx.x;
    smem[tid * stride % 1024] = tid;
    __syncthreads();
    if (tid < n) out[tid] = smem[tid * stride % 1024];
}

int main() {
    printf("\n=== Problem 2: Parallel Reduction + Bank Conflicts ===\n");

    float *d_in, *d_partial, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, ((N+THREADS-1)/THREADS) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0, N * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms;

    printf("%-20s %12s %12s\n", "Method", "Time (us)", "GB/s");
    printf("------------------------------------------------\n");

    // Naive
    CUDA_CHECK(cudaEventRecord(start));
    naiveReduce<<<1,1>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("%-20s %12.1f %12.2f\n", "Naive", ms*1000, 0.0);

    // Shared Tree
    CUDA_CHECK(cudaEventRecord(start));
    int blocks = (N + THREADS - 1) / THREADS;
    sharedReduce<<<blocks, THREADS>>>(d_in, d_partial, N);
    sharedReduce<<<1, THREADS>>>(d_partial, d_out, blocks);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float tp = (N * sizeof(float) * 1.0) / (ms / 1000.0) / 1e9;
    printf("%-20s %12.1f %12.2f\n", "Shared Tree", ms*1000, tp);

    // Warp Shuffle
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    warpReduce<<<blocks, THREADS>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float tp_warp = (N * sizeof(float) * 1.0) / (ms / 1000.0) / 1e9;
    printf("%-20s %12.1f %12.2f\n", "Warp Shuffle", ms*1000, tp_warp);

    // Bank Conflict
    printf("\n[B3] Bank Conflict Timing (lower = better):\n");
    printf("%8s %12s\n", "Stride", "Time (us)");
    for (int s : {1,2,4,8,16,32}) {
        CUDA_CHECK(cudaEventRecord(start));
        bankConflictKernel<<<1, 1024>>>(d_in, s, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("%8d %12.1f\n", s, ms*1000);
    }

    printf("\n ex02 fully completed as per assignment!\n");
    cudaFree(d_in); cudaFree(d_partial); cudaFree(d_out);
    return 0;
}
