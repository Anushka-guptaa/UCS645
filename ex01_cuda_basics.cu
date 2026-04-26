#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define THREADS_BASE 256

/* ================================================================
 * SECTION A — PROVIDED
 * ================================================================ */
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

void cpu_vectorAdd(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}

/* ================================================================
 * PART A: FULL BENCHMARK (N = 2^10 to 2^26)
 * ================================================================ */
void benchmark_speedup(void) {
    int exponents[] = {10, 14, 18, 22, 26};
    int num_tests = 5;

    printf("\n=== Problem 1 Part A: CPU vs GPU Speedup Benchmark ===\n");
    printf("%12s %12s %12s %12s %12s\n", "N", "CPU (ms)", "GPU (ms)", "H2D (ms)", "Speedup");
    printf("--------------------------------------------------------------------------------\n");

    for (int t = 0; t < num_tests; t++) {
        int N = 1 << exponents[t];
        size_t bytes = N * sizeof(float);

        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C = (float*)malloc(bytes);
        float *h_ref = (float*)malloc(bytes);

        for (int i = 0; i < N; i++) {
            h_A[i] = (float)rand() / RAND_MAX;
            h_B[i] = (float)rand() / RAND_MAX;
        }

        /* CPU Time */
        clock_t start = clock();
        cpu_vectorAdd(h_A, h_B, h_ref, N);
        double cpu_ms = (clock() - start) * 1000.0 / CLOCKS_PER_SEC;

        /* GPU Setup */
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        /* H2D Time */
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float h2d_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, t0, t1));

        /* GPU Kernel Time */
        int threads = THREADS_BASE;
        int blocks = (N + threads - 1) / threads;

        CUDA_CHECK(cudaEventRecord(t0));
        vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float gpu_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t0, t1));

        float speedup = cpu_ms / gpu_ms;

        printf("%12d %12.2f %12.2f %12.2f %12.2f\n", N, cpu_ms, gpu_ms, h2d_ms, speedup);

        /* Cleanup */
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        free(h_A); free(h_B); free(h_C); free(h_ref);
    }
}

/* ================================================================
 * PART B: Launch Config Analysis
 * ================================================================ */
void benchmark_launch_config(void) {
    int block_sizes[] = {64, 128, 256, 512, 1024};
    int N = 1 << 20;   // 1M elements
    size_t bytes = N * sizeof(float);

    printf("\n=== Problem 1 Part B: Launch Configuration Analysis ===\n");
    printf("%10s %12s %12s\n", "BlockSize", "Blocks", "KernelTime(ms)");
    printf("--------------------------------------------\n");

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    for (int bs = 0; bs < 5; bs++) {
        int threads = block_sizes[bs];
        int blocks = (N + threads - 1) / threads;

        CUDA_CHECK(cudaEventRecord(t0));
        vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

        printf("%10d %12d %12.3f\n", threads, blocks, ms);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    free(h_A); free(h_B); free(h_C);
}

/* ================================================================
 * MAIN
 * ================================================================ */
int main(void) {
    printf("\n========================================================\n");
    printf("  CUDA DIY Exercise 1: GPU Architecture & Profiling (COMPLETE)\n");
    printf("========================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    benchmark_speedup();           // Part A: Full benchmark
    benchmark_launch_config();     // Part B: Launch config test

    printf("\n ex01 fully completed as per assignment requirements!\n");
    printf("Now you have table for report + crossover analysis ready.\n");
    return 0;
}
