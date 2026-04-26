#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    printf("Number of CUDA devices: %d\n\n", count);

    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("=== Device %d: %s ===\n", i, prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Max block dimensions: (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024*1024));
        printf("Constant memory: %lu bytes\n", prop.totalConstMem);
        printf("Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
        printf("Warp size: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Double precision supported: %s\n\n", 
               (prop.major >= 2) ? "Yes" : "No");
    }
    return 0;
}
