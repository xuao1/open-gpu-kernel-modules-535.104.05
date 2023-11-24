#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
    // if (argc != 2) {
    //     fprintf(stderr, "Usage: %s <memory size in MiB>\n", argv[0]);
    //     return -1;
    // }

    // Parse memory size from command line
    // size_t bytes = (size_t)atoll(argv[1]) * 1024 * 1024;
    size_t bytes = 100 * 1024 * 1024;
    char *d_memory;

    printf("Trying to allocate and memset %zu bytes (%zu MiB) of GPU memory\n", bytes, bytes / (1024 * 1024));

    printf("Allocating %zu bytes (%zu MiB) of GPU memory\n", bytes, bytes / (1024 * 1024));
    cudaError_t status = cudaMalloc((void**)&d_memory, bytes);    
    printf("Allocated %zu bytes (%zu MiB) of GPU memory\n", bytes, bytes / (1024 * 1024));

    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        return -1;
    } 
    else printf("Successfully allocated %zu bytes (%zu MiB) of GPU memory\n", bytes, bytes / (1024 * 1024));

    status = cudaMemset(d_memory, 0, bytes);

    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(status));
        cudaFree(d_memory);
        return -1;
    }
    else printf("Successfully memset %zu bytes (%zu MiB) of GPU memory\n", bytes, bytes / (1024 * 1024));

    // printf("Successfully allocated and memset %zu bytes (%zu MiB) of GPU memory\n", bytes, bytes / (1024 * 1024));
    
    status = cudaFree(d_memory);

    if(status != cudaSuccess){
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    else printf("Successfully freed %zu bytes (%zu MiB) of GPU memory\n", bytes, bytes / (1024 * 1024));

    printf("Successfully allocated and memset %zu bytes (%zu MiB) of GPU memory\n", bytes, bytes / (1024 * 1024));

    return 0;
}
