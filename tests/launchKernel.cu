#include <cuda_runtime.h>
#include <iostream>
#include <unistd.h> 

__global__ void simpleKernel(int *data) {
    int idx = threadIdx.x;
    data[idx] = idx;
}

int main() {
    int *d_data;
    cudaMalloc(&d_data, sizeof(int) * 256);

    void *kernelArgs[] = { &d_data };
    dim3 grid(1);
    dim3 block(256);

    std::cout << "60 s..." << std::endl;
    sleep(60);

    simpleKernel<<<1, 256>>>(d_data);

    std::cout << "60 s..." << std::endl;
    sleep(60);

    cudaFree(d_data);

    return 0;
}

