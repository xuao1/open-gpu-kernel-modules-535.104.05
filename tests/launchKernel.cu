#include <cuda_runtime.h>
#include <iostream>

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

    std::cout << "Before launch kernel..." << std::endl;
    std::cin.get();

    cudaLaunchKernel((void*)simpleKernel, grid, block, kernelArgs, 0, NULL);

    std::cout << "After launcher kernel..." << std::endl;
    std::cin.get();

    cudaFree(d_data);

    return 0;
}

