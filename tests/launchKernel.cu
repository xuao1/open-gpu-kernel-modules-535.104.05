#include <cuda_runtime.h>
#include <iostream>

__global__ void simpleKernel(int *data) {
    int idx = threadIdx.x;
    data[idx] = idx;
}

int main() {
    int *d_data;
    cudaMalloc(&d_data, sizeof(int) * 256);

    std::cout << "Before launch kernel..." << std::endl;
    std::cin.get();

    simpleKernel<<<1, 256>>>(d_data);

    std::cout << "After launcher kernel..." << std::endl;
    std::cin.get();

    cudaFree(d_data);

    return 0;
}
