#include <stdio.h>
#include <cuda.h>

__global__ void testKernel(int *a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    a[idx] = a[idx] + 1;
}

int main() {
    int *d_a;
    int size = 256 * sizeof(int);
    int *a = (int*)malloc(size);

    for (int i = 0; i < 256; i++) {
        a[i] = i;
    }

    cudaMalloc((void **)&d_a, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    testKernel<<<1, 256>>>(d_a);

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    cudaFree(d_a);
    free(a);
    return 0;
}
