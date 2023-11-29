#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <unistd.h>

//CUDA RunTime API
#include <cuda_runtime.h>
//单个block大小
#define THREAD_NUM 256
///矩阵大小
#define MATRIX_SIZE 1024
///block个数
int blocks_num = (MATRIX_SIZE * MATRIX_SIZE + THREAD_NUM - 1) / THREAD_NUM;

void gemm_baseline(float* A, float* B, float* C) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            float sum = 0.0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                sum += A[i * MATRIX_SIZE + k] * B[k * MATRIX_SIZE + j];
            }
            C[i * MATRIX_SIZE + j] = sum;
        }
    }
}

__global__ static void CUDAkernal(const float* a, const float* b, float* c, int n)
{
    //block内的threadID
    const int tid = threadIdx.x;
    //blockID
    const int bid = blockIdx.x;
    //全局threadID
    const int idx = bid * THREAD_NUM + tid;
    // printf("%d ", idx);
    const int row = idx / n;
    const int column = idx % n;
    //计算矩阵乘法
    if (row < n && column < n)
    {
        float t = 0;
        for (int i = 0; i < n; i++)
        {
            t += a[row * n + i] * b[i * n + column];
        }
        c[row * n + column] = t;
    }
}

int main() 
{
    srand(time(NULL));
    //定义矩阵
    float *a, *b, *c;
    int n = MATRIX_SIZE;
    //分配主机端内存
    a = (float*)malloc(sizeof(float)* n * n); 
    b = (float*)malloc(sizeof(float)* n * n); 
    c = (float*)malloc(sizeof(float)* n * n); 
    float *v_c = (float*)malloc(sizeof(float)* n * n); 

    ///生成矩阵a, b
    for (int i = 0; i < n * n; i++) {
        a[i] = (float)rand() / (float)(RAND_MAX);
        b[i] = (float)rand() / (float)(RAND_MAX);
        c[i] = 0.0;
        v_c[i] = 0.0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float *cuda_a, *cuda_b, *cuda_c;
    //分配设备端显存 
    cudaMalloc((void**)&cuda_a, sizeof(float)* n * n);
    cudaMalloc((void**)&cuda_b, sizeof(float)* n * n);
    cudaMalloc((void**)&cuda_c, sizeof(float)* n * n);

    //cudaMemcpyHostToDevice - 从内存复制到显存
    //cudaMemcpyDeviceToHost - 从显存复制到内存
    cudaMemcpy(cuda_a, a, sizeof(float)* n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float)* n * n, cudaMemcpyHostToDevice);

    std::cout << "60s..." << std::endl;
    sleep(60);

    ///设备端函数
    CUDAkernal <<< blocks_num, THREAD_NUM, 0 >>>(cuda_a , cuda_b , cuda_c , n);

    std::cout << "60s..." << std::endl;
    sleep(60);

    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(c, cuda_c, sizeof(float)* n * n, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    /*
    gemm_baseline(a, b, v_c);

    int flag = 1;
    for(int i = 0; i < n * n; i++) {
        printf("%d %f %f\n", i, c[i], v_c[i]);
        if(abs(c[i] - v_c[i]) > 0.1){
            flag = 0;
        }
    }

	if(flag)  printf("Results are correct.\n");
	else printf("Results are wrong.\n");

    */

    float timecost;
    cudaEventElapsedTime(&timecost, start, stop);
    printf("CUDA time %.4fms\n", timecost);

    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
    free(a);
    free(b);
    free(c);
    free(v_c);

    return 0;
}