// 存放 CUDA 核心算子（GPU 端执行）

// src/cuda_kernels.cu
#include <cuda_runtime.h>
#include <iostream>
#include "cuda_kernels.h"

// ======================
// 内核函数定义
// ======================

// 元素加法 kernel
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// 元素乘法 kernel
__global__ void mul_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

// 向量点积 kernel (每个线程做部分求和)
__global__ void dot_kernel(const float* a, const float* b, float* partial_sum, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (i < n) val = a[i] * b[i];
    sdata[tid] = val;
    __syncthreads();

    // 归约求和 (block 内)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // 每个 block 输出一个部分和
    if (tid == 0)
        partial_sum[blockIdx.x] = sdata[0];
}

// ======================
// 启动封装函数
// ======================

void launch_add_kernel(float* a, float* b, float* out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, out, n);
    cudaDeviceSynchronize();
}

void launch_mul_kernel(float* a, float* b, float* out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads>>>(a, b, out, n);
    cudaDeviceSynchronize();
}

float launch_dot_kernel(float* a, float* b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    float* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));

    dot_kernel<<<blocks, threads, threads * sizeof(float)>>>(a, b, d_partial, n);

    // 将每个 block 的部分和拷回 CPU 进行最终求和
    std::vector<float> h_partial(blocks);
    cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_partial);

    float total = 0.0f;
    for (float v : h_partial) total += v;
    return total;
}
