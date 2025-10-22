#include "cutlass/cutlass.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])        //向量化
#define OFFSET(row, col, a) ((row) * (a) + (col))                       //计算偏移量

__global__ void gemm_baseline_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template<const int BLOCKSIZE_M,const int BLOCKSIZE_K,const int BLOCKSIZE_N,const int THREADSIZE_Y,const int THREADSIZE_X>
__global__ void gemm_kernel(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int M, int N, int K) {
	
    __shared__ float As[2][BLOCKSIZE_K][BLOCKSIZE_M];
    __shared__ float Bs[2][BLOCKSIZE_K][BLOCKSIZE_N];           //double buffer
    
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int THREAD_PER_BLOCK_X = BLOCKSIZE_N / THREADSIZE_X;
    const int THREAD_PER_BLOCK_Y = BLOCKSIZE_M / THREADSIZE_Y;
    const int THREAD_PER_BLOCK = THREAD_PER_BLOCK_X * THREAD_PER_BLOCK_Y;
    const int tid = ty*THREAD_PER_BLOCK_X + tx;


    float res[THREADSIZE_Y][THREADSIZE_X];              //  result register
    for (int i = 0; i < THREADSIZE_Y; i++) {
        for (int j = 0; j < THREADSIZE_X; j++) {
            res[i][j] = 0.0;
        }
    }
    float reg_a[2][THREADSIZE_Y];                       //  register for As
    float reg_b[2][THREADSIZE_X];                       //  register for Bs

    //to load A and B into shared memory
    const int ld_num_a = (BLOCKSIZE_M* BLOCKSIZE_K)/(THREAD_PER_BLOCK*4);       //load的次数
    const int ld_num_b = (BLOCKSIZE_K* BLOCKSIZE_N)/(THREAD_PER_BLOCK*4);
    float ld_reg_a[ld_num_a*4];                                  //register for loading A
    float ld_reg_b[ld_num_b*4];

    const int A_tile_thread_per_row = BLOCKSIZE_K /4 ;
    const int B_tile_thread_per_row = BLOCKSIZE_N /4 ;
    
    const int A_tile_row_start = tid / A_tile_thread_per_row;
    const int A_tile_col_start = tid % A_tile_thread_per_row*4;
    
    const int B_tile_row_start = tid / B_tile_thread_per_row;
    const int B_tile_col_start = tid % B_tile_thread_per_row*4;

    const int A_tile_row_stride = THREAD_PER_BLOCK / A_tile_thread_per_row;
    const int B_tile_row_stride = THREAD_PER_BLOCK / B_tile_thread_per_row;

    A = &A[by*BLOCKSIZE_M*K];
    B = &B[bx*BLOCKSIZE_N];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int a_tile_idx = (warp_id/4)*32 + ((lane_id%16)/2)*4;                 //z-order式排列
    const int b_tile_idx = (warp_id%4)*16 + (lane_id/16)*8 + (lane_id%2)*4;

    #pragma unroll                                                      // load the first tile
    for(int i = 0;i<BLOCKSIZE_M;i+=A_tile_row_stride) {
        int reg_idx = i / A_tile_row_stride*4;
        FLOAT4(ld_reg_a[reg_idx]) = FLOAT4(A[OFFSET(i + A_tile_row_start, A_tile_col_start, K)]);
        As[0][A_tile_col_start][A_tile_row_start + i] = ld_reg_a[reg_idx];
        As[0][A_tile_col_start + 1][A_tile_row_start + i] = ld_reg_a[reg_idx + 1];
        As[0][A_tile_col_start + 2][A_tile_row_start + i] = ld_reg_a[reg_idx + 2];
        As[0][A_tile_col_start + 3][A_tile_row_start + i] = ld_reg_a[reg_idx + 3];
    }

    #pragma unroll
    for(int i = 0;i<BLOCKSIZE_K;i+=B_tile_row_stride){
        FLOAT4(Bs[0][B_tile_row_start+i][B_tile_col_start]) = FLOAT4(B[OFFSET(i + B_tile_row_start, B_tile_col_start, N)]);
    }
    __syncthreads();
    FLOAT4(reg_a[0][0]) = FLOAT4(As[0][0][a_tile_idx]);
    FLOAT4(reg_a[0][4]) = FLOAT4(As[0][0][a_tile_idx + 64]);

    FLOAT4(reg_b[0][0]) = FLOAT4(Bs[0][0][b_tile_idx]);
    FLOAT4(reg_b[0][4]) = FLOAT4(Bs[0][0][b_tile_idx + 64]);

    int write_flag = 1;
    int tile_idx = 0;
    //大迭代
    do
    {
        tile_idx += BLOCKSIZE_K;
        if (tile_idx < K) {
            #pragma unroll
            for(int i = 0;i<BLOCKSIZE_M;i+=A_tile_row_stride)
            {
                int reg_idx = i / A_tile_row_stride*4;
                FLOAT4(ld_reg_a[reg_idx]) = FLOAT4(A[OFFSET(i + A_tile_row_start, A_tile_col_start + tile_idx, K)]);
            }
            #pragma unroll
            for(int i = 0;i<BLOCKSIZE_K;i+=B_tile_row_stride){
                int reg_idx = i / B_tile_row_stride*4;
                FLOAT4(ld_reg_b[reg_idx]) = FLOAT4(B[OFFSET(i + B_tile_row_start + tile_idx, B_tile_col_start, N)]);
            }
        }
        int load_flag = write_flag ^ 1;
        //小迭代
        #pragma unroll
        for(int i = 0;i<BLOCKSIZE_K-1;++i)
        {
            FLOAT4(reg_a[(i+1)%2][0]) = FLOAT4(As[load_flag][i+1][a_tile_idx]); 
            FLOAT4(reg_a[(i+1)%2][4]) = FLOAT4(As[load_flag][i+1][a_tile_idx + 64]);
            
            FLOAT4(reg_b[(i+1)%2][0]) = FLOAT4(Bs[load_flag][i+1][b_tile_idx]);
            FLOAT4(reg_b[(i+1)%2][4]) = FLOAT4(Bs[load_flag][i+1][b_tile_idx + 64]);

            #pragma unroll
            for(int y = 0;y<THREADSIZE_Y;++y){
                #pragma unroll
                for(int x = 0;x<THREADSIZE_X;++x){
                    res[y][x] += reg_a[i%2][y] * reg_b[i%2][x];
                }
            }
        }

        if(tile_idx<K){
            #pragma unroll
            for(int i = 0;i<BLOCKSIZE_M;i+=A_tile_row_stride)
            {
                int reg_idx = i / A_tile_row_stride*4;
                As[write_flag][A_tile_col_start][A_tile_row_start + i] = ld_reg_a[reg_idx];
                As[write_flag][A_tile_col_start + 1][A_tile_row_start + i] = ld_reg_a[reg_idx + 1];
                As[write_flag][A_tile_col_start + 2][A_tile_row_start + i] = ld_reg_a[reg_idx + 2];
                As[write_flag][A_tile_col_start + 3][A_tile_row_start + i] = ld_reg_a[reg_idx + 3];
            }

            #pragma unroll
            for(int i = 0;i<BLOCKSIZE_K;i+=B_tile_row_stride){
                int reg_idx = i / B_tile_row_stride*4;
                FLOAT4(Bs[write_flag][B_tile_row_start+i][B_tile_col_start]) = FLOAT4(ld_reg_b[reg_idx]);
            }

            __syncthreads();
            write_flag ^= 1; 
        }
        //最后一次小迭代
        FLOAT4(reg_a[0][0]) = FLOAT4(As[load_flag^1][0][a_tile_idx]);
        FLOAT4(reg_a[0][4]) = FLOAT4(As[load_flag^1][0][a_tile_idx + 64]);

        FLOAT4(reg_b[0][0]) = FLOAT4(Bs[load_flag^1][0][b_tile_idx]);
        FLOAT4(reg_b[0][4]) = FLOAT4(Bs[load_flag^1][0][b_tile_idx + 64]);

        #pragma unroll
        for(int y = 0;y<THREADSIZE_Y;++y){
            #pragma unroll
            for(int x = 0;x<THREADSIZE_X;++x){
                res[y][x] += reg_a[1][y] * reg_b[1][x];      
            }
        }
    } while (tile_idx<K);

    //store C
    for(int i = 0;i<4;++i)
    {
        FLOAT4(C[OFFSET(by * BLOCKSIZE_M + a_tile_idx + i, bx * BLOCKSIZE_N + b_tile_idx, N)]) = FLOAT4(res[i][0]);
    }
    for(int i = 0;i<4;++i)
    {
        FLOAT4(C[OFFSET(by * BLOCKSIZE_M + a_tile_idx + i, bx * BLOCKSIZE_N + b_tile_idx + 64, N)]) = FLOAT4(res[i][4]);
    }
    for(int i = 0;i<4;++i)
    {
        FLOAT4(C[OFFSET(by * BLOCKSIZE_M + a_tile_idx + i + 64, bx * BLOCKSIZE_N + b_tile_idx, N)]) = FLOAT4(res[i + 4][0]);
    }
    for(int i = 0;i<4;++i)
    {
        FLOAT4(C[OFFSET(by * BLOCKSIZE_M + a_tile_idx + i + 64, bx * BLOCKSIZE_N + b_tile_idx + 64, N)]) = FLOAT4(res[i + 4][4]);
    }

}

void init_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
    }
}

void cpu_gemm_reference(float *A, float *B, float *C_ref, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }
}

bool verify_result(float *C1, float *C2, int size, float epsilon = 1e-4f) {
    for (int i = 0; i < size; i++) {
        if (std::abs(C1[i] - C2[i]) > epsilon) {
            std::cout << "Failed at " << i << ", GPU=" << C1[i] 
                     << ", CPU=" << C2[i] << ", diff=" << std::abs(C1[i] - C2[i]) << std::endl;
            return false;
        }
    }
    return true;
}

void test_cutlass() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        std::cout << "No CUDA device found" << std::endl;
    } else {
        std::cout << "Found " << deviceCount << " CUDA devices" << std::endl;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device name: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    }
}

void GEMM_Baseline(float *A, float *B, float *C, int M, int N, int K,double *run_time) {
    // 设备指针
    float *d_A, *d_B, *d_C;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // 分配GPU内存
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    
    // 设置线程块和网格大小
    dim3 blockSize(16, 16);  // 16x16 线程块
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start);
    
    // 启动kernel
    gemm_baseline_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    
    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 计算执行时间
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    run_time[0] = milliseconds;
    // 将结果复制回主机
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // 计算性能指标
    double flops = 2.0 * M * N * K; // 每次乘加算2次浮点运算
    double gflops = flops / (milliseconds * 1e6); // 转换为GFLOPS
    
    std::cout << "Baseline GEMM [" << M << "x" << K << "] x [" << K << "x" << N 
              << "] = [" << M << "x" << N << "]" << std::endl;
    std::cout << "执行时间: " << milliseconds << " ms" << std::endl;
    std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
    
    // 清理GPU内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void my_gemm(float *A, float *B, float *C, int M, int N, int K,double *run_time) {
    float *d_A, *d_B, *d_C;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // 分配GPU内存
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;

    dim3 blockSize(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 gridSize(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    gemm_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X>
        <<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 计算执行时间
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    run_time[1] = milliseconds;

    // 将结果复制回主机
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 计算性能指标
    double flops = 2.0 * M * N * K; // 每次乘加算2次浮点运算
    double gflops = flops / (milliseconds * 1e6); // 转换为GFLOPS

    std::cout << "My GEMM [" << M << "x" << K << "] x [" << K << "x" << N 
              << "] = [" << M << "x" << N << "]" << std::endl;
    std::cout << "执行时间: " << milliseconds << " ms" << std::endl;
    std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
    

    // 清理GPU内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {
    test_cutlass();
    srand(42);
    int test_sizes[] = {256, 512, 1024,2048};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    double run_time[2] = {0.0, 0.0}; // 0: Baseline, 1: My GEMM
    for (int t = 0; t < num_tests; t++) {
        int M = test_sizes[t];
        int N = test_sizes[t];
        int K = test_sizes[t];
        
        std::cout << "\n--- 测试矩阵大小: " << M << "x" << K << " x " 
                  << K << "x" << N << " ---" << std::endl;
        
        // 分配主机内存
        float *A = new float[M * K];
        float *B = new float[K * N];
        float *C = new float[M * N];
        float *C_ref = new float[M * N];
        
        // 初始化输入矩阵
        init_matrix(A, M * K);
        init_matrix(B, K * N);
        
        // CPU参考实现（用于验证正确性）
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_gemm_reference(A, B, C_ref, M, N, K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
        
        std::cout << "CPU参考时间: " << cpu_duration.count() << " ms" << std::endl;
        
        // CUDA Baseline实现
        GEMM_Baseline(A, B, C, M, N, K, run_time);

         // 验证结果正确性
        bool correct = verify_result(C, C_ref, M * N);
        std::cout << "结果验证: " << (correct ? "\033[1;32m通过\033[0m" : "\033[1;31m失败\033[0m") << std::endl;

        if (!correct) {
            std::cout << "\033[1;31m警告: 结果不正确, 请检查实现! \033[0m" << std::endl;
        }

        //my_gemm函数调用
        my_gemm(A, B, C, M, N, K, run_time);

        // 验证结果正确性
        correct = verify_result(C, C_ref, M * N);
        std::cout << "结果验证: " << (correct ? "\033[1;32m通过\033[0m" : "\033[1;31m失败\033[0m") << std::endl;

        if (!correct) {
            std::cout << "\033[1;31m警告: 结果不正确, 请检查实现! \033[0m" << std::endl;
        }

        std::cout << "加速比 = " << run_time[0] / run_time[1] << std::endl;
        
        // 释放内存
        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_ref;
    }
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
	// TODO: 实现你自己的Kernel，并计算和 Baseline 的加速比
    //GEMM_Kernel(A, B, C, M, N, K);
    return 0;
}
