// optimize sgemm

#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 
#include <torch/extension.h>
#include <torch/types.h>
#include <mma.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace nvcuda;

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

__global__ void f32x4_tf32x4_kernel(float *x, float *y, int N) {           //transform f32 to tf32
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = wmma::__float_to_tf32(reg_x.x);
    reg_y.y = wmma::__float_to_tf32(reg_x.y);
    reg_y.z = wmma::__float_to_tf32(reg_x.z);
    reg_y.w = wmma::__float_to_tf32(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

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

template<
    const int BLOCK_SIZE_M,  
    const int BLOCK_SIZE_K,  
    const int BLOCK_SIZE_N,  
    const int THREAD_SIZE_Y, 
    const int THREAD_SIZE_X
    > 
__global__ void sgemm( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    

    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];
    // registers load global memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4*ldg_num_a];
    float ldg_b_reg[4*ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;


    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = &A[(BLOCK_SIZE_M * by)* K];
    B = &B[BLOCK_SIZE_N * bx];

    // load A from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K )]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
    }
    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL, // col
                N )]);
    }
    __syncthreads();
    // load A from shared memory to register
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        // load next tile from global mem
        if(tile_idx< K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
            }
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL, // col
                    N )]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for(int j=0; j<BLOCK_SIZE_K-1; ++j){

            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
            }
            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
            }
            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        if(tile_idx < K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            __syncthreads();

            write_stage_idx ^= 1;
        }


        // load A from shared memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
        }
        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
        }
        //compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    }while(tile_idx< K);

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x+=4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}
template <
    const int BLOCKSIZE_M=128,  
    const int BLOCKSIZE_K=8,  
    const int BLOCKSIZE_N=128,  
    const int THREADSIZE_Y=8, 
    const int THREADSIZE_X=8
    >
__global__ void gemm_kernal(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int M, int N, int K) {
	
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


    float res[THREADSIZE_Y][THREADSIZE_X];
    for (int i = 0; i < THREADSIZE_Y; i++) {
        for (int j = 0; j < THREADSIZE_X; j++) {
            res[i][j] = 0.0;
        }
    }
    float reg_a[2][THREADSIZE_Y];
    float reg_b[2][THREADSIZE_X];

    const int ld_num_a = (BLOCKSIZE_M* BLOCKSIZE_K)/(THREAD_PER_BLOCK*4);
    const int ld_num_b = (BLOCKSIZE_K* BLOCKSIZE_N)/(THREAD_PER_BLOCK*4);
    float ld_reg_a[ld_num_a*4];
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
    const int a_tile_idx = (warp_id/4)*32 + ((lane_id%16)/2)*4;
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

template <
    const int BLOCK_SIZE_M=128,  
    const int BLOCK_SIZE_K=8,  
    const int BLOCK_SIZE_N=128,  
    const int THREAD_SIZE_Y=8, 
    const int THREAD_SIZE_X=8
    > 
__global__ void gemm_cp_sync( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K) {

    int bx = blockIdx.x;
    int by = blockIdx.y;


    int tx = threadIdx.x;
    int ty = threadIdx.y;


    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;


    const int tid = ty * THREAD_X_PER_BLOCK + tx;


    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X];
    #pragma unroll
    for(int i=0; i<THREAD_SIZE_Y; i++){
        #pragma unroll
        for(int j=0; j<THREAD_SIZE_X; j++){
            accum[i][j]=0.0;
        }
    }

    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
//    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4*ldg_num_a];
//    float ldg_b_reg[4*ldg_num_b];


    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;


    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

 
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = &A[(BLOCK_SIZE_M * by)* K];
    B = &B[BLOCK_SIZE_N * bx];


    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int a_tile_index =  warp_id/2*16 + lane_id/8*4;
    const int b_tile_index =  warp_id%2*32 + lane_id%8*4; 

    uint32_t load_b_ptr = __cvta_generic_to_shared(&Bs[0][B_TILE_ROW_START][B_TILE_COL]);
    
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        CP_ASYNC_CG(load_b_ptr+i*BLOCK_SIZE_N*4, &B[OFFSET(
            i + B_TILE_ROW_START, // row
            B_TILE_COL, // col
            N )], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K )]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
    }
    CP_ASYNC_WAIT_GROUP(0);

    __syncthreads();
    

    FETCH_FLOAT4(frag_a[0][0]) = FETCH_FLOAT4(As[0][0][a_tile_index]);
    FETCH_FLOAT4(frag_a[0][4]) = FETCH_FLOAT4(As[0][0][a_tile_index + 64]);
    

    FETCH_FLOAT4(frag_b[0][0]) = FETCH_FLOAT4(Bs[0][0][b_tile_index]);
    FETCH_FLOAT4(frag_b[0][4]) = FETCH_FLOAT4(Bs[0][0][b_tile_index + 64]);
    
    int write_stage_idx = 1;
    int tile_idx = 0;
    do{

        tile_idx += BLOCK_SIZE_K;

        if(tile_idx< K){

            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i+= B_TILE_ROW_STRIDE) {
                load_b_ptr = __cvta_generic_to_shared(&Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]);
                CP_ASYNC_CG(load_b_ptr+i*BLOCK_SIZE_N*4, &B[OFFSET(
                    i + B_TILE_ROW_START + tile_idx, // row
                    B_TILE_COL, // col
                    N )], 16);
            }
            CP_ASYNC_COMMIT_GROUP();

            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
            }

        }

        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for(int j=0; j<BLOCK_SIZE_K - 1; ++j){
          
            FETCH_FLOAT4(frag_a[(j+1)%2][0]) = FETCH_FLOAT4(As[load_stage_idx][(j+1)][a_tile_index]);
            FETCH_FLOAT4(frag_a[(j+1)%2][4]) = FETCH_FLOAT4(As[load_stage_idx][(j+1)][a_tile_index + 64]);

            FETCH_FLOAT4(frag_b[(j+1)%2][0]) = FETCH_FLOAT4(Bs[load_stage_idx][(j+1)][b_tile_index]);
            FETCH_FLOAT4(frag_b[(j+1)%2][4]) = FETCH_FLOAT4(Bs[load_stage_idx][(j+1)][b_tile_index + 64]);

            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        if(tile_idx < K){

            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();

            write_stage_idx ^= 1;
        }


        FETCH_FLOAT4(frag_a[0][0]) = FETCH_FLOAT4(As[load_stage_idx^1][0][a_tile_index]);
        FETCH_FLOAT4(frag_a[0][4]) = FETCH_FLOAT4(As[load_stage_idx^1][0][a_tile_index + 64]);

        FETCH_FLOAT4(frag_b[0][0]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][b_tile_index]);
        FETCH_FLOAT4(frag_b[0][4]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][b_tile_index + 64]);

        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    }while(tile_idx< K);
    
    const int c_block_row = a_tile_index;
    const int c_block_col = b_tile_index;


    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + c_block_row + i,
        BLOCK_SIZE_N * bx + c_block_col,
        N)]) = FETCH_FLOAT4(accum[i][0]);
    }

    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + c_block_row + i,
        BLOCK_SIZE_N * bx + c_block_col + 64,
        N)]) = FETCH_FLOAT4(accum[i][4]);
    }

    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + c_block_row + 64 + i,
        BLOCK_SIZE_N * bx + c_block_col,
        N)]) = FETCH_FLOAT4(accum[i+4][0]);
    }

    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + c_block_row + 64 + i,
        BLOCK_SIZE_N * bx + c_block_col + 64,
        N)]) = FETCH_FLOAT4(accum[i+4][4]);
    }
}


template <
    const int BLOCKSIZE_M=128,  
    const int BLOCKSIZE_K=8,  
    const int BLOCKSIZE_N=128,  
    const int THREADSIZE_Y=8, 
    const int THREADSIZE_X=8
    >
__global__ void gemm_block_swizzle(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int M, int N, int K) {
	
    __shared__ float As[2][BLOCKSIZE_K][BLOCKSIZE_M];
    __shared__ float Bs[2][BLOCKSIZE_K][BLOCKSIZE_N];           //double buffer
    
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 蛇形翻转：偶数行不动，奇数行镜像
    if ((by & 1) == 1) {
        bx = gridDim.x - bx - 1; // 镜像翻转
    }

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int THREAD_PER_BLOCK_X = BLOCKSIZE_N / THREADSIZE_X;
    const int THREAD_PER_BLOCK_Y = BLOCKSIZE_M / THREADSIZE_Y;
    const int THREAD_PER_BLOCK = THREAD_PER_BLOCK_X * THREAD_PER_BLOCK_Y;

    const int tid = ty*THREAD_PER_BLOCK_X + tx;


    float res[THREADSIZE_Y][THREADSIZE_X];
    for (int i = 0; i < THREADSIZE_Y; i++) {
        for (int j = 0; j < THREADSIZE_X; j++) {
            res[i][j] = 0.0;
        }
    }
    float reg_a[2][THREADSIZE_Y];
    float reg_b[2][THREADSIZE_X];

    const int ld_num_a = (BLOCKSIZE_M* BLOCKSIZE_K)/(THREAD_PER_BLOCK*4);
    const int ld_num_b = (BLOCKSIZE_K* BLOCKSIZE_N)/(THREAD_PER_BLOCK*4);
    float ld_reg_a[ld_num_a*4];
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
    const int a_tile_idx = (warp_id/4)*32 + ((lane_id%16)/2)*4;
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

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 8,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
            const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void sgemm_wmma(float *A, float *B, float *C,
                                                 int M, int N, int K){
    // 256 threads(8 warps) per block.
    const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;    //// BLOCK_SWIZZLE 0/1 控制是否使用 block swizzle
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
    constexpr int BK = WMMA_K;                             // 8
    __shared__ float s_a[K_STAGE][BM][BK], s_b[K_STAGE][BK][BN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int warp_m = warp_id / 2;      // 0,1,2,3
  const int warp_n = warp_id % 2;      // 0,1

  int load_smem_a_m = tid / 2;                // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4; // col 0,4

  int load_smem_b_k = tid / 32;       // row 0~7
  int load_smem_b_n = (tid % 32) * 4; // col 0,4,...,124,...

  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>       // 
      C_frag[WARP_TILE_M][WARP_TILE_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {          
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
    wmma::fill_fragment(C_frag[i][j], 0.0);
    }
}
    #pragma unroll           //数据预取
    for (int k = 0; k < (K_STAGE - 1); ++k) {         // 0, 1
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        __cvta_generic_to_shared(&s_a[k][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        __cvta_generic_to_shared(&s_b[k][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2); 
  __syncthreads();

  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
    int smem_sel = (k + 1) % K_STAGE;       //要加载计算的 smem 的索引
    int smem_sel_next = k % K_STAGE;        //用于load下一个seme的索引


    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    // load stage 2, k start from 2
    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
        &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
        &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        B_frag[WARP_TILE_N];

// compute stage 0
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0],      //load as
                             BK );
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {

      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n],     //load bs
                             BN);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);      //compute
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }
    // wait for the last k stage to finish loading.
   if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  // processing last (K_STAGE-1) k iters.
  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          B_frag[WARP_TILE_N];

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[stage_sel][warp_smem_a_m][0],
                               BK );
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[stage_sel][0][warp_smem_b_n],
                               BN );
      }

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
  }

//store back to C matrix.
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m =
          by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n =
          bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n,
                              C_frag[i][j], N, wmma::mem_row_major);
    }
  }

}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define LAUNCH_SWIZZLE_KERNEL(stages, stride)                      \
  {                                                                            \
    const int N_SWIZZLE = (N + (stride) - 1) / (stride);                       \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),  \
              N_SWIZZLE);                                                      \
    sgemm_wmma<                          \
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,         \
        WARP_TILE_N, (stages), true>                             \
        <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),             \
                          reinterpret_cast<float *>(b.data_ptr()),             \
                          reinterpret_cast<float *>(c.data_ptr()), M, N, K);   \
  }

  #define LAUNCH_NO_SWIZZLE_KERNEL(stages)                           \
  {                                                                            \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                               \
    sgemm_wmma<                          \
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,         \
        WARP_TILE_N, (stages), false>                            \
        <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),             \
                          reinterpret_cast<float *>(b.data_ptr()),             \
                          reinterpret_cast<float *>(c.data_ptr()), M, N, K);   \
  }

void naive_sgemm_test(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    dim3 blockSize(16, 16);  // 16x16 线程块
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);

    gemm_baseline_kernel<<<gridSize, blockSize>>>(
        reinterpret_cast<float *>(a.data_ptr()),
        reinterpret_cast<float *>(b.data_ptr()),
        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemmopt_test(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    constexpr int BLOCK_SIZE_M = 128;
    constexpr int BLOCK_SIZE_K = 8;
    constexpr int BLOCK_SIZE_N = 128;
    constexpr int THREAD_SIZE_X = 8;
    constexpr int THREAD_SIZE_Y = 8;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    
    sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X>
        <<<dimGrid, dimBlock>>>(reinterpret_cast<float *>(a.data_ptr()),
                                reinterpret_cast<float *>(b.data_ptr()),
                                reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void gemm_bank_free_test(torch::Tensor a,torch::Tensor b,torch::Tensor c){
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    constexpr int BLOCK_SIZE_M = 128;
    constexpr int BLOCK_SIZE_K = 8;
    constexpr int BLOCK_SIZE_N = 128;
    constexpr int THREAD_SIZE_X = 8;
    constexpr int THREAD_SIZE_Y = 8;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    gemm_kernal<128, 8, 128, 8, 8>
    <<<dimGrid,dimBlock>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void gemm_block_swizzle_test(torch::Tensor a,torch::Tensor b,torch::Tensor c){
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    constexpr int BLOCK_SIZE_M = 128;
    constexpr int BLOCK_SIZE_K = 8;
    constexpr int BLOCK_SIZE_N = 128;
    constexpr int THREAD_SIZE_X = 8;
    constexpr int THREAD_SIZE_Y = 8;
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    gemm_block_swizzle<128, 8, 128, 8, 8>
    <<<dimGrid,dimBlock>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}
void gemm_cp_sync_test(torch::Tensor a,torch::Tensor b,torch::Tensor c){
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
   constexpr int BLOCK_SIZE_M = 128;
    constexpr int BLOCK_SIZE_K = 8;
    constexpr int BLOCK_SIZE_N = 128;
    constexpr int THREAD_SIZE_X = 8;
    constexpr int THREAD_SIZE_Y = 8;
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    gemm_cp_sync<128, 8, 128, 8, 8>
    <<<dimGrid,dimBlock>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}
void sgemm_wmma_test(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages=2, bool block_swizzle=false,int swizzle_stride=256){
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    const int Na = M * K;
    const int Nb = K * N;
    constexpr int T = 256;

    f32x4_tf32x4_kernel<<<((Na + T * 4 - 1) / (T * 4)), T>>>(
      reinterpret_cast<float *>(a.data_ptr()),
      reinterpret_cast<float *>(a.data_ptr()), Na);

    f32x4_tf32x4_kernel<<<((Nb + T * 4 - 1) / (T * 4)), T>>>(
      reinterpret_cast<float *>(b.data_ptr()),
      reinterpret_cast<float *>(b.data_ptr()), Nb);
    
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 8;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;

    constexpr int NUM_THREADS =
      (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
  constexpr int BK = WMMA_K;
  if(block_swizzle){
    switch (stages) {
        case 2:
            LAUNCH_SWIZZLE_KERNEL(2, swizzle_stride);
            break;
        case 3:
            LAUNCH_SWIZZLE_KERNEL(3, swizzle_stride);
            break;
        case 4:
            LAUNCH_SWIZZLE_KERNEL(4, swizzle_stride);
            break;
    }
  }else{
    switch (stages) {
        case 2:
            LAUNCH_NO_SWIZZLE_KERNEL(2);
            break;
        case 3:
            LAUNCH_NO_SWIZZLE_KERNEL(3);
            break;
        case 4:
            LAUNCH_NO_SWIZZLE_KERNEL(4);
            break;
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(naive_sgemm_test);
    TORCH_BINDING_COMMON_EXTENSION(sgemmopt_test);
    TORCH_BINDING_COMMON_EXTENSION(gemm_bank_free_test);
    TORCH_BINDING_COMMON_EXTENSION(gemm_block_swizzle_test);
    TORCH_BINDING_COMMON_EXTENSION(gemm_cp_sync_test);
    TORCH_BINDING_COMMON_EXTENSION(sgemm_wmma_test);
}