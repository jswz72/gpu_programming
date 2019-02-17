/**
  Jacob Sword
  Parallelized multiplication of matrix and vector of random values given matrix dimensions
**/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <time.h>
#include "./error_handler.h"

using std::cout;
using std::endl;

// Sequential implementation of matrix x vector
void mat_vec_mult(int *mat, int *vec, int *res, int num_rows, int num_cols)
{
    for(int i = 0; i < num_rows; i ++)
    {
        int temp_res = 0;
        for (int j = 0; j < num_cols; j ++)
        {
            temp_res += mat[i * num_cols + j] * vec[j];
        }

        res[i] = temp_res;
    }
}

// Parllel implementation of matrix x vector - 1 block per row
__global__ void mat_mult_kernel(int *mat, int *vec, int *res, int mat_rows, int mat_cols) {
    /*
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < mat_rows) {
        int res = 0;
        for (int i = 0; i < mat_cols; i++) {
            res += a[tid * mat_cols + i] * b[i];
        }
        c[tid] = res;
        tid += blockDim.x * gridDim.x;
    }
    */

    // el for each thread, shared per block
    // 128
    __shared__ int smem[128]; 
    // 1024 --> 8 iter of block
    for (int block_iter = 0; block_iter * gridDim.x < mat_rows; block_iter++) {
        // 128
        for (int thread_iter = 0; thread_iter * blockDim.x < mat_cols; thread_iter++) {
            int row = blockIdx.x * block_iter; //row
            int col = threadIdx.x * thread_iter; //col
            // load mult in shmem accounting for iters
            smem[threadIdx.x] = mat[row * mat_cols + col] * vec[col];
            __syncthreads();

            // parallel reduction
            for (int i = blockDim.x / 2; i > 0; i /= 2) {
                if (threadIdx.x < i) {
                    int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
                    smem[threadIdx.x] = temp;
                }
                __syncthreads();
            }
            // Load into ans
            res[row] = smem[0];
        }
    }
}

int main (int args, char **argv) {
    int num_rows = 1024;
    int num_cols = 512;

    int *a = (int *) malloc(sizeof(int) * num_rows * num_cols);
    int *b = new int[num_cols];
    int *c = new int[num_rows];

    srand(time(NULL));
    // Initialize matrix
    cout << "Matrix (a): " << num_rows << " x " << num_cols << endl;

    // Initialize vector
    cout << "Vector (b): " << num_cols << " x 1 " << endl;

    int *a_d, *b_d, *c_d;
    HANDLE_ERR(cudaMalloc((void **) &a_d, sizeof (int) * num_rows * num_cols));
    HANDLE_ERR(cudaMalloc((void **) &b_d, sizeof (int) * num_cols));
    HANDLE_ERR(cudaMalloc((void **) &c_d, sizeof (int) * num_rows));

    HANDLE_ERR(cudaMemcpy (a_d, a, sizeof (int) * num_rows * num_cols, cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy (b_d, b, sizeof (int) * num_cols, cudaMemcpyHostToDevice));
    mat_mult_kernel <<< 128, 128 >>> (a_d, b_d, c_d, num_rows, num_cols);

    cudaDeviceSynchronize();
    HANDLE_ERR(cudaMemcpy (c, c_d, sizeof (int) * num_rows, cudaMemcpyDeviceToHost));

    //Make sure parallel work is equal to sequential work (for testing)
    int *test_res = new int[num_rows];
    mat_vec_mult(a, b, test_res, num_rows, num_cols);
    for (int i = 0; i < num_rows; i++) {
        if (c[i] != test_res[i])
            cout << "wrong" << endl;
        assert(c[i] == test_res[i]);
    }
}
