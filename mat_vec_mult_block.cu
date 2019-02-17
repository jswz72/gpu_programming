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

// Parallel implementation of matrix x vector - 1 block per row: matrix is 1024 x 512
__global__ void mat_vec_mult_fixed_dims(int *mat, int *vec, int *res) {
    int mat_rows = 1024;
    int mat_cols = 512;
    // El for each thread, shared per block
    __shared__ int smem[128];
    for (int block_i = 0; block_i * gridDim.x < mat_rows; block_i++) {
        int row = blockIdx.x + (block_i * gridDim.x);
        int row_total = 0;
        for (int thread_i = 0; thread_i * blockDim.x < mat_cols; thread_i++) {
            int col = threadIdx.x + (thread_i * blockDim.x);
            // Load mult in shmem
            smem[threadIdx.x] = mat[row * mat_cols + col] * vec[col];
            __syncthreads();

            // Parallel reduction
            for (int i = blockDim.x / 2; i > 0; i /= 2) {
                if (threadIdx.x < i) {
                    int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
                    smem[threadIdx.x] = temp;
                }
                __syncthreads();
            }
            // Only 1 thread needs to do this
            if (threadIdx.x == 0)
                row_total += smem[threadIdx.x];
        }
        // Load into ans (single thread)
        if (threadIdx.x == 0)
            res[row] = row_total;
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
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            int el = rand() % 10;
            a[i * num_cols + j] = el;
        }
    }

    // Initialize vector
    cout << "Vector (b): " << num_cols << " x 1" << endl;
    for (int i = 0; i < num_cols; i++) {
        int el = rand() % 5;
        b[i] = el;
    }

    int *a_d, *b_d, *c_d;
    HANDLE_ERR(cudaMalloc((void **) &a_d, sizeof (int) * num_rows * num_cols));
    HANDLE_ERR(cudaMalloc((void **) &b_d, sizeof (int) * num_cols));
    HANDLE_ERR(cudaMalloc((void **) &c_d, sizeof (int) * num_rows));

    HANDLE_ERR(cudaMemcpy (a_d, a, sizeof (int) * num_rows * num_cols, cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy (b_d, b, sizeof (int) * num_cols, cudaMemcpyHostToDevice));
    mat_vec_mult_fixed_dims <<< 128, 128 >>> (a_d, b_d, c_d);

    HANDLE_ERR(cudaMemcpy (c, c_d, sizeof (int) * num_rows, cudaMemcpyDeviceToHost));

    //Make sure parallel work is equal to sequential work (for testing)
    int *test_res = new int[num_rows];
    mat_vec_mult(a, b, test_res, num_rows, num_cols);
    for (int i = 0; i < num_rows; i++) {
        if (c[i] != test_res[i]) {
            cout << "Not Equal: " << "Parallel work " << c[i] 
                << ", Sequential Work: " << test_res[i] << endl;
        }
        assert(c[i] == test_res[i]);
    }
}
