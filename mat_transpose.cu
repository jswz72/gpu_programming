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

// Sequential matrix transpose
void mat_transpose(int *mat, int *res, int num_rows, int num_cols)
{
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            res[j * num_rows + i] = mat[i  * num_cols + j];
        }
    }
}

// Parallel matrix tranpose
__global__ void mat_transpose_kernel(int *mat, int *res) {
    int tile_dim = 32;
   __shared__ int smem[32 * 32];

   for (int block_iter = 0; block_iter < 512; block_iter++) {

       // num blocks can fit in "row"
       if (blockIdx.x < 32) {
           local_block_start = blockIdx.x * 32;
       }
       else {
           local_block_start = (tile_dim * tile_dim * 32) + blockIdx.x * 32;
       }
       int block_start = local_block_start + (block_iter * (tile_dim * tile_dim * 64));
       if (threadIdx.x < tile_dim) {
           local_thread_idx = threadIdx.x;
       } else {
           local_thread_idx = ((threadIdx.x / tile_dim) * 32) + (threadIdx.x % tile_dim)

       int block_start = (blockIdx.x * tile_dim);
       if (block_start >= 32) {
           block_start += 
       }
       + (32 * 32 * block_iter);
   }

   int local_tid = threadIdx.x;
   if (local_tid > tile_dim) {
       local_tid += (threadIdx.x / tile_dim) * tile_dim;
       local_tid += threadIdx % tile_dim;
   }

   int block_start = blockIdx.x * tile_dims;
    if (block_start > tile_dim) {
       block_start += blockIdx.x / 
       block_start += threadIdx % tile_dim;
   }
   int idx = blockIdx.x * tile_dims + threadIdx.x;
   if (idx > time_dims) 

}

int main (int args, char **argv) {
    int num_rows = 1024;
    int num_cols = 1024;

    int *a = (int *) malloc(sizeof(int) * num_rows * num_cols);
    int *c = (int *) malloc(sizeof(int) * num_rows * num_cols);

    srand(time(NULL));
    // Initialize matrix
    cout << "Matrix: " << num_rows << " x " << num_cols << endl;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            int el = rand() % 10;
            a[i * num_cols + j] = el;
        }
    }
    int *a_d, *c_d;
    HANDLE_ERR(cudaMalloc((void **) &a_d, sizeof (int) * num_rows * num_cols));
    HANDLE_ERR(cudaMalloc((void **) &c_d, sizeof (int) * num_rows * num_cols));

    HANDLE_ERR(cudaMemcpy (a_d, a, sizeof (int) * num_rows * num_cols, cudaMemcpyHostToDevice));
    mat_transpose_kernel <<< 64, 1024 >>> (a_d, c_d);

    HANDLE_ERR(cudaMemcpy (c, c_d, sizeof (int) * num_rows * num_cols, cudaMemcpyDeviceToHost));

    //Make sure parallel work is equal to sequential work (for testing)
    int *test_res = (int *) malloc(sizeof(int) * num_rows * num_cols);
    mat_transpose(a, test_res, num_rows, num_cols);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++){
            int idx = i * num_cols + j;
            if (c[idx] != test_res[idx]) {
                cout << "Not Equal: " << "Parallel work " << c[idx] 
                    << ", Sequential Work: " << test_res[idx] << endl;
            }
            assert(c[idx] == test_res[idx]);
        }
    }
}
