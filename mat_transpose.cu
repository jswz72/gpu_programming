/**
  Jacob Sword
  Parallelized multiplication of matrix and vector of random values given matrix dimensions
**/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include "wtime.h"
#include "./error_handler.h"

using std::cout;
using std::endl;

// Parallel matrix tranpose
__global__ void mat_transpose_kernel(int *mat, int *res) {
    // Square tile
    int tile_dim = 32;
    // 32 Blocks across for 1024 mat
    int blocks_per_row = 32;

    __shared__ int smem[32 * 32];

    int rows_per_block_iter = 64;
    // Each iter has 2 block-rows
    for (int block_iter = 0; block_iter < 16; block_iter++) {
        int tile_row = blockIdx.x / blocks_per_row;
        int tile_col = blockIdx.x % blocks_per_row;

        int intile_row = threadIdx.x / tile_dim;
        int intile_col = threadIdx.x % tile_dim;

        int read_row = (tile_row * tile_dim) + intile_row + (rows_per_block_iter * block_iter);
        int read_col = (tile_col * tile_dim) + intile_col;

        int write_row = (tile_col * tile_dim) + intile_row;
        int write_col = (tile_row * tile_dim) + intile_col + (rows_per_block_iter * block_iter);
        if (write_col >= 1024) printf("c: %d\n", write_col);
        if (write_row >= 1024) printf("r: %d\n", write_row);

        int shm_row = threadIdx.x / 32;
        int shm_col = threadIdx.x % 32;

        smem[(shm_row * tile_dim) + shm_col] = mat[(read_row * 1024) + read_col];
        __syncthreads();
        res[(write_row * 1024) + write_col] = smem[(shm_col * tile_dim) + shm_row];
    }
    /*int tile_row = blockIdx.x / 8;
    int tile_col = blockIdx.x % 8;

    int intile_row = threadIdx.x / 32;
    int intile_col = threadIdx.x % 32;

    int read_row = tile_row * 32 + intile_row;
    int read_col = tile_col * 32 + intile_col;

    int write_row = tile_col * 32 + intile_row;
    int write_col = tile_row * 32 + intile_col;

    int shm_row = threadIdx.x / 32;
    int shm_col = threadIdx.x % 32;

    smem[shm_row * 32 + shm_col] = mat[read_row * 256 + read_col];
    __syncthreads();
    res[write_row * 256 + write_col] = smem[shm_col * 32 + shm_row];*/
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

    double starttime = wtime();
    mat_transpose_kernel <<< 64, 1024 >>> (a_d, c_d);
    cudaDeviceSynchronize();
    double algotime = wtime() - starttime;
    cout << "Time: " << algotime << endl;

    HANDLE_ERR(cudaMemcpy (c, c_d, sizeof (int) * num_rows * num_cols, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++)
            assert(a[i * num_cols + j] == c[j * num_cols + i]);
    }
}
