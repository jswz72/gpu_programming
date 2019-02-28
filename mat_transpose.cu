/**
  Jacob Sword
  Parallelized matrix tranpose
  Comparison between naive and coalesced transpose
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

// Parallel matrix tranpose using coalesced write for 1024x1024 matrix
__global__ void mat_transpose_coalesced_kernel(int *mat, int *res) {
    // Square tile
    int tile_dim = 32;
    // 32 Blocks across for 1024 mat
    int blocks_per_row = 32;

    __shared__ int smem[32 * 32];

    int rows_per_block_iter = 64;
    // Each iter has 2 "block-rows"
    for (int block_iter = 0; block_iter < 16; block_iter++) {
        int tile_row = blockIdx.x / blocks_per_row;
        int tile_col = blockIdx.x % blocks_per_row;

        int intile_row = threadIdx.x / tile_dim;
        int intile_col = threadIdx.x % tile_dim;

        int read_row = (tile_row * tile_dim) + intile_row + (rows_per_block_iter * block_iter);
        int read_col = (tile_col * tile_dim) + intile_col;

        int write_row = (tile_col * tile_dim) + intile_row;
        int write_col = (tile_row * tile_dim) + intile_col + (rows_per_block_iter * block_iter);


        smem[(intile_row * tile_dim) + intile_col] = mat[(read_row * 1024) + read_col];
        __syncthreads();
        res[(write_row * 1024) + write_col] = smem[(intile_col * tile_dim) + intile_row];
    }
}

__global__ void mat_transpose_regular_kernel(int *mat, int *res) {
    // Square tile
    int tile_dim = 32;
    // 32 Blocks across for 1024 mat
    int blocks_per_row = 32;

    int rows_per_block_iter = 64;
    // Each iter has 2 "block-rows"
    for (int block_iter = 0; block_iter < 16; block_iter++) {
        int tile_row = blockIdx.x / blocks_per_row;
        int tile_col = blockIdx.x % blocks_per_row;

        int intile_row = threadIdx.x / tile_dim;
        int intile_col = threadIdx.x % tile_dim;

        int my_row = (tile_row * tile_dim) + intile_row + (rows_per_block_iter * block_iter);
        int my_col = (tile_col * tile_dim) + intile_col;

        res[(my_col * 1024) + my_row] = mat[(my_row * 1024) + my_col];
    }
}

void fill_matrix(int *mat, int num_rows, int num_cols) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            int el = rand() % 10;
            mat[i * num_cols + j] = el;
        }
    }
}

int main (int args, char **argv) {
    int num_rows = 1024;
    int num_cols = 1024;

    int *a = (int *) malloc(sizeof(int) * num_rows * num_cols);
    int *reg_result = (int *) malloc(sizeof(int) * num_rows * num_cols);
    int *coal_result = (int *) malloc(sizeof(int) * num_rows * num_cols);

    srand(time(NULL));
    fill_matrix(a, num_rows, num_cols);

    int *a_d, *b_d, *c_d;
    HANDLE_ERR(cudaMalloc((void **) &a_d, sizeof (int) * num_rows * num_cols));
    HANDLE_ERR(cudaMalloc((void **) &b_d, sizeof (int) * num_rows * num_cols));
    HANDLE_ERR(cudaMalloc((void **) &c_d, sizeof (int) * num_rows * num_cols));

    HANDLE_ERR(cudaMemcpy (a_d, a, sizeof (int) * num_rows * num_cols, cudaMemcpyHostToDevice));

    double starttime = wtime();
    mat_transpose_regular_kernel <<< 64, 1024 >>> (a_d, b_d);
    cudaDeviceSynchronize();
    double algotime = wtime() - starttime;
    cout << "Regular Transpose Time: " << algotime << endl;

    HANDLE_ERR(cudaMemcpy (reg_result, b_d, sizeof (int) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++)
            assert(a[i * num_cols + j] == reg_result[j * num_cols + i]);
    }

    // Refill
    fill_matrix(a, num_rows, num_cols);
    HANDLE_ERR(cudaMemcpy (a_d, a, sizeof (int) * num_rows * num_cols, cudaMemcpyHostToDevice));

    starttime = wtime();
    mat_transpose_coalesced_kernel<<< 64, 1024 >>> (a_d, c_d);
    cudaDeviceSynchronize();
    algotime = wtime() - starttime;
    cout << "Coalesced Transpose Time: " << algotime << endl;

    HANDLE_ERR(cudaMemcpy (coal_result, c_d, sizeof (int) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++)
            assert(a[i * num_cols + j] == coal_result[j * num_cols + i]);
    }
}
