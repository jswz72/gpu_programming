/**
  Jacob Sword
  Parallelized multiplication of matrix and matrix of random values given fixed matrix dimensions
  Comparisons in speed between regular block by block mult, using padding to combat unalignment,
  and using mtrix transpose to combat strided memory access.
  Matrix dimensions are (256 rows by 240 cols) x (240 rows by 512 cols)
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

const int A_ROWS = 256;
const int A_COLS = 240;

const int B_ROWS = 240;
const int B_COLS = 512;

const int C_ROWS = A_ROWS;
const int C_COLS = B_COLS;


//Sequential mat_mult for testing
void mat_mult(int *mat_a, int *mat_b, int *result, int a_rows, int a_cols, int b_cols, int padding = 0)
{
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            int temp_res = 0;
            for (int k = 0; k < a_cols; k++) {
                temp_res += mat_a[i * (a_cols + padding) + k] * mat_b[k * b_cols + j];
            }
            result[i * b_cols + j] = temp_res;
        }
    }
}

/*Parallel implementation of matrix a x matrix b
 * 1 block per row
 * matrix A is 256 x 240, matrix b is 240 * 512
 * resultant matrix is 256 rows x 512 cols
 * Supports optional padding of matrix a
 */
__global__ void mat_mult_kernel(int *mat_a, int *mat_b, int *res, 
        int a_rows, int a_cols, int b_cols, int padding = 0) {
    // El for each thread, shared per block
    __shared__ int smem[128];
    for (int row_block = 0; row_block * gridDim.x < a_rows; row_block++) {

        int a_row = blockIdx.x + (row_block * gridDim.x);
        for (int b_col = 0; b_col < b_cols; b_col++) {

            int total = 0;
            for (int thread_i = 0; thread_i * blockDim.x < a_cols; thread_i++) {

                int thread_col = threadIdx.x + (thread_i * blockDim.x);
                // Need to check because 240 not even multiple of 128
                if (thread_col >= a_cols)
                    smem[threadIdx.x] = 0;
                else
                    smem[threadIdx.x] = mat_a[a_row * (a_cols + padding) + thread_col] * mat_b[thread_col * b_cols + b_col];
                __syncthreads();

                //Parallel reduction
                for (int i = blockDim.x / 2; i > 0; i /= 2) {
                    if (threadIdx.x < i) {
                        int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
                        smem[threadIdx.x] = temp;
                    }
                    __syncthreads();
                }
                if (threadIdx.x == 0) {
                    total += smem[threadIdx.x];
                }
            }
            if (threadIdx.x == 0) {
                res[a_row * b_cols + b_col] = total;
            }
        }
    }
}


__global__ void mat_mult_transposed_kernel(int *mat_a, int *mat_b, int *res) {
    int B_TRANS_ROWS = B_COLS;
    int B_TRANS_COLS = B_ROWS;
    // El for each thread, shared per block
    __shared__ int smem[128];
    for (int row_block = 0; row_block * gridDim.x < A_ROWS; row_block++) {

        int a_row = blockIdx.x + (row_block * gridDim.x);
        for (int b_row = 0; b_row < B_TRANS_ROWS; b_row++) {

            int total = 0;
            for (int thread_i = 0; thread_i * blockDim.x < A_COLS; thread_i++) {

                int thread_col = threadIdx.x + (thread_i * blockDim.x);
                // Need to check because 240 not even multiple of 128
                if (thread_col >= A_COLS)
                    smem[threadIdx.x] = 0;
                else
                    smem[threadIdx.x] = mat_a[a_row * A_COLS + thread_col] * mat_b[b_row * B_TRANS_COLS + thread_col];
                __syncthreads();

                //Parallel reduction
                for (int i = blockDim.x / 2; i > 0; i /= 2) {
                    if (threadIdx.x < i) {
                        int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
                        smem[threadIdx.x] = temp;
                    }
                    __syncthreads();
                }
                if (threadIdx.x == 0) {
                    total += smem[threadIdx.x];
                }
            }
            if (threadIdx.x == 0) {
                res[a_row * C_COLS + b_row] = total;
            }
        }
    }
}

void basic_multiplication() {
    int *a = (int *) malloc(sizeof(int) * A_ROWS * A_COLS);
    int *b = (int *) malloc(sizeof(int) * B_ROWS * B_COLS);
    int *c = (int *) malloc(sizeof(int) * C_ROWS * C_COLS);

    srand(time(NULL));
    // Initialize matrix a
    for (int i = 0; i < A_ROWS; i++) {
        for (int j = 0; j < A_COLS; j++) {
            int el = rand() % 10;
            a[i * A_COLS + j] = el;
        }
    }

    // Initialize matrix b
    for (int i = 0; i < B_ROWS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            int el = rand() % 5;
            b[i * B_COLS + j] = el;
        }
    }

    int *a_d, *b_d, *c_d;
    HANDLE_ERR(cudaMalloc((void **) &a_d, sizeof (int) * A_ROWS * A_COLS));
    HANDLE_ERR(cudaMalloc((void **) &b_d, sizeof (int) * B_ROWS * B_COLS));
    HANDLE_ERR(cudaMalloc((void **) &c_d, sizeof (int) * C_ROWS * C_COLS));

    HANDLE_ERR(cudaMemcpy (a_d, a, sizeof (int) * A_ROWS * A_COLS, cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy (b_d, b, sizeof (int) * B_ROWS * B_COLS, cudaMemcpyHostToDevice));

    double starttime = wtime();
    mat_mult_kernel <<< 128, 128 >>> (a_d, b_d, c_d, A_ROWS, A_COLS, B_COLS);
    cudaDeviceSynchronize();
    double algotime = wtime() - starttime;
    cout << "Base multiplication: " << algotime << endl;

    HANDLE_ERR(cudaMemcpy (c, c_d, sizeof (int) * C_ROWS * C_COLS, cudaMemcpyDeviceToHost));

    //Make sure parallel work is equal to sequential work (for testing)
    int *test_res = (int *) malloc(sizeof(int) * C_ROWS * C_COLS);
    mat_mult(a, b, test_res, A_ROWS, A_COLS, B_COLS);

    for (int i = 0; i < C_ROWS; i++) {
        for (int j = 0; j < C_COLS; j++){
            int idx = i * C_COLS + j;
            if (c[idx] != test_res[idx]) {
                cout << "Not Equal at idx: " << i << ", " << j 
                    << " Parallel work " << c[idx] << ", Sequential Work: " << test_res[idx] << endl;
            }
            assert(c[idx] == test_res[idx]);
        }
    }
}

void padded_multiplication() {
    int padding = 16;
    int A_COLS_PADDED = A_COLS + padding;
    int *a = (int *) malloc(sizeof(int) * A_ROWS * A_COLS_PADDED);
    int *b = (int *) malloc(sizeof(int) * B_ROWS * B_COLS);
    int *c = (int *) malloc(sizeof(int) * C_ROWS * C_COLS);

    srand(time(NULL));
    // Initialize matrix
    for (int i = 0; i < A_ROWS; i++) {
        for (int j = 0; j < A_COLS; j++) {
            int el = rand() % 10;
            a[i * A_COLS_PADDED + j] = el;
        }
    }

    // Initialize vector
    for (int i = 0; i < B_ROWS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            int el = rand() % 5;
            b[i * B_COLS + j] = el;
        }
    }

    int *a_d, *b_d, *c_d;
    HANDLE_ERR(cudaMalloc((void **) &a_d, sizeof (int) * A_ROWS * A_COLS_PADDED));
    HANDLE_ERR(cudaMalloc((void **) &b_d, sizeof (int) * B_ROWS * B_COLS));
    HANDLE_ERR(cudaMalloc((void **) &c_d, sizeof (int) * C_ROWS * C_COLS));

    HANDLE_ERR(cudaMemcpy (a_d, a, sizeof (int) * A_ROWS * A_COLS_PADDED, cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy (b_d, b, sizeof (int) * B_ROWS * B_COLS, cudaMemcpyHostToDevice));

    double starttime = wtime();
    mat_mult_kernel <<< 128, 128 >>> (a_d, b_d, c_d, A_ROWS, A_COLS, B_COLS, padding);
    cudaDeviceSynchronize();
    double algotime = wtime() - starttime;
    cout << "Padded multiplication time: " << algotime << endl;

    HANDLE_ERR(cudaMemcpy (c, c_d, sizeof (int) * C_ROWS * C_COLS, cudaMemcpyDeviceToHost));

    //Make sure parallel work is equal to sequential work (for testing)
    int *test_res = (int *) malloc(sizeof(int) * C_ROWS * C_COLS);
    mat_mult(a, b, test_res, A_ROWS, A_COLS, B_COLS, padding);

    for (int i = 0; i < C_ROWS; i++) {
        for (int j = 0; j < C_COLS; j++){
            int idx = i * C_COLS + j;
            if (c[idx] != test_res[idx]) {
                cout << "Not Equal at idx: " << i << ", " << j 
                    << " Parallel work " << c[idx] << ", Sequential Work: " << test_res[idx] << endl;
            }
            assert(c[idx] == test_res[idx]);
        }
    }
}

void transpose_multiplication() {
    int *a = (int *) malloc(sizeof(int) * A_ROWS * A_COLS);
    int *b = (int *) malloc(sizeof(int) * B_ROWS * B_COLS);
    int *trans = (int *) malloc(sizeof(int) * B_ROWS * B_COLS);
    int *c = (int *) malloc(sizeof(int) * C_ROWS * C_COLS);

    srand(time(NULL));
    // Initialize matrix a
    for (int i = 0; i < A_ROWS; i++) {
        for (int j = 0; j < A_COLS; j++) {
            int el = rand() % 10;
            a[i * A_COLS + j] = el;
        }
    }

    // Initialize matrix b
    for (int i = 0; i < B_ROWS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            int el = rand() % 5;
            b[i * B_COLS + j] = el;
        }
    }

    // Transpose matrix b
    for (int i = 0; i < B_ROWS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            trans[j * B_ROWS + i] = b[i * B_COLS + j];
        }
    }

    int *a_d, *b_d, *c_d;
    HANDLE_ERR(cudaMalloc((void **) &a_d, sizeof (int) * A_ROWS * A_COLS));
    HANDLE_ERR(cudaMalloc((void **) &b_d, sizeof (int) * B_ROWS * B_COLS));
    HANDLE_ERR(cudaMalloc((void **) &c_d, sizeof (int) * C_ROWS * C_COLS));

    HANDLE_ERR(cudaMemcpy (a_d, a, sizeof (int) * A_ROWS * A_COLS, cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy (b_d, trans, sizeof (int) * B_ROWS * B_COLS, cudaMemcpyHostToDevice));

    double starttime = wtime();
    mat_mult_transposed_kernel <<< 128, 128 >>> (a_d, b_d, c_d);
    cudaDeviceSynchronize();
    double algotime = wtime() - starttime;
    cout << "Transposed multiplication time: " << algotime << endl;

    HANDLE_ERR(cudaMemcpy (c, c_d, sizeof (int) * C_ROWS * C_COLS, cudaMemcpyDeviceToHost));

    //Make sure parallel work is equal to sequential work (for testing)
    int *test_res = (int *) malloc(sizeof(int) * C_ROWS * C_COLS);
    mat_mult(a, b, test_res, A_ROWS, A_COLS, B_COLS);

    for (int i = 0; i < C_ROWS; i++) {
        for (int j = 0; j < C_COLS; j++){
            int idx = i * C_COLS + j;
            if (c[idx] != test_res[idx]) {
                cout << "Not Equal at idx: " << i << ", " << j 
                    << " Parallel work " << c[idx] << ", Sequential Work: " << test_res[idx] << endl;
            }
            assert(c[idx] == test_res[idx]);
        }
    }
}

int main (int args, char **argv) {
    basic_multiplication();
    padded_multiplication();
    transpose_multiplication();
}
