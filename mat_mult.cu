/**
  Jacob Sword
  Parallelized multiplication of two randomized matrices given dimensions of each
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

void mat_mult(int *mat_a, int *mat_b, int *result, int a_rows, int a_cols, int b_cols)
{
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            int temp_res = 0;
            for (int k = 0; k < a_cols; k++) {
                temp_res += mat_a[i * a_cols + k] * mat_b[k * b_cols + j];
            }
            result[i * b_cols + j] = temp_res;
        }
    }
}

__global__ void mat_mult_kernel(int *mat_a, int *mat_b, int *result, int a_rows, int a_cols, int b_cols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < a_rows) {
        for (int j = 0; j < b_cols; j++) {
            int temp_res = 0;
            for (int k = 0; k < a_cols; k++) {
                temp_res  += mat_a[tid * a_cols + k] * mat_b[k * b_cols + j];
            }
            result[tid * b_cols + j] = temp_res;
        }
        tid += blockDim.x * gridDim.x;
    }
}

int main (int argc, char **argv) {
    if (argc < 5) {
        cout << "Need a_row a_col b_row b_col" << endl;
        return 1;
    }
    int a_rows = atoi(argv[1]);
    int a_cols = atoi(argv[2]);

    int b_rows = atoi(argv[3]);
    int b_cols = atoi(argv[4]);

    if (a_cols != b_rows) {
        cout << "Columns of matrix a must equal rows of matrix b" << endl;
        return 1;
    }

    int a_dims = a_rows * a_cols;
    int b_dims = b_rows * b_cols;
    int c_dims = a_rows * b_cols;

    int *a = (int *) malloc(sizeof(int) * a_dims);
    int *b = (int *) malloc(sizeof(int) * b_dims);
    int *c = (int *) malloc(sizeof(int) * c_dims);

    srand(time(NULL));

    // Initialize matrix a
    cout << "Matrix a:" << endl;
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < a_cols; j++) {
            int el = rand() % 10;
            a[i * a_cols + j] = el;
            cout << el << ", ";
        }
        cout << endl;
    }

    // Initialize matrix b
    cout << "Matrix b:" << endl;
    for (int i = 0; i < b_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            int el = rand() % 5;
            b[i * b_cols + j] = el;
            cout << el << ", ";
        }
        cout << endl;
    }

    int *a_d, *b_d, *c_d;
    HANDLE_ERR(cudaMalloc((void **) &a_d, sizeof (int) * a_dims));
    HANDLE_ERR(cudaMalloc((void **) &b_d, sizeof (int) * b_dims));
    HANDLE_ERR(cudaMalloc((void **) &c_d, sizeof (int) * c_dims));

    HANDLE_ERR(cudaMemcpy (a_d, a, sizeof (int) * a_dims, cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy (b_d, b, sizeof (int) * b_dims, cudaMemcpyHostToDevice));
    mat_mult_kernel <<< 256, 256 >>> (a_d, b_d, c_d, a_rows, a_cols, b_cols);

    HANDLE_ERR(cudaMemcpy (c, c_d, sizeof (int) * c_dims, cudaMemcpyDeviceToHost));

    int *test_res = (int *) malloc(sizeof(int) * c_dims);

    // Make sure parallel and sequential implementation same (for testing)
    mat_mult(a, b, test_res, a_rows, a_cols, b_cols);
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++){
            assert(c[i * b_cols + j] == test_res[i * b_cols + j]);
        }
    }

    cout << "Result matrix:" << endl;
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            cout << c[i * b_cols + j] << ", ";
        }
        cout << endl;
    }
}
