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

// Parllel implementation of matrix x vector
__global__ void mat_mult_kernel(int *a, int *b, int *c, int mat_rows, int mat_cols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < mat_rows) {
        int res = 0;
        for (int i = 0; i < mat_cols; i++) {
            res += a[tid * mat_cols + i] * b[i];
        }
        c[tid] = res;
        tid += blockDim.x * gridDim.x;
    }
}

int main (int args, char **argv) {
    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);

    int *a = (int *) malloc(sizeof(int) * num_rows * num_cols);
    int *b = (int *) malloc(sizeof(int) * num_cols);
    int *c = new int[num_rows];

    srand(time(NULL));
    // Initialize matrix
    cout << "Matrix (a):" << endl;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            int el = rand() % 10;
            a[i * num_cols + j] = el;
            cout << el << ", ";
        }
        cout << endl;
    }

    // Initialize vector
    cout << "Vector (b):" << endl;
    for (int i = 0; i < num_cols; i++) {
        int el = rand() % 5;
        b[i] = el;
        cout << el << endl;
    }

    int *a_d, *b_d, *c_d;
    HANDLE_ERR(cudaMalloc((void **) &a_d, sizeof (int) * num_rows * num_cols));
    HANDLE_ERR(cudaMalloc((void **) &b_d, sizeof (int) * num_cols));
    HANDLE_ERR(cudaMalloc((void **) &c_d, sizeof (int) * num_rows));

    HANDLE_ERR(cudaMemcpy (a_d, a, sizeof (int) * num_rows * num_cols, cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy (b_d, b, sizeof (int) * num_cols, cudaMemcpyHostToDevice));
    mat_mult_kernel <<< 256, 256 >>> (a_d, b_d, c_d, num_rows, num_cols);

    HANDLE_ERR(cudaMemcpy (c, c_d, sizeof (int) * num_rows, cudaMemcpyDeviceToHost));

    //Make sure parallel work is equal to sequential work (for testing)
    int *test_res = new int[num_rows];
    mat_vec_mult(a, b, test_res, num_rows, num_cols);
    for (int i = 0; i < num_rows; i++) {
        assert(c[i] == test_res[i]);
    }

    cout << "Result (c):" << endl;
    for (int i = 0; i < num_rows; i++) {
        cout << c[i] << endl;
    }
}
