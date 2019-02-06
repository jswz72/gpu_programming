/**
  Jacob Sword
  Parallelized multiplication of two matrices given dimensions of each
**/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include<cstdlib>
#include<time.h>

using std::cout;
using std::endl;

void mat_mult(int *mat, int *vec, int *res, int num_rows, int num_cols)
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

__global__ void mat_mult_kernel(int *matrix_a, int *matrix_b, int *result, int a_cols, int a_rows, int b_cols, int b_rows) {
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
    cudaMalloc((void **) &a_d, sizeof (int) * a_dims);
    cudaMalloc((void **) &b_d, sizeof (int) * b_dims);
    cudaMalloc((void **) &c_d, sizeof (int) * c_dims);

    cudaMemcpy (a_d, a, sizeof (int) * a_dims, cudaMemcpyHostToDevice);
    cudaMemcpy (b_d, b, sizeof (int) * b_dims, cudaMemcpyHostToDevice);
    mat_mult_kernel <<< 256, 256 >>> (a_d, b_d, c_d, a_rows, a_cols, b_rows, b_cols);

    cudaMemcpy (c, c_d, sizeof (int) * c_dims, cudaMemcpyDeviceToHost);

    int *test_res = (int *) malloc(sizeof(int) * c_dims);

    mat_vec_mult(a, b, test_res, a_rows, a_cols, b_rows, b_cols);
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_rows; j++){
            assert(c[i * a_rows + j] == test_res[i * a_rows + j]);
        }
    }

    cout << "Result (c):" << endl;
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_rows; j++){
            cout << (c[i * a_rows + j] << ", ";
        }
        cout << endl;
    }
}
