#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include<cstdlib>

using std::cout;
using std::endl;

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
    int *b = (int *) malloc(sizeof(int) * num_rows * num_cols);
    int *c = new int[num_rows];

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            a[i * num_cols + j] = rand() % 10;
        }
    }

    for (int i = 0; i < num_cols; i++) {
        for (int j = 0; j < num_rows; j++) {
            b[i * num_rows + j] = rand() % 5;
        }
    }

    int *a_d, *b_d, *c_d;
    cudaMalloc((void **) &a_d, sizeof (int) * num_rows * num_cols);
    cudaMalloc((void **) &b_d, sizeof (int) * num_rows * num_cols);
    cudaMalloc((void **) &c_d, sizeof (int) * num_rows);

    cudaMemcpy (a_d, a, sizeof (int) * num_rows * num_cols, cudaMemcpyHostToDevice);
    cudaMemcpy (b_d, b, sizeof (int) * num_cols, cudaMemcpyHostToDevice);
    mat_mult_kernel <<< 256, 256 >>> (a_d, b_d, c_d, num_rows, num_cols);

    cudaMemcpy (c, c_d, sizeof (int) * num_rows, cudaMemcpyDeviceToHost);

    cout << "A (mat):" << endl;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            cout << a[i * num_cols +j] << ", ";
        }
        cout << endl;
    }

    cout << "B (mat2):" << endl;
    for (int i = 0; i < num_cols; i++) {
        for (int j = 0; j < num_rows; j++) {
            cout << a[i * num_rows +j] << ", ";
        }
        cout << endl;
    } 

    cout << "C (result):" << endl;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_rows; j++) {
            cout << a[i * num_rows +j] << ", ";
        }
        cout << endl;
    }

    int *test_res = new int[num_rows];
    for(int i = 0; i < num_rows; i ++)
    {
        for (int j = 0; j < num_cols; j ++)
        {
            int temp_res = 0;
            for (int k = 0; k < num_rows; k++) {
                temp_res += a[i * num_cols + k] * b[k * num_rows + j];
            }
        }
        test_res[i * num_cols + j] = temp_res;
    }
    for (int i = 0; i < num_rows; i++) {
        assert(c[i] == test_res[i]);
    }
    cout << "Correct results" << endl;
}
