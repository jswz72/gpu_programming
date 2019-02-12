/**
  Jacob Sword
  Parallelized multiplication of matrix and vector of random values given matrix dimensions
**/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include "./error_handler.h"
#include "./wtime.h"

using std::cout;
using std::endl;

__global__ void vec_add_kernel_first(int *a, int *b, int *c, int len) {
    int thread_count = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int step = (len / thread_count) + (len % thread_count != 0);
    int idx_begin = tid * step;
    int idx_end = (tid + 1) * step;
    if (idx_end >= len)
        idx_end = len - 1;
    while (idx_begin < idx_end) {
        c[idx_begin] = a[idx_begin] + b[idx_begin];
        idx_begin++;
    }
}

__global__ void vec_add_kernel_old(int *a, int *b, int *c, int len) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < len) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main (int args, char **argv) {
    int len = 1 << 20;

    int *a = new int [len];
    int *b = new int [len];
    int *c = new int [len];

    srand(time(NULL));
    // Initialize vecs
    for (int i = 0; i < 1 << 20; i++) {
        int el = rand() % 100;
        a[i] = el;
    }
    for (int i = 0; i < 1 << 20; i++) {
        int el = rand() % 50;
        b[i] = el;
    }

    int *a_d, *b_d, *c_d;
    HANDLE_ERR(cudaMalloc((void **) &a_d, sizeof (int) * len));
    HANDLE_ERR(cudaMalloc((void **) &b_d, sizeof (int) * len));
    HANDLE_ERR(cudaMalloc((void **) &c_d, sizeof (int) * len));

    HANDLE_ERR(cudaMemcpy (a_d, a, sizeof (int) * len, cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy (b_d, b, sizeof (int) * len, cudaMemcpyHostToDevice));

    double start = wtime();
    vec_add_kernel_first <<< 128, 128 >>> (a_d, b_d, c_d, len);
    cudaDeviceSynchronize();
    printf("new way: %lf\n", wtime() - start);

    for (int i = 0; i < 1 << 20; i++) {
        int el = rand() % 100;
        a[i] = el;
    }
    for (int i = 0; i < 1 << 20; i++) {
        int el = rand() % 50;
        b[i] = el;
    }

    start = wtime();
    vec_add_kernel_old <<< 128, 128 >>> (a_d, b_d, c_d, len);
    cudaDeviceSynchronize();
    printf("old way: %lf\n", wtime() - start);

    HANDLE_ERR(cudaMemcpy (c, c_d, sizeof (int) * len, cudaMemcpyDeviceToHost));

}
