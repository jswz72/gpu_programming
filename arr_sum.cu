/**
  Jacob Sword
**/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <time.h>
#include "./error_handler.h"
#include "./wtime.h"

using std::cout;
using std::endl;

int sum_cpu(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++)
        sum += arr[i];
    return sum;
}

// Every thread atomoically adds its integers to global sum
__global__ void sum_naive_kernel(int *arr, int size, int *sum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        atomicAdd(sum, arr[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

// Every threads gets local sum, smemm used to get block sums.
// Block sums atomically summed to total sum
__global__ void sum_improved_kernel(int *arr, int size, int *sum) {
    int num_threads = blockDim.x * gridDim.x;
    int division = size / num_threads;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start_idx = tid * division;
    int end_idx = (tid + 1) * division;
    if (tid == 0)
        *sum = 0;


    // Each thread finds local sum of its assigned area
    int my_sum = 0;
    __shared__ int smem[128];
    for (int i = start_idx; i < end_idx; i++) {
        // Off by one thing i think
        if (i == 15999999)
            printf("asdfasdf\n");

        my_sum += arr[i];
    }
    atomicAdd(sum, my_sum);
    //smem[threadIdx.x] = my_sum;
    /*
    // Barrier then use parallel reduction to get block sum
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
            smem[threadIdx.x] = temp;
        }
        __syncthreads();
    }
    // Block sum atomically added to global sum
    if (threadIdx.x == 0) {
        atomicAdd(sum, smem[0]);
    }*/
}

int main (int argc, char **argv) {
    int arr_size = 16 * pow(10, 6);
    int *arr = (int *) malloc(sizeof(int) * arr_size);

    srand(time(NULL));

    // Initialize arr
    for (int i = 0; i < arr_size; i++)
        arr[i] = 1 + (rand() % 4);

    int *arr_d;
    HANDLE_ERR(cudaMalloc((void **) &arr_d, sizeof (int) * arr_size));
    HANDLE_ERR(cudaMemcpy (arr_d, arr, sizeof (int) * arr_size, cudaMemcpyHostToDevice));

    int *sum_d;
    HANDLE_ERR(cudaMalloc((void **) &sum_d, sizeof (int)));

    // NAIVE GPU
    double starttime = wtime();
    sum_naive_kernel <<< 128, 128 >>> (arr_d, arr_size, sum_d);
    cudaDeviceSynchronize();
    double endtime = wtime();
    double naive_gpu_time = endtime - starttime;
    cout << "Time for naive GPU summation: " << naive_gpu_time << endl;

    int sum;
    HANDLE_ERR(cudaMemcpy (&sum, sum_d, sizeof (int), cudaMemcpyDeviceToHost));

    // CPU
    starttime = wtime();
    int cpu_sum = sum_cpu(arr, arr_size);
    endtime = wtime();
    double cpu_time = endtime - starttime;
    cout << "Time for cpu summation: " << cpu_time << endl;
    cout << "Naive GPU " << (int)(cpu_time / naive_gpu_time) 
        << " times faster than CPU" << endl;
    assert(sum == cpu_sum);

    // IMPROVED GPU
    starttime = wtime();
    sum_improved_kernel <<< 128, 128 >>> (arr_d, arr_size, sum_d);
    cudaDeviceSynchronize();
    endtime = wtime();
    double improved_gpu_time = endtime - starttime;
    cout << "Time for improved GPU summation: " << improved_gpu_time << endl;
    cout << "Improved GPU is " << (int)(cpu_time / improved_gpu_time) 
        << " times faster than CPU" << endl;
    cout << "Improved GPU is " << (int)(naive_gpu_time / improved_gpu_time) 
        << " times faster than Naive GPU" << endl;
    cout << "Naive GPU is " << (int)(improved_gpu_time / naive_gpu_time) 
        << " times faster than Naive GPU" << endl;

    sum = 0;
    HANDLE_ERR(cudaMemcpy (&sum, sum_d, sizeof (int), cudaMemcpyDeviceToHost));

    if (sum != cpu_sum)
        cout << "S: " << sum << " CPU S: " << cpu_sum << endl;
    assert(sum == cpu_sum);
}
