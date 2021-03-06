/**
  * Jacob Sword
  * Comparison of GPU summation methods
**/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <time.h>
#include "./error_handler.h"
#include "./wtime.h"

using std::cout;
using std::endl;

// Linear CPU summation for baseline
int sum_cpu(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++)
        sum += arr[i];
    return sum;
}

// Every thread atomically adds its integers to global sum
__global__ void sum_naive_kernel(int *arr, int size, int *sum) {
    int num_threads = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        atomicAdd(sum, arr[tid]);
        tid += num_threads;
    }
}

// Every thread gets local sum, shared mem used to get block sums.
// Block sums atomically summed to total sum
__global__ void sum_improved_atomic_kernel(int *arr, int size, int *sum) {
    int num_threads = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread finds local sum of its assigned area
    int my_sum = 0;
    __shared__ int smem[128];

    while (tid < size) {
        my_sum += arr[tid];
        tid += num_threads;
    }

    // Load local sum into shared mem
    smem[threadIdx.x] = my_sum;

    // Barrier then use parallel reduction to get block sum
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (threadIdx.x < i)
            smem[threadIdx.x] += smem[threadIdx.x + i];
        __syncthreads();
    }
    // Block sum atomically added to global sum
    if (threadIdx.x == 0) {
        atomicAdd(sum, smem[0]);
    }
}

// Every thread gets local sum, smemm used to get block sums.
__global__ void block_sum_kernel(int *arr, int size, int *block_sums) {
    int num_threads = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread finds local sum of its assigned area
    int my_sum = 0;
    __shared__ int smem[128];
    while (tid < size) {
        my_sum += arr[tid];
        tid += num_threads;
    }

    smem[threadIdx.x] = my_sum;

    // Barrier then use parallel reduction to get block sum
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
            smem[threadIdx.x] = temp;
        }
        __syncthreads();
    }
    // Block sum added to global arr
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = smem[0];
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "Enter power for arr size " << endl;
        return 1;
    }
    int arr_size = 16 * pow(10, atoi(argv[1]));
    cout << "Using array size of " << arr_size << endl;
    int *arr = (int *) malloc(sizeof(int) * arr_size);

    srand(time(NULL));

    // Initialize arr
    for (int i = 0; i < arr_size; i++)
        arr[i] = 1 + (rand() % 5);

    int *arr_d;
    HANDLE_ERR(cudaMalloc((void **) &arr_d, sizeof (int) * arr_size));
    HANDLE_ERR(cudaMemcpy (arr_d, arr, sizeof (int) * arr_size, cudaMemcpyHostToDevice));

    int *sum_d;
    HANDLE_ERR(cudaMalloc((void **) &sum_d, sizeof (int)));

    // CPU
    double starttime = wtime();
    int cpu_sum = sum_cpu(arr, arr_size);
    double endtime = wtime();
    double cpu_time = endtime - starttime;
    cout << "Time for cpu summation: " << cpu_time << endl;


    // NAIVE GPU
    std::string naive = "Naive GPU";
    starttime = wtime();
    sum_naive_kernel <<< 128, 128 >>> (arr_d, arr_size, sum_d);
    cudaDeviceSynchronize();
    endtime = wtime();
    double naive_gpu_time = endtime - starttime;
    cout << "Time for " << naive << ": " << naive_gpu_time << endl;

    int sum;
    HANDLE_ERR(cudaMemcpy (&sum, sum_d, sizeof (int), cudaMemcpyDeviceToHost));
        
    // Check sum
    assert(sum == cpu_sum);

    // IMPROVED GPU, using atomic add
    std::string improved_1 = "Improved GPU using atomic add";
    // Reset device sum
    HANDLE_ERR(cudaMemset(sum_d, 0, sizeof(int)));
    starttime = wtime();
    sum_improved_atomic_kernel <<< 128, 128 >>> (arr_d, arr_size, sum_d);
    cudaDeviceSynchronize();
    endtime = wtime();
    double improved_gpu_time = endtime - starttime;
    cout << "Time for " << improved_1 << ": " << improved_gpu_time << endl;

    // Check sum
    sum = 0;
    HANDLE_ERR(cudaMemcpy (&sum, sum_d, sizeof (int), cudaMemcpyDeviceToHost));
    assert(sum == cpu_sum);

    // IMPROVED GPU using CPU add
    std::string improved_2 = "Improved GPU using CPU add";
    // Create block sum arr
    int *block_sums_d;
    HANDLE_ERR(cudaMalloc((void **) &block_sums_d, sizeof (int) * 128));

    sum = 0;
    int *block_sums = (int *)malloc(sizeof(int) * 128);
    starttime = wtime();
    block_sum_kernel <<< 128, 128 >>> (arr_d, arr_size, block_sums_d);
    cudaDeviceSynchronize();
    endtime = wtime();
    HANDLE_ERR(cudaMemcpy (block_sums, block_sums_d, sizeof (int) * 128, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 128; i++)
        sum += block_sums[i];
    double endtime_2 = wtime();

    double improved_gpu_2_time = endtime - starttime;
    cout << "Time for " << improved_2 << " (before CPU sum): " << improved_gpu_2_time << endl;
    cout << "Time for " << improved_2 << " complete: " << endtime_2 - starttime << endl;

    // Check sum
    assert(sum == cpu_sum);
}
