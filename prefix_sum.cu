/**
  * Jacob Sword
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

int main(int argc, char **argv) {
    int arr_size = 8;
    int *arr = new int[arr_size];

    // Fill array
    for (int i = 0; i < arr_len; i++) {
        arr[i] = rand() % 11;
    }

    int *arr_d;
    HANDLE_ERR(cudaMalloc((void **) &arr_d, sizeof (int) * arr_size));
    HANDLE_ERR(cudaMemcpy (arr_d, arr, sizeof (int) * arr_size, cudaMemcpyHostToDevice));

    // CPU
    double starttime = wtime();
    cpu_sum = prefix_sum_cpu(arr, arr_size);
    double endtime = wtime();
    double cpu_time = endtime - starttime;
    cout << "Time for cpu summation: " << cpu_time << endl;

    starttime = wtime();
    kernel <<< 128, 128 >>> (arr_d, arr_size);
    cudaDeviceSynchronize();
    endtime = wtime();
    double gpu_time = endtime - starttime;
    cout << "Time for GPU" << ": " << gpu_time << endl;
        
    // Check sum
    assert(sum == cpu_sum);

}
