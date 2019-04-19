#include <limits>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iostream>
#include "graph.h"
#include "error_handler.h"
#include "wtime.h"

using std::cout;
using std::endl;

#define BLOCK_SIZE 16;
#define NUM_ASYNCHRONOUS_ITERATIONS 20  // Number of async loop iterations before attempting to read results back

// Check whether all verticies done
bool all_vertices_done(bool *finished_verts, int num_vtx) {

    for (int i = 0; i < num_vtx; i++)
        if (finished_verts[i] == true)
            return false;

    return true;
}

// division ceiling
int div_ceil(int numer, int denom) {
    return (numer % denom != 0) ? (numer / denom + 1) : (numer / denom);
}

__global__ void init_sssp_arrs(bool * __restrict__ d_finished_verts, 
        int* __restrict__ d_dists, 
        int* __restrict__ d_update_dists, const int source, 
        const int num_vtx) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vtx) {
        if (source == tid) {
            d_finished_verts[tid] = true;
            d_dists[tid] = 0;
            d_update_dists[tid] = 0;
        }
        else {
            d_finished_verts[tid] = false;
            d_dists[tid] = INT_MAX;
            d_update_dists[tid] = INT_MAX;
        }
    }
}

// First portion GPU work for djikstra's
__global__  void Kernel1(const int * __restrict__ beg_pos, 
        const int* __restrict__ adj_list, const int * __restrict__ weights,
        bool * __restrict__ finished_verts, int* __restrict__ dists,
        int * __restrict__ update_dists, const int num_vtx) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < num_vtx) {
        if (finished_verts[tid] == true) {
            finished_verts[tid] = false;
            for (int edge = beg_pos[tid]; edge < beg_pos[tid + 1]; edge++) {
                int other = adj_list[edge];
                atomicMin(&update_dists[other], 
                        dists[tid] + weights[edge]);
            }
        }
    }
}

// Second portion of GPU work for djikstra's
__global__  void Kernel2(const int * __restrict__ beg_pos, 
        const int * __restrict__ adj_list, 
        const int* __restrict__ weights, bool * __restrict__ finished_verts,
        int* __restrict__ dists, 
        int* __restrict__ update_dists, const int num_vtx) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vtx) {
        if (dists[tid] > update_dists[tid]) {
            dists[tid] = update_dists[tid];
            finished_verts[tid] = true; }
        update_dists[tid] = dists[tid];
    }
}

// Main function handling djikstra's kernels
void dijkstraGPU(int *beg_pos, int *adj_list, int *weights, const int source, 
        int * __restrict__ dists, int num_vtx, int num_edge) {
    double s1 = wtime();
    int *d_beg_pos;
    HANDLE_ERR(cudaMalloc(&d_beg_pos, sizeof(int) * num_vtx));
    int *d_adj_list;
    HANDLE_ERR(cudaMalloc(&d_adj_list, sizeof(int) * num_edge));
    int *d_weights;
    HANDLE_ERR(cudaMalloc(&d_weights, sizeof(int) * num_edge));

    HANDLE_ERR(cudaMemcpy(d_beg_pos, beg_pos, sizeof(int) * num_vtx,
                cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy(d_adj_list, adj_list, sizeof(int) * num_edge,
                cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy(d_weights, weights, sizeof(int) * num_edge,
                cudaMemcpyHostToDevice));

    // Mask array
    bool *d_finished_verts;
    HANDLE_ERR(cudaMalloc(&d_finished_verts, sizeof(bool) * num_vtx));
    // Cost array
    int *d_dists;
    HANDLE_ERR(cudaMalloc(&d_dists, sizeof(int) * num_vtx));
    // Updating cost array
    int *d_update_dists;
    HANDLE_ERR(cudaMalloc(&d_update_dists, sizeof(int) * num_vtx));

    bool *finished_vtxs = (bool *)malloc(sizeof(bool) * num_vtx);

    double e1 = wtime();
    cout << "Time to init n copy: " << e1 - s1 << endl;
    double start = wtime();
    // Mask to 0's, cost and updating arrays to inf
    init_sssp_arrs <<<div_ceil(num_vtx, 16), 16 >>>
        (d_finished_verts, d_dists, d_update_dists, source, num_vtx);
    HANDLE_ERR(cudaDeviceSynchronize());

    HANDLE_ERR(cudaMemcpy(finished_vtxs, d_finished_verts, 
                sizeof(bool) * num_vtx, cudaMemcpyDeviceToHost));

    while (!all_vertices_done(finished_vtxs, num_vtx)) {
        // --- In order to improve performance, we run some number of iterations without reading the results.  This might result
        //     in running more iterations than necessary at times, but it will in most cases be faster because we are doing less
        //     stalling of the GPU waiting for results.
        for (int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; 
                asyncIter++) {
            Kernel1 <<<div_ceil(num_vtx, 16), 16 >>>(d_beg_pos, d_adj_list, 
                    d_weights, d_finished_verts, d_dists, d_update_dists, num_vtx);
            HANDLE_ERR(cudaDeviceSynchronize());
            Kernel2 <<<div_ceil(num_vtx, 16), 16 >>>(d_beg_pos, d_adj_list, 
                d_weights, d_finished_verts, d_dists,
                d_update_dists, num_vtx);
            HANDLE_ERR(cudaDeviceSynchronize());
        }
        HANDLE_ERR(cudaMemcpy(finished_vtxs, d_finished_verts,
                sizeof(bool) * num_vtx, cudaMemcpyDeviceToHost));
    }
    double end = wtime();
    double s2 = wtime();
    cout << "Inner GPU time: " << end - start << endl;

    HANDLE_ERR(cudaMemcpy(dists, d_dists, sizeof(int) * num_vtx, cudaMemcpyDeviceToHost));

    free(finished_vtxs);

    HANDLE_ERR(cudaFree(d_beg_pos));
    HANDLE_ERR(cudaFree(d_adj_list));
    HANDLE_ERR(cudaFree(d_weights));
    HANDLE_ERR(cudaFree(d_finished_verts));
    HANDLE_ERR(cudaFree(d_dists));
    HANDLE_ERR(cudaFree(d_update_dists));
    double e2 = wtime();
    cout << "Copy out and free time: " << e2 - s2 << endl;
}

// Get index of vertex not included in path with min dist
int min_dist(int *dists, bool *finished_verts, 
        const int source, const int N) {
    int minIndex = source;
    int min = INT_MAX;
    for (int v = 0; v < N; v++)
        if (finished_verts[v] == false && dists[v] <= min) {
            min = dists[v];
            minIndex = v;
        }
    return minIndex;
}

void dijkstraCPU(int *beg_pos, int *adj_list, int *weights, int *dists, int source, const int N) {

    // --- finished_vtxs[i] is true if vertex i is included in the shortest path tree
    //     or the shortest distance from the source node to i is finalized
    bool *finished_vtxs = (bool *)malloc(N * sizeof(bool));

    for (int i = 0; i < N; i++) {
        dists[i] = INT_MAX;
        finished_vtxs[i] = false;
    }

    dists[source] = 0;

    for (int i = 0; i < N - 1; i++) {
        int cur = min_dist(dists, finished_vtxs, source, N);
        finished_vtxs[cur] = true;
        // Relaxation?
        for (int v = 0; v < N; v++) {
            // Update if not finished, edge exits, and cost lower than cur cost
            bool found = false;
            int idx = 0;
            for (int i = beg_pos[cur]; i < beg_pos[cur + 1]; 
                    i++) {
                if (v == adj_list[i]) {
                    found = true;
                    idx = i;
                }
            }
            bool update = !finished_vtxs[v] && found &&
                dists[cur] != INT_MAX &&
                dists[cur] + weights[idx] <
                    dists[v];
            if (update)
                dists[v] = dists[cur]
                    + weights[idx];
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { 
        cout << "Enter source idx" << endl;
        return -1;
    }
    int source_vertex = atoi(argv[1]);
    graph<long, long, double, long, long, double> *csr = 
        new graph<long, long, double, long, long, double>
        ("../../get_edges/edge-list.txt_beg_pos.bin",
         "../../get_edges/edge-list.txt_csr.bin",
         "../../get_edges/edge-list.txt_weight.bin");

    int *shortest_dist_cpu = (int *)malloc(csr->vert_count * sizeof(int));

    int *beg_pos = (int *)malloc(csr->vert_count * sizeof(int));
    int *adj_list = (int *)malloc(csr->edge_count * sizeof(int));
    int *weight = (int *)malloc(csr->edge_count * sizeof(int));

    for (int i = 0; i < csr->vert_count; i++) beg_pos[i] = (int) csr->beg_pos[i];
    
    for (int i = 0; i < csr->edge_count; i++) adj_list[i] = (int) csr->csr[i];
    for (int i = 0; i < csr->edge_count; i++) weight[i] = (int) (csr->weight[i] * 1000);

        
    int *shortest_dist_gpu = (int *) malloc(csr->vert_count * sizeof(int));
    double gpu_start = wtime();
    dijkstraGPU(beg_pos, adj_list, weight, source_vertex, shortest_dist_gpu, csr->vert_count, csr->edge_count);
    double gpu_end = wtime();
    cout << "GPU completed in " << gpu_end - gpu_start << endl;

    double cpu_start = wtime();
    dijkstraCPU(beg_pos, adj_list, weight, shortest_dist_cpu, source_vertex, csr->vert_count);
    double cpu_end = wtime();
    cout << "CPU completed in " << cpu_end - cpu_start << endl;


    for (int i = 0; i < csr->vert_count; i++) {
        if (shortest_dist_gpu[i] != shortest_dist_cpu[i]) {
            cout << "Index " << i << " fail: " << "CPU " << shortest_dist_cpu[i] 
            << " GPU " << shortest_dist_gpu[i] << endl;
        }
        assert(shortest_dist_gpu[i] == shortest_dist_cpu[i]);
    }
}
