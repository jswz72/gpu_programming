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
bool allFinalizedVertices(bool *finalizedVertices, int numVertices) {

    for (int i = 0; i < numVertices; i++)
        if (finalizedVertices[i] == true)
            return false;

    return true;
}

//Round a / b to ceiling
int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

/*************************/
/* ARRAY INITIALIZATIONS */
/*************************/
__global__ void initializeArrays(bool * __restrict__ d_finalizedVertices, 
        int* __restrict__ d_shortestDistances, 
        int* __restrict__ d_updatingShortestDistances, const int sourceVertex, 
        const int numVertices) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numVertices) {
        if (sourceVertex == tid) {
            d_finalizedVertices[tid] = true;
            d_shortestDistances[tid] = 0;
            d_updatingShortestDistances[tid] = 0;
        }
        else {
            d_finalizedVertices[tid] = false;
            d_shortestDistances[tid] = INT_MAX;
            d_updatingShortestDistances[tid] = INT_MAX;
        }
    }
}

// First portion GPU work for djikstra's
__global__  void Kernel1(const int * __restrict__ vertexArray, 
        const int* __restrict__ edgeArray, const int * __restrict__ weightArray,
        bool * __restrict__ finalizedVertices, int* __restrict__ shortestDistances,
        int * __restrict__ updatingShortestDistances, const int numVertices, 
        const int numEdges) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < numVertices) {
        if (finalizedVertices[tid] == true) {
            finalizedVertices[tid] = false;
            int edgeStart = vertexArray[tid];
            int edgeEnd = vertexArray[tid + 1];
            for (int edge = edgeStart; edge < edgeEnd; edge++) {
                int nid = edgeArray[edge];
                atomicMin(&updatingShortestDistances[nid], 
                        shortestDistances[tid] + weightArray[edge]);
            }
        }
    }
}

// Second portion of GPU work for djikstra's
__global__  void Kernel2(const int * __restrict__ vertexArray, 
        const int * __restrict__ edgeArray, 
        const int* __restrict__ weightArray, bool * __restrict__ finalizedVertices,
        int* __restrict__ shortestDistances, 
        int* __restrict__ updatingShortestDistances, const int numVertices) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numVertices) {
        if (shortestDistances[tid] > updatingShortestDistances[tid]) {
            shortestDistances[tid] = updatingShortestDistances[tid];
            finalizedVertices[tid] = true; }
        updatingShortestDistances[tid] = shortestDistances[tid];
    }
}

// Main function handling djikstra's kernels
void dijkstraGPU(int *beg_pos, int *adj_list, int *weights, const int sourceVertex, int * __restrict__ h_shortestDistances, int num_vtx, int num_edge) {

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
    bool *d_finalizedVertices;
    HANDLE_ERR(cudaMalloc(&d_finalizedVertices, sizeof(bool) * num_vtx));
    // Cost array
    int *d_shortestDistances;
    HANDLE_ERR(cudaMalloc(&d_shortestDistances, sizeof(int) * num_vtx));
    // Updating cost array
    int *d_updatingShortestDistances;
    HANDLE_ERR(cudaMalloc(&d_updatingShortestDistances, sizeof(int) * num_vtx));

    bool *h_finalizedVertices = (bool *)malloc(sizeof(bool) * num_vtx);

    // Mask to 0's, cost and updating arrays to inf
    initializeArrays 
        <<<iDivUp(num_vtx, 16), 16 >>>
        (d_finalizedVertices, d_shortestDistances, d_updatingShortestDistances, sourceVertex, num_vtx);
    HANDLE_ERR(cudaPeekAtLastError());
    HANDLE_ERR(cudaDeviceSynchronize());

    HANDLE_ERR(cudaMemcpy(h_finalizedVertices, d_finalizedVertices, 
                sizeof(bool) * num_vtx, cudaMemcpyDeviceToHost));

    while (!allFinalizedVertices(h_finalizedVertices, num_vtx)) {
        // --- In order to improve performance, we run some number of iterations without reading the results.  This might result
        //     in running more iterations than necessary at times, but it will in most cases be faster because we are doing less
        //     stalling of the GPU waiting for results.
        for (int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; 
                asyncIter++) {
            Kernel1 <<<iDivUp(num_vtx, 16), 16 >>>(d_beg_pos, d_adj_list, 
                    d_weights, d_finalizedVertices, d_shortestDistances, d_updatingShortestDistances, num_vtx, num_edge);
            HANDLE_ERR(cudaPeekAtLastError());
            HANDLE_ERR(cudaDeviceSynchronize());
            Kernel2 <<<iDivUp(num_vtx, 16), 16 >>>(d_beg_pos, d_adj_list, 
                d_weights, d_finalizedVertices, d_shortestDistances,
                d_updatingShortestDistances, num_vtx);
            HANDLE_ERR(cudaPeekAtLastError());
            HANDLE_ERR(cudaDeviceSynchronize());
        }

        HANDLE_ERR(cudaMemcpy(h_finalizedVertices, d_finalizedVertices,
                sizeof(bool) * num_vtx, cudaMemcpyDeviceToHost));
    }

    HANDLE_ERR(cudaMemcpy(h_shortestDistances, d_shortestDistances, sizeof(int) * num_vtx, cudaMemcpyDeviceToHost));

    free(h_finalizedVertices);

    HANDLE_ERR(cudaFree(d_beg_pos));
    HANDLE_ERR(cudaFree(d_adj_list));
    HANDLE_ERR(cudaFree(d_weights));
    HANDLE_ERR(cudaFree(d_finalizedVertices));
    HANDLE_ERR(cudaFree(d_shortestDistances));
    HANDLE_ERR(cudaFree(d_updatingShortestDistances));
}

// Get index of vertex not included in path with min dist
int minDistance(int *shortestDistances, bool *finalizedVertices, 
        const int sourceVertex, const int N) {
    int minIndex = sourceVertex;
    int min = INT_MAX;
    for (int v = 0; v < N; v++)
        if (finalizedVertices[v] == false && shortestDistances[v] <= min) {
            min = shortestDistances[v];
            minIndex = v;
        }
    return minIndex;
}

void dijkstraCPU(int *beg_pos, int *adj_list, int *weights, int *h_shortestDistances, int sourceVertex, const int N) {

    // --- h_finalizedVertices[i] is true if vertex i is included in the shortest path tree
    //     or the shortest distance from the source node to i is finalized
    bool *h_finalizedVertices = (bool *)malloc(N * sizeof(bool));

    for (int i = 0; i < N; i++) {
        h_shortestDistances[i] = INT_MAX;
        h_finalizedVertices[i] = false;
    }

    h_shortestDistances[sourceVertex] = 0;

    for (int iterCount = 0; iterCount < N - 1; iterCount++) {

        int currentVertex = minDistance(h_shortestDistances, h_finalizedVertices,
                sourceVertex, N);

        h_finalizedVertices[currentVertex] = true;

        // --- Relaxation loop
        for (int v = 0; v < N; v++) {

            // --- Update dist[v] only if it is not in h_finalizedVertices, there is an edge
            //     from u to v, and the cost of the path from the source vertex to v through
            //     currentVertex is smaller than the current value of h_shortestDistances[v]
            bool found = false;
            int idx = 0;
            for (int i = beg_pos[currentVertex]; i < beg_pos[currentVertex + 1]; 
                    i++) {
                if (v == adj_list[i]) {
                    found = true;
                    idx = i;
                }
            }
            bool update = !h_finalizedVertices[v] && found &&
                h_shortestDistances[currentVertex] != INT_MAX &&
                h_shortestDistances[currentVertex] + weights[idx] <
                    h_shortestDistances[v];
            if (update)
                h_shortestDistances[v] = h_shortestDistances[currentVertex]
                    + weights[idx];
        }
    }
}

int main(int argc, char **argv) {
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

    double cpu_start = wtime();
    dijkstraCPU(beg_pos, adj_list, weight, shortest_dist_cpu, source_vertex, csr->vert_count);
    double cpu_end = wtime();
    cout << "CPU completed in " << cpu_end - cpu_start << endl;
    
    int *shortest_dist_gpu = (int *) malloc(csr->vert_count * sizeof(int));
    double gpu_start = wtime();
    dijkstraGPU(beg_pos, adj_list, weight, source_vertex, shortest_dist_gpu, csr->vert_count, csr->edge_count);
    double gpu_end = wtime();
    cout << "GPU completed in " << gpu_end - gpu_start << endl;

    for (int i = 0; i < csr->vert_count; i++) {
        if (shortest_dist_gpu[i] != shortest_dist_cpu[i]) {
            cout << "Index " << i << " fail: " << "CPU " << shortest_dist_cpu[i] 
            << " GPU " << shortest_dist_gpu[i] << endl;
        }
        assert(shortest_dist_gpu[i] == shortest_dist_cpu[i]);
    }
}
