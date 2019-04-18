#include <limits>
#include <algorithm>
#include <iostream>
#include "error_handler.h"

using std::cout;
using std::endl;

#define BLOCK_SIZE 16;
#define NUM_ASYNCHRONOUS_ITERATIONS 20  // Number of async loop iterations before attempting to read results back

/***************************/
/* MASKARRAYEMPTY FUNCTION */
/***************************/
// --- Check whether all the vertices have been finalized. This tells the algorithm whether it needs to continue running or not.
bool allFinalizedVertices(bool *finalizedVertices, int numVertices) {

    for (int i = 0; i < numVertices; i++)  if (finalizedVertices[i] == true) { return false; }

    return true;
}

//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

/*************************/
/* ARRAY INITIALIZATIONS */
/*************************/
__global__ void initializeArrays(bool * __restrict__ d_finalizedVertices, int* __restrict__ d_shortestDistances, int* __restrict__ d_updatingShortestDistances,
                                 const int sourceVertex, const int numVertices) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (sourceVertex == tid) {

            d_finalizedVertices[tid]            = true;
            d_shortestDistances[tid]            = 0;
            d_updatingShortestDistances[tid]    = 0; }

        else {

            d_finalizedVertices[tid]            = false;
            d_shortestDistances[tid]            = INT_MAX;
            d_updatingShortestDistances[tid]    = INT_MAX;
        }
    }
}

/**************************/
/* DIJKSTRA GPU KERNEL #1 */
/**************************/
__global__  void Kernel1(const int * __restrict__ vertexArray, const int* __restrict__ edgeArray,
                         const int * __restrict__ weightArray, bool * __restrict__ finalizedVertices, int* __restrict__ shortestDistances,
                         int * __restrict__ updatingShortestDistances, const int numVertices, const int numEdges) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (finalizedVertices[tid] == true) {

            finalizedVertices[tid] = false;

            int edgeStart = vertexArray[tid];
            int edgeEnd = vertexArray[tid + 1];

            for (int edge = edgeStart; edge < edgeEnd; edge++) {
                int nid = edgeArray[edge];
                atomicMin(&updatingShortestDistances[nid], shortestDistances[tid] + weightArray[edge]);
            }
        }
    }
}

/**************************/
/* DIJKSTRA GPU KERNEL #1 */
/**************************/
__global__  void Kernel2(const int * __restrict__ vertexArray, const int * __restrict__ edgeArray, const int* __restrict__ weightArray,
                         bool * __restrict__ finalizedVertices, int* __restrict__ shortestDistances, int* __restrict__ updatingShortestDistances,
                         const int numVertices) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (shortestDistances[tid] > updatingShortestDistances[tid]) {
            shortestDistances[tid] = updatingShortestDistances[tid];
            finalizedVertices[tid] = true; }

        updatingShortestDistances[tid] = shortestDistances[tid];
    }
}

/* dijkstraGPU FUNCTION */
/************************/
void dijkstraGPU(int *beg_pos, int *adj_list, int *weights, const int sourceVertex, int * __restrict__ h_shortestDistances, int num_vtx, int num_edge) {

    // --- Create device-side adjacency-list, namely, vertex array Va, edge array Ea and weight array Wa from G(V,E,W)
    int     *d_beg_pos;         HANDLE_ERR(cudaMalloc(&d_beg_pos,    sizeof(int)   *  num_vtx));
    int     *d_adj_list;           HANDLE_ERR(cudaMalloc(&d_adj_list,  sizeof(int)   * num_edge));
    int   *d_weights;         HANDLE_ERR(cudaMalloc(&d_weights,    sizeof(int) * num_edge));

    // --- Copy adjacency-list to the device
    HANDLE_ERR(cudaMemcpy(d_beg_pos, beg_pos, sizeof(int)   * num_vtx, cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy(d_adj_list,   adj_list,   sizeof(int)   * num_edge,    cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy(d_weights, weights, sizeof(int) * num_edge,    cudaMemcpyHostToDevice));

    // --- Create mask array Ma, cost array Ca and updating cost array Ua of size V
    bool    *d_finalizedVertices;           HANDLE_ERR(cudaMalloc(&d_finalizedVertices,       sizeof(bool)   * num_vtx));
    int   *d_shortestDistances;           HANDLE_ERR(cudaMalloc(&d_shortestDistances,       sizeof(int) * num_vtx));
    int   *d_updatingShortestDistances;   HANDLE_ERR(cudaMalloc(&d_updatingShortestDistances, sizeof(int) * num_vtx));

    bool *h_finalizedVertices = (bool *)malloc(sizeof(bool) * num_vtx);

    // --- Initialize mask Ma to false, cost array Ca and Updating cost array Ua to \u221e
    initializeArrays 
        <<<iDivUp(num_vtx, 16), 16 >>>
        (d_finalizedVertices, d_shortestDistances, d_updatingShortestDistances, sourceVertex, num_vtx);
    HANDLE_ERR(cudaPeekAtLastError());
    HANDLE_ERR(cudaDeviceSynchronize());

    // --- Read mask array from device -> host
    HANDLE_ERR(cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * num_vtx, cudaMemcpyDeviceToHost));

    while (!allFinalizedVertices(h_finalizedVertices, num_vtx)) {

        // --- In order to improve performance, we run some number of iterations without reading the results.  This might result
        //     in running more iterations than necessary at times, but it will in most cases be faster because we are doing less
        //     stalling of the GPU waiting for results.
        for (int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++) {

            Kernel1 <<<iDivUp(num_vtx, 16), 16 >>>(d_beg_pos, d_adj_list, d_weights, d_finalizedVertices, d_shortestDistances,
                                                            d_updatingShortestDistances, num_vtx, num_edge);
            HANDLE_ERR(cudaPeekAtLastError());
            HANDLE_ERR(cudaDeviceSynchronize());
            Kernel2 <<<iDivUp(num_vtx, 16), 16 >>>(d_beg_pos, d_adj_list, d_weights, d_finalizedVertices, d_shortestDistances, d_updatingShortestDistances,
                                                            num_vtx);
            HANDLE_ERR(cudaPeekAtLastError());
            HANDLE_ERR(cudaDeviceSynchronize());
        }

        HANDLE_ERR(cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * num_vtx, cudaMemcpyDeviceToHost));
    }

    // --- Copy the result to host
    HANDLE_ERR(cudaMemcpy(h_shortestDistances, d_shortestDistances, sizeof(int) * num_vtx, cudaMemcpyDeviceToHost));

    free(h_finalizedVertices);

    HANDLE_ERR(cudaFree(d_beg_pos));
    HANDLE_ERR(cudaFree(d_adj_list));
    HANDLE_ERR(cudaFree(d_weights));
    HANDLE_ERR(cudaFree(d_finalizedVertices));
    HANDLE_ERR(cudaFree(d_shortestDistances));
    HANDLE_ERR(cudaFree(d_updatingShortestDistances));
}

/* minDistance FUNCTION */
/************************/
// --- Finds the vertex with minimum distance value, from the set of vertices not yet included in shortest path tree
int minDistance(int *shortestDistances, bool *finalizedVertices, const int sourceVertex, const int N) {

    // --- Initialize minimum value
    int minIndex = sourceVertex;
    int min = INT_MAX;

    for (int v = 0; v < N; v++)
        if (finalizedVertices[v] == false && shortestDistances[v] <= min) min = shortestDistances[v], minIndex = v;

    cout << "mIn: " << minIndex << endl;
    return minIndex;
}

void dijkstraCPU(int *beg_pos, int *adj_list, int *weights, int *h_shortestDistances, int sourceVertex, const int N) {

    // --- h_finalizedVertices[i] is true if vertex i is included in the shortest path tree
    //     or the shortest distance from the source node to i is finalized
    bool *h_finalizedVertices = (bool *)malloc(N * sizeof(bool));

    // --- Initialize h_shortestDistancesances as infinite and h_shortestDistances as false
    for (int i = 0; i < N; i++) h_shortestDistances[i] = INT_MAX, h_finalizedVertices[i] = false;

    // --- h_shortestDistancesance of the source vertex from itself is always 0
    h_shortestDistances[sourceVertex] = 0;

    // --- Dijkstra iterations
    for (int iterCount = 0; iterCount < N - 1; iterCount++) {

        // --- Selecting the minimum distance vertex from the set of vertices not yet
        //     processed. currentVertex is always equal to sourceVertex in the first iteration.
        int currentVertex = minDistance(h_shortestDistances, h_finalizedVertices, sourceVertex, N);

        // --- Mark the current vertex as processed
        h_finalizedVertices[currentVertex] = true;

        // --- Relaxation loop
        for (int v = 0; v < N; v++) {

            // --- Update dist[v] only if it is not in h_finalizedVertices, there is an edge
            //     from u to v, and the cost of the path from the source vertex to v through
            //     currentVertex is smaller than the current value of h_shortestDistances[v]
            bool found = false;
            int idx = 0;
            for (int i = beg_pos[currentVertex]; i < beg_pos[currentVertex + 1]; i++) {
                if (v == adj_list[i]) {
                    found = true;
                    idx = i;
                }
            }
            if (!h_finalizedVertices[v] &&
                found &&
                h_shortestDistances[currentVertex] != INT_MAX &&
                h_shortestDistances[currentVertex] + weights[idx] < h_shortestDistances[v])

                h_shortestDistances[v] = h_shortestDistances[currentVertex] + weights[idx];
        }
    }
}

int main() {
    int source_vertex = 0;

    int num_vtx = 4;
    int *beg_pos = new int[5];
    beg_pos[0] = 0;
    beg_pos[1] = 2;
    beg_pos[2] = 5;
    beg_pos[3] = 8;
    beg_pos[4] = 10;

    int *adj_list = new int[10];
    adj_list[0] = 1;
    adj_list[1] = 2;
    adj_list[2] = 0;
    adj_list[3] = 2;
    adj_list[4] = 3;
    adj_list[5] = 0;
    adj_list[6] = 1;
    adj_list[7] = 3;
    adj_list[8] = 1;
    adj_list[9] = 2;

    int *weights = new int[10];
    weights[0] = 2;
    weights[1] = 3;
    weights[2] = 2;
    weights[3] = 4;
    weights[4] = 4;
    weights[5] = 3;
    weights[6] = 4;
    weights[7] = 10;
    weights[8] = 4;
    weights[9] = 10;

    for (int i = 0; i < num_vtx; i++) {
        for (int j = beg_pos[i]; j < beg_pos[i + 1]; j++) {
            cout << i << " -> " << adj_list[j] << ": " << weights[j] << endl;
        }
        cout << endl;
    }

    int *shortest_dist_cpu = (int *)malloc(num_vtx * sizeof(int));
    dijkstraCPU(beg_pos, adj_list, weights, shortest_dist_cpu, source_vertex, num_vtx);
    cout << "CPU Results:" << endl;
    for (int i = 0; i < num_vtx; i++) {
        cout << shortest_dist_cpu[i] << endl;
    }
    cout << endl;

    int *shortest_dist_gpu = (int *) malloc(num_vtx * sizeof(int));
    dijkstraGPU(beg_pos, adj_list, weights, source_vertex, shortest_dist_gpu, 5, 10);

    cout << "GPU Results:" << endl;
    for (int i = 0; i < num_vtx; i++) {
        cout << shortest_dist_gpu[i] << endl;
    }
    cout << endl;
}
