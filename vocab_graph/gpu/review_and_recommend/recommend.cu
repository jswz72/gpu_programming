#include <iostream>
#include <limits>
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include "graph.h"
#include "error_handler.h"
#include "wtime.h"

using std::cout;
using std::endl;
using std::string;

struct WordDist {
    double dist;
    int word_id;
    WordDist(double dist, int id): dist(dist), word_id(id) {};
};

typedef graph<long, long, double, long, long, double> CSR;

std::vector<string> get_word_mapping(const char *mapping_file) {
	std::ifstream infile(mapping_file);
	std::vector<string> words;
	string line;
	while (std::getline(infile, line))
		words.push_back(line);
	return words;
}

#define BLOCK_SIZE 16;
#define NUM_ASYNCHRONOUS_ITERATIONS 20  // Number of async loop iterations before attempting to read results back

const bool const_true = true;

// Check whether all verticies done
__global__ void CheckDoneKernel(bool *finished_verts, int num_vtx, bool *finished) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (*finished && tid < num_vtx) {
        if (finished_verts[tid] == true)
           *finished = false;
        tid += blockDim.x * gridDim.x;
    }

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
    //cout << "Time to init and copy: " << e1 - s1 << endl;
    double start = wtime();
    // Mask to 0's, cost and updating arrays to inf
    init_sssp_arrs <<<div_ceil(num_vtx, 16), 16 >>>
        (d_finished_verts, d_dists, d_update_dists, source, num_vtx);
    HANDLE_ERR(cudaDeviceSynchronize());

    HANDLE_ERR(cudaMemcpy(finished_vtxs, d_finished_verts, 
                sizeof(bool) * num_vtx, cudaMemcpyDeviceToHost));

    bool finished = false;
    bool *d_finished;
    HANDLE_ERR(cudaMalloc(&d_finished, sizeof(bool)));
    while (!finished) {
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
        HANDLE_ERR(cudaMemcpy(d_finished, &const_true, sizeof(bool), cudaMemcpyHostToDevice));
        CheckDoneKernel <<< div_ceil(num_vtx, 16), 16 >>> (d_finished_verts, num_vtx, d_finished);
        HANDLE_ERR(cudaDeviceSynchronize());
        HANDLE_ERR(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
    }
    double end = wtime();
    double s2 = wtime();
    //cout << "Inner GPU time: " << end - start << endl;

    HANDLE_ERR(cudaMemcpy(dists, d_dists, sizeof(int) * num_vtx, cudaMemcpyDeviceToHost));

    free(finished_vtxs);

    HANDLE_ERR(cudaFree(d_beg_pos));
    HANDLE_ERR(cudaFree(d_adj_list));
    HANDLE_ERR(cudaFree(d_weights));
    HANDLE_ERR(cudaFree(d_finished_verts));
    HANDLE_ERR(cudaFree(d_dists));
    HANDLE_ERR(cudaFree(d_update_dists));
    double e2 = wtime();
    //cout << "Copy out and free time: " << e2 - s2 << endl;
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

int *shortest_path_weights(CSR *csr, int source_word) {
    int *beg_pos = (int *)malloc(csr->vert_count * sizeof(int));
    int *adj_list = (int *)malloc(csr->edge_count * sizeof(int));
    int *weight = (int *)malloc(csr->edge_count * sizeof(int));

    // Long -> int
    for (int i = 0; i < csr->vert_count; i++) 
        beg_pos[i] = (int) csr->beg_pos[i];
    
    // Long -> int
    for (int i = 0; i < csr->edge_count; i++) 
        adj_list[i] = (int) csr->csr[i];
    // Double -> int
    for (int i = 0; i < csr->edge_count; i++) 
        weight[i] = (int) (csr->weight[i] * 1000);

    int *shortest_dist_gpu = (int *) malloc(csr->vert_count * sizeof(int));
    dijkstraGPU(beg_pos, adj_list, weight, source_word, shortest_dist_gpu, csr->vert_count, csr->edge_count);
    return shortest_dist_gpu;
}

double get_collective_dist(int *dist, int rows, int cols, int col) {
    double sum = 0;
    for (int i = 0; i < rows; i++) {
        //cout << dist[i * cols + col] << endl;
        if (dist[i * cols + col] == 0) {
            return 0;
        }
        sum += (1 / (double)dist[i * cols + col]);
    }
    return sum;
}

WordDist** collective_closest(std::vector<int> &source_words, int n, CSR *csr) {
    // Row for each source word, col for each vtx
    int *dist = (int *)malloc(sizeof(int) * n * csr->vert_count);

    // All vtxs, sorted in terms of closest
	WordDist ** word_dist = (WordDist **)malloc(sizeof(WordDist*) * csr->vert_count);

    // Fill out dists to all vtxs (dist col) from word (dist row)
    for (int i = 0; i < n; i++) {
        int cols = csr->vert_count;
        int *shortest_paths = shortest_path_weights(csr, source_words[i]);
        for (int j = 0; j < cols; j++) {
            dist[i * cols + j] = shortest_paths[j];
        }
    }

    // Get collective dist of vtx (col) to all source words (row)
    for (int i = 0; i < csr->vert_count; i++) {
        WordDist *wd = new WordDist(get_collective_dist(dist, n, csr->vert_count, i), i);
        word_dist[i] = wd;
    }
    // Sort in terms of collect closest
	std::sort(word_dist, word_dist + csr->vert_count, [](WordDist *a, WordDist *b) -> bool
    {
        return a->dist > b->dist;
    });

	return word_dist;
}

std::vector<WordDist*> recommend(CSR *csr, std::vector<int> &source_words, int num_recs) {
	double start_time = wtime();
    WordDist** word_dist = collective_closest(source_words, source_words.size(), csr);
	cout << "Total aglo time: " << wtime() - start_time << endl;

	std::vector<WordDist*> related_words;
	
    // Word has no relation to given set
    int no_relation = (1 / (double)INT_MAX) * source_words.size();
	
    // Filter out all dists that are 0 (source word) or not related to any source words
    std::copy_if(word_dist, word_dist + csr->vert_count, std::back_inserter(related_words), 
			[no_relation] (WordDist *a) -> bool {
                    return a->dist != 0 && a->dist != no_relation;
    });
    
	if (num_recs < related_words.size())
		related_words.resize(num_recs);

	double final_time = wtime() - start_time;
	cout << "Final Time: " << final_time << endl;

	return related_words;
}

int main(int argc, char **argv) {
	if (argc < 5) {
		cout << "Input: ./exe base_file mapping_file num_recs source_words..." << endl;
		return 1;
	}
	
    // Get files and make graph
	string base_filename (argv[1]);
	string beg_file = base_filename + "_beg_pos.bin";
	string csr_file = base_filename + "_csr.bin";
	string weight_file = base_filename + "_weight.bin";
	const char *mapping_file = argv[2];
	graph<long, long, double, long, long, double> *csr = 
		new graph <long, long, double, long, long, double>
		(beg_file.c_str(), csr_file.c_str(), weight_file.c_str());

    // Output Graph Info
	std::cout << "Edges: " << csr->edge_count << std::endl;
    std::cout << "Verticies: " << csr->vert_count << std::endl;

	int num_recs = atoi(argv[3]);
	int num_source_words = argc - 4;

    // Get source word indices and make sure they are in graph
	std::vector<int> source_word_idxs;
	std::vector<string> words = get_word_mapping(mapping_file);
    for (int i = 0; i < num_source_words; i++) {
		const char *source_word = argv[i + 4];
		auto it = std::find(words.begin(), words.end(), source_word);
		if (it == words.end()) {
			cout << "Not found in graph: " << source_word << endl;
			return 1;
		}
		int idx = std::distance(words.begin(), it);
		source_word_idxs.push_back(idx);
    }

    std::vector<WordDist*> closest_words = recommend(csr, source_word_idxs, num_recs);
    cout << "\nLearning recommendations :" << endl;
	for (int i = 0; i < closest_words.size(); i++) {
		cout << words[closest_words[i]->word_id] << " (Value: "
			<< closest_words[i]->dist << ")" << endl;
	}
	if (closest_words.size() < num_recs)
		cout << "End" << endl;

    return 0;	
}
