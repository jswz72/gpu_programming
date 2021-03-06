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

const bool const_true = true;

// Get word vector from word file
std::vector<string> get_word_mapping(const char *mapping_file) {
	std::ifstream infile(mapping_file);
	std::vector<string> words;
	string line;
	while (std::getline(infile, line))
		words.push_back(line);
	return words;
}

// Check whether all verticies done
__global__ void check_done_kernel(bool *mask, int num_vtx, bool *finished) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (*finished && tid < num_vtx) {
        if (mask[tid])
           *finished = false;
        tid += blockDim.x * gridDim.x;
    }

}

// Division ceiling
int div_ceil(int numer, int denom) {
    return (numer % denom != 0) ? (numer / denom + 1) : (numer / denom);
}

__global__ void init_sssp_data(bool * d_mask, 
        int* d_dists, 
        int* d_update_dists, const int source, 
        const int num_vtx) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vtx) {
        if (source == tid) {
            d_mask[tid] = true;
            d_dists[tid] = 0;
            d_update_dists[tid] = 0;
        }
        else {
            d_mask[tid] = false;
            d_dists[tid] = INT_MAX;
            d_update_dists[tid] = INT_MAX;
        }
    }
}

// First portion GPU work for djikstra's - get new values for updating dists
__global__  void get_dists_kernel(const int * beg_pos, 
        const int* adj_list, const int * weights,
        bool * mask, int* dists,
        int * update_dists, const int num_vtx) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < num_vtx) {
        if (mask[tid] == true) {
            mask[tid] = false;
            for (int edge = beg_pos[tid]; edge < beg_pos[tid + 1]; edge++) {
                int other = adj_list[edge];
                atomicMin(&update_dists[other], 
                        dists[tid] + weights[edge]);
            }
        }
    }
}

// Second portion of GPU work for djikstra's
__global__  void update_dists_kernel(const int * beg_pos, 
        const int * adj_list, 
        const int* weights, bool * mask,
        int* dists, 
        int* update_dists, const int num_vtx) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vtx) {
        if (dists[tid] > update_dists[tid]) {
            dists[tid] = update_dists[tid];
            mask[tid] = true; 
        }
        update_dists[tid] = dists[tid];
    }
}

// Main function handling SSSP using kernels to implement parallel dijkstra's algorithm
void SSSP_GPU(int *d_beg_pos, int *d_adj_list, int *d_weights, const int source, 
        int * dists, int num_vtx, int num_edge) {
    
    // Mask array
    bool *d_mask;
    HANDLE_ERR(cudaMalloc(&d_mask, sizeof(bool) * num_vtx));
    // Cost array
    int *d_dists;
    HANDLE_ERR(cudaMalloc(&d_dists, sizeof(int) * num_vtx));
    // Updating cost array
    int *d_update_dists;
    HANDLE_ERR(cudaMalloc(&d_update_dists, sizeof(int) * num_vtx));


    bool *finished_vtxs = (bool *)malloc(sizeof(bool) * num_vtx);
    // Mask to 0's, cost and updating arrays to inf
    init_sssp_data <<<div_ceil(num_vtx, 16), 16 >>>
        (d_mask, d_dists, d_update_dists, source, num_vtx);
    HANDLE_ERR(cudaDeviceSynchronize());

    HANDLE_ERR(cudaMemcpy(finished_vtxs, d_mask, 
                sizeof(bool) * num_vtx, cudaMemcpyDeviceToHost));

    bool finished = false;
    bool *d_finished;
    HANDLE_ERR(cudaMalloc(&d_finished, sizeof(bool)));
    while (!finished) {
        get_dists_kernel <<<div_ceil(num_vtx, 16), 16 >>>(d_beg_pos, d_adj_list, 
                d_weights, d_mask, d_dists, d_update_dists, num_vtx);
        HANDLE_ERR(cudaDeviceSynchronize());
        update_dists_kernel <<<div_ceil(num_vtx, 16), 16 >>>(d_beg_pos, d_adj_list, 
            d_weights, d_mask, d_dists,
            d_update_dists, num_vtx);
        HANDLE_ERR(cudaDeviceSynchronize());
        HANDLE_ERR(cudaMemcpy(d_finished, &const_true, sizeof(bool), cudaMemcpyHostToDevice));
        check_done_kernel <<< div_ceil(num_vtx, 16), 16 >>> (d_mask, num_vtx, d_finished);
        HANDLE_ERR(cudaDeviceSynchronize());
        HANDLE_ERR(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
    }

    HANDLE_ERR(cudaMemcpy(dists, d_dists, sizeof(int) * num_vtx, cudaMemcpyDeviceToHost));

    free(finished_vtxs);
    HANDLE_ERR(cudaFree(d_mask));
    HANDLE_ERR(cudaFree(d_dists));
    HANDLE_ERR(cudaFree(d_update_dists));

}

// Get index of vertex not included in mask with min dist
int min_dist(int *dists, bool *mask, 
        const int source, const int N) {
    int min_index = source;
    int min = INT_MAX;
    for (int v = 0; v < N; v++)
        if (mask[v] == false && dists[v] <= min) {
            min = dists[v];
            min_index = v;
        }
    return min_index;
}


// Get collective dist via inverse sum of column
__device__ double get_collective_dist(int *dist, int rows, int cols, int col) {
    double sum = 0;
    for (int i = 0; i < rows; i++) {
        if (dist[i * cols + col] == 0) {
            return 0;
        }
        sum += (1 / (double)dist[i * cols + col]);
    }
    return sum;
}

// Get collective dist, 1 thread per column of dist matrix
__global__ void collective_dist_kernel(int *dist, int rows, int cols, 
        double *col_dist)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < cols) {
        col_dist[tid] = get_collective_dist(dist, rows, cols, tid);
        tid += blockDim.x * gridDim.x;
    }
}

WordDist** collective_closest(std::vector<int> &source_words, int n, CSR *csr) {
    int *beg_pos = (int *)malloc(csr->vert_count * sizeof(int));
    int *adj_list = (int *)malloc(csr->edge_count * sizeof(int));
    int *weight = (int *)malloc(csr->edge_count * sizeof(int));

    for (int i = 0; i < csr->vert_count; i++) 
        beg_pos[i] = (int) csr->beg_pos[i];
    for (int i = 0; i < csr->edge_count; i++) 
        adj_list[i] = (int) csr->csr[i];
    // Double -> int
    for (int i = 0; i < csr->edge_count; i++) 
        weight[i] = (int) (csr->weight[i] * 1000);

    int *shortest_dist_gpu = (int *) malloc(csr->vert_count * sizeof(int));
    int *d_beg_pos;
    HANDLE_ERR(cudaMalloc(&d_beg_pos, sizeof(int) * csr->vert_count));
    int *d_adj_list;
    HANDLE_ERR(cudaMalloc(&d_adj_list, sizeof(int) * csr->edge_count));
    int *d_weights;
    HANDLE_ERR(cudaMalloc(&d_weights, sizeof(int) * csr->edge_count));
    

    HANDLE_ERR(cudaMemcpy(d_beg_pos, beg_pos, sizeof(int) * csr->vert_count, cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy(d_adj_list, adj_list, sizeof(int) * csr->edge_count,
                cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy(d_weights, weight, sizeof(int) * csr->edge_count,
                cudaMemcpyHostToDevice));


    double inner_start = wtime();
    // Row for each source word, col for each vtx
    int *dist = (int *)malloc(sizeof(int) * n * csr->vert_count);

    // All vtxs, sorted in terms of closest
	WordDist ** word_dist = (WordDist **)malloc(sizeof(WordDist*) * csr->vert_count);

    // Fill out dists to all vtxs (dist col) from word (dist row)
    for (int i = 0; i < n; i++) {
        int cols = csr->vert_count;
        SSSP_GPU(d_beg_pos, d_adj_list, d_weights, source_words[i], shortest_dist_gpu, csr->vert_count, csr->edge_count);
        for (int j = 0; j < cols; j++) {
            dist[i * cols + j] = shortest_dist_gpu[j];
        }
    }



    HANDLE_ERR(cudaFree(d_beg_pos));
    HANDLE_ERR(cudaFree(d_adj_list));
    HANDLE_ERR(cudaFree(d_weights));

    int *d_dist;
    HANDLE_ERR(cudaMalloc(&d_dist, sizeof(int) * n * csr->vert_count));
    HANDLE_ERR(cudaMemcpy(d_dist, dist, sizeof(int) * n * csr->vert_count, cudaMemcpyHostToDevice));

    double *d_col_dist;
    HANDLE_ERR(cudaMalloc(&d_col_dist, sizeof(double) * csr->vert_count));

    collective_dist_kernel<<<div_ceil(csr->vert_count, 16), 16>>>
        (d_dist, n, csr->vert_count, d_col_dist);

    double *col_dist = (double *)malloc(sizeof(double) * csr->vert_count);
    HANDLE_ERR(cudaMemcpy(col_dist, d_col_dist, sizeof(double) * csr->vert_count, cudaMemcpyDeviceToHost));
    
    // Get collective dist of vtx (col) to all source words (row)
    for (int i = 0; i < csr->vert_count; i++) {
        WordDist *wd = new WordDist(col_dist[i], i);
        word_dist[i] = wd;
    }
    // Sort in terms of collect closest
	std::sort(word_dist, word_dist + csr->vert_count, [](WordDist *a, WordDist *b) -> bool
    {
        return a->dist > b->dist;
    });

    cout << "GPU Algorithm Time: " << wtime() - inner_start << endl;

	return word_dist;
}

std::vector<WordDist*> recommend(CSR *csr, std::vector<int> &source_words, int num_recs) {
	double start_time = wtime();
    WordDist** word_dist = collective_closest(source_words, source_words.size(), csr);
	cout << "Total  GPU time: " << wtime() - start_time << endl;

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

    long total_degrees = 0;
    for (int i = 0; i < csr->vert_count; i++) {
        total_degrees += csr->beg_pos[i + 1] - csr->beg_pos[i];
    }
    cout << "Average degree: " << total_degrees / csr->vert_count << endl;

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
