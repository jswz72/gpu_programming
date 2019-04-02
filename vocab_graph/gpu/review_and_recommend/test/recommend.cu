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

__device__ double DOUBLE_MAX = std::numeric_limits<double>::max();
__device__ double DOUBLE_INF = std::numeric_limits<double>::infinity();

// Inverse sum rule, closness of vtx to all sources
__device__ double get_collective_dist(double *dist, int rows, int cols, int col) {
    double sum = 0;
    for (int i = 0; i < rows; i++) {
        sum += (1 / dist[i * cols + col]);
    }
    return sum;
}

__device__ long min_dist(double *distances, unsigned int *path, int vert_count)
{
    double min = DOUBLE_MAX;
    long min_idx;
    for (int i = 0; i < vert_count; i++)
    {
        if (!path[i] && distances[i] <= min)
        {
            min = distances[i];
            min_idx = i;
        }
    }
    return min_idx;
}

/**
 * Find shortest weighted path to all nodes from source using djikstra's algorithm
 */
__global__ void shortest_path_weights_kernel(long *beg_pos, long *adj_list, double *weight, int vert_count, int source, double *distances)
{
    // bitset true if included in path
    unsigned int *path = new unsigned int[vert_count];
    for (int i = 0; i < vert_count; i++)
    {
        distances[i] = DOUBLE_MAX;
        path[i] = 0;
    }

    distances[source] = 0;
    for (int count = 0; count < vert_count - 1; count++)
    {
        long cur = min_dist(distances, path, vert_count);
        path[cur] = true;

        // Update distances
        for (int i = beg_pos[cur]; i < beg_pos[cur+1]; i++)
        {
			int neighbor = adj_list[i];
            if (!path[neighbor] && 
                    distances[cur] != DOUBLE_MAX &&
                     distances[cur] + weight[i] < distances[neighbor])
            {
                double to_write = distances[cur] + weight[i];
                distances[neighbor] = to_write;
            }
        }
    }
}


__global__ void collective_closest_kernel(double *dist, int num_source_words, int vert_count, int *word_ids, double *dists) {

    // Word has no relation to given set
    double no_relation = (1 / DOUBLE_MAX) * num_source_words;

    // Get collective dist of vtx (col) to all source words (row)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (!tid)
        printf("Starting collective closest\n");

    while (tid < vert_count) {
        double my_dist = get_collective_dist(dist, num_source_words, vert_count, tid);
        //printf("%d\n", dist[tid]);
        bool append = my_dist != DOUBLE_INF && my_dist != no_relation;
        if (append) {
            word_ids[tid] = tid;
            dists[tid] = my_dist;
        }
        else {
            word_ids[tid] = -1;
            dists[tid] = -1;
        }
        tid += blockDim.x * gridDim.x;
    }
    if (!tid)
        printf("Done with collective closest\n");
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

    // Allocate and copy device variables

    // IDXs of source_words
    int *source_idxs_d;
    HANDLE_ERR(cudaMalloc((void **) &source_idxs_d, sizeof(int) * source_word_idxs.size()));
    HANDLE_ERR(cudaMemcpy (source_idxs_d, source_word_idxs.data(), sizeof(int) * source_word_idxs.size(), cudaMemcpyHostToDevice));

    // Beg pos array
    long *beg_pos_d;
    HANDLE_ERR(cudaMalloc((void **) &beg_pos_d, sizeof(long) * (csr->vert_count + 1)));
    HANDLE_ERR(cudaMemcpy (beg_pos_d, csr->beg_pos, sizeof(long) * (csr->vert_count + 1), cudaMemcpyHostToDevice));

    // Adj list
    long *csr_d;
    HANDLE_ERR(cudaMalloc((void **) &csr_d, sizeof(long) * csr->edge_count));
    HANDLE_ERR(cudaMemcpy (csr_d, csr->csr, sizeof(long) * csr->edge_count, cudaMemcpyHostToDevice));

    // Weights array
    double *weight_d;
    HANDLE_ERR(cudaMalloc((void **) &weight_d, sizeof(double) * csr->edge_count));
    HANDLE_ERR(cudaMemcpy (weight_d, csr->weight, sizeof(double) * csr->edge_count, cudaMemcpyHostToDevice));

    // Matrix of dists results of sssp
    double *dist_mat_d;
    HANDLE_ERR(cudaMalloc((void **) &dist_mat_d, sizeof(double) * csr->vert_count * num_source_words));

    // Word ids of summed array
    int *word_ids_d;
    HANDLE_ERR(cudaMalloc((void **) &word_ids_d, sizeof(int) * csr->vert_count));

    // Word dists of summed array
    double *dists_d;
    HANDLE_ERR(cudaMalloc((void **) &dists_d, sizeof(double) * csr->vert_count));

    // Number of recommendations to make/made
    int *num_recs_d;
    HANDLE_ERR(cudaMalloc((void **) &num_recs_d, sizeof(int)));
    HANDLE_ERR(cudaMemcpy (num_recs_d, &num_recs, sizeof(int), cudaMemcpyHostToDevice));





    // Just doing it for first word this time
    int tid = 0;
    double starttime = wtime();
    printf("Starting SSSP\n");

    // Repurposing dist_mat_d
    printf("num sw: %d\n", num_source_words);
    printf("asdf: %d\n", source_word_idxs[tid]);
    shortest_path_weights_kernel<<<1, 1>>>(beg_pos_d, csr_d, weight_d, csr->vert_count, source_word_idxs[tid], dist_mat_d);
    cudaDeviceSynchronize();
    printf("SSSP Done\n");
    double endtime = wtime();
    cout << "Time for SSSP: " << endtime - starttime << endl;

    collective_closest_kernel <<< 128, 128 >>> (dist_mat_d, num_source_words, csr->vert_count, word_ids_d, dists_d);
    cudaDeviceSynchronize();


    
    
    
    // BELOW THIS SAME 
    // Copy back closest_words
    int *ids = (int *)malloc(sizeof(int*) * csr->vert_count);
    double *dists = (double *)malloc(sizeof(double*) * csr->vert_count);
    HANDLE_ERR(cudaMemcpy (dists, dists_d, sizeof(double) * csr->vert_count, cudaMemcpyDeviceToHost));
    HANDLE_ERR(cudaMemcpy (ids, word_ids_d, sizeof(int) * csr->vert_count, cudaMemcpyDeviceToHost));

    WordDist **wd = (WordDist**)malloc(sizeof(WordDist*) * csr->vert_count);
    for (int i = 0; i < csr->vert_count; i++) {
        wd[i] = new WordDist(dists[i], ids[i]);
    }

    // Sort in terms of collect closest
	std::sort(wd, wd + csr->vert_count, [](WordDist *a, WordDist *b) -> bool
    {
        return a->dist > b->dist;
    });

	cout << "\nLearning recommendations :" << endl;
	for (int i = 0; i < num_recs; i++) {
        if (wd[i]->word_id == -1) {
            cout << "End" << endl;
            break;
        }
		cout << words[wd[i]->word_id] << " (Value: "
			<< wd[i]->dist << ")" << endl;
	}
	return 0;	
}
