#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>
#include <omp.h>
#include <thrust/sort.h>
#include "graph.h"
#include "review_and_recommend.h"

using std::cout;
using std::endl;
using std::string;

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
__device__ double *shortest_path_weights(long *beg_pos, long *adj_list, double *weight, int vert_count, int source)
{
    // distance from start to vertex 
    double *distances = new double[vert_count];
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
        printf("%ld, %ld\n", beg_pos[cur], beg_pos[cur+1]);
        if (beg_pos[cur+1] == 0)
            printf("DEBUG: c: %d, c1:%d\n", cur, cur+1);
        for (int i = beg_pos[cur]; i < beg_pos[cur+1]; i++)
        {
			int neighbor = adj_list[i];
            if (!path[neighbor] && 
                    distances[cur] != DOUBLE_MAX &&
                     distances[cur] + weight[i] < distances[neighbor])
            {
                distances[neighbor] = distances[cur] + weight[i];
            }
        }
    }
    return distances;
}

__global__ void shortest_paths_kernel(int *source_words, int n, long *beg_pos, long *adj_list, double *weight, int vert_count, double *dist) {
    printf("Starting sssp\n");
    // Row for each source word, col for each vtx
    // All vtxs, sorted in terms of closest

    // Fill out dists to all vtxs (dist col) from word (dist row)

    //int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //while (tid < n) {
    for (int tid = 0; tid < n; tid++) {
        int cols = vert_count;
        double *shortest_paths = shortest_path_weights(beg_pos, adj_list, weight, vert_count, source_words[tid]);
        printf("Fin\n");
        for (int j = 0; j < cols; j++) {
            dist[tid * cols + j] = shortest_paths[j];
        }
        //tid += blockDim.x * gridDim.x;
    }
    printf("Done with sssps\n");
}

__global__ void collective_closest_kernel(double *dist, int num_source_words, int vert_count, int *word_ids, double *dists) {

    // Word has no relation to given set
    double no_relation = (1 / DOUBLE_MAX) * num_source_words;

    // Get collective dist of vtx (col) to all source words (row)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < vert_count) {
        double my_dist = get_collective_dist(dist, num_source_words, vert_count, tid);
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
    //printf("Done with collective closest\n");
}

/*std::vector<int> review (CSR *csr, std::vector<int> &reviewed, std::vector<int> &learned, int rev_count) {
	WordDist** word_dist = collective_closest(reviewed, reviewed.size(), csr);
	std::vector<int> cur_review_set;

	// Get intersection of recommended words (word_dist) and already leared words (in sorted order)
	for (int i = 0; i < csr->vert_count; i++) {
		int cur_id = word_dist[i]->word_id;
		bool is_learned = std::find(learned.begin(), learned.end(), cur_id) != learned.end();
		bool is_in_cur_rev= std::find(cur_review_set.begin(), cur_review_set.end(), cur_id) != cur_review_set.end();
		
		// Skip already reviewed words
		if (word_dist[i]->dist == DOUBLE_INF)
			continue;
		if (is_learned && !is_in_cur_rev)
			cur_review_set.push_back(cur_id);
		if (cur_review_set.size() == rev_count)
			break;
	}

	return cur_review_set;
}*/
