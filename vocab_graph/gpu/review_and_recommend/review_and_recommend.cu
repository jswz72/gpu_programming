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

__device__ int min_dist(double *distances, unsigned int *path, int vert_count)
{
    double min = DOUBLE_MAX;
    int min_idx;
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
__device__ double *shortest_path_weights(int *beg_pos, int *adj_list, double *weight, int vert_count, int source)
{
    printf("vc: %d\n", vert_count);
    printf("s: %d\n", source);
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
        int cur = min_dist(distances, path, vert_count);
        path[cur] = true;

        // Update distances
        if (count < 5)
            printf("bg: %d, bg: %d\n", beg_pos[cur], beg_pos[cur+1]);
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

__device__ WordDist** collective_closest(int *source_words, int n, int *beg_pos, int *adj_list, double *weight, int vert_count) {
    printf("yes2\n");
    // Row for each source word, col for each vtx
    double *dist = (double *)malloc(sizeof(double) * n * vert_count);

    // All vtxs, sorted in terms of closest
	WordDist ** word_dist = (WordDist **)malloc(sizeof(WordDist*) * vert_count);

    // Fill out dists to all vtxs (dist col) from word (dist row)
    printf("n: %d\n", n);
    for (int i = 0; i < n; i++) {
        int cols = vert_count;
        double *shortest_paths = shortest_path_weights(beg_pos, adj_list, weight, vert_count, source_words[i]);
        for (int j = 0; j < cols; j++) {
            dist[i * cols + j] = shortest_paths[j];
        }
        printf("go tit\n");
    }

    // Get collective dist of vtx (col) to all source words (row)
    for (int i = 0; i < vert_count; i++) {
        WordDist *wd = new WordDist(get_collective_dist(dist, n, vert_count, i), i);
        word_dist[i] = wd;
    }
    // Sort in terms of collect closest
	thrust::sort(word_dist, word_dist + vert_count, [](WordDist *a, WordDist *b) -> bool
    {
        return a->dist > b->dist;
    });

	return word_dist;
}

__global__ void recommend_kernel(int *beg_pos, int *adj_list, double *weight, int *source_words, 
        int num_source_words, int vert_count, WordDist **closest_words, int *num_recs) {
    printf("yes");
    WordDist** related_words = collective_closest(source_words, num_source_words, beg_pos, adj_list, weight, vert_count);

    // Word has no relation to given set
    double no_relation = (1 / DOUBLE_MAX) * num_source_words;
	
    int len = 0;
    // Filter out all dists that are 0 (source word) or not related to any source words
    for (int i = 0; i < vert_count; i++) {
        bool append = related_words[i]->dist != DOUBLE_INF && related_words[i]->dist != no_relation;
        if (append)
            closest_words[len++] = related_words[i];
    }
    *num_recs = len;
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
