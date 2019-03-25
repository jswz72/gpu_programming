#include <limits>
#include "graph.h"

struct WordDist {
    double dist;
    int word_id;
    __device__ WordDist(double dist, int id): dist(dist), word_id(id) {};
};

typedef graph<long, long, double, long, long, double> CSR;


__global__ void shortest_paths_kernel(int *source_words, int n, long *beg_pos, long *adj_list, double *weight, int vert_count, double *dist);

__global__ void collective_closest_kernel(double *dist, int num_source_words, int vert_count, int *word_ids, double *dists);

/**
 * Given list of reviewed words, learned words, graph, and number of words to recommend to review,
 * Return order to review learned words based on collctive closeness
 * to already reviewed words
 */
std::vector<int> review (CSR *csr, std::vector<int> &reviewed, std::vector<int> &learned, int rec_count);
