#include <limits>
#include "graph.h"

struct WordDist {
    double dist;
    int word_id;
    __device__ WordDist(double dist, int id): dist(dist), word_id(id) {};
};

typedef graph<long, long, double, long, long, double> CSR;

/**
 * Given souce words (known), graph, and number to recommend,
 * Recommend new words to learn based of their collective closeness
 * to aready known words
 */
__global__ void recommend_kernel (int *beg_pos, int *adj_list, double *weight, int *source_words, int num_source_words,
        int vert_count, WordDist **closest_words, int *num_recs);
/**
 * Given list of reviewed words, learned words, graph, and number of words to recommend to review,
 * Return order to review learned words based on collctive closeness
 * to already reviewed words
 */
std::vector<int> review (CSR *csr, std::vector<int> &reviewed, std::vector<int> &learned, int rec_count);
