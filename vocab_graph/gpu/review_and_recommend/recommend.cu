#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include "review_and_recommend.h"
#include "error_handler.h"
#include "wtime.h"

using std::cout;
using std::endl;
using std::string;

std::vector<string> get_word_mapping(const char *mapping_file) {
	std::ifstream infile(mapping_file);
	std::vector<string> words;
	string line;
	while (std::getline(infile, line))
		words.push_back(line);
	return words;
}

int main(int argc, char **argv) {
	if (argc < 5) {
		cout << "Input: ./exe base_file mapping_file num_recs source_words..." << endl;
		return 1;
	}
	
	string base_filename (argv[1]);
	string beg_file = base_filename + "_beg_pos.bin";
	string csr_file = base_filename + "_csr.bin";
	string weight_file = base_filename + "_weight.bin";

	const char *mapping_file = argv[2];
	int num_recs = atoi(argv[3]);
	
	graph<long, long, double, long, long, double> *csr = 
		new graph <long, long, double, long, long, double>
		(beg_file.c_str(), csr_file.c_str(), weight_file.c_str());

	std::cout << "Edges: " << csr->edge_count << std::endl;
    std::cout << "Verticies: " << csr->vert_count << std::endl;

	int num_source_words = argc - 4;
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

    int *source_idxs_d;
    long *beg_pos_d, *csr_d;
    double *weight_d;
    double *dists_d;
    int *word_ids_d;
    int *num_recs_d;
    // Dist matrix
    double *dist_d;

    HANDLE_ERR(cudaMalloc((void **) &source_idxs_d, sizeof(int) * source_word_idxs.size()));
    HANDLE_ERR(cudaMalloc((void **) &beg_pos_d, sizeof(long) * (csr->vert_count + 1)));
    HANDLE_ERR(cudaMalloc((void **) &csr_d, sizeof(long) * csr->edge_count));
    HANDLE_ERR(cudaMalloc((void **) &weight_d, sizeof(double) * csr->edge_count));
    // Crate closest_words of size num_rec
    HANDLE_ERR(cudaMalloc((void **) &dists_d, sizeof(double) * csr->vert_count));
    HANDLE_ERR(cudaMalloc((void **) &word_ids_d, sizeof(int) * csr->vert_count));
    HANDLE_ERR(cudaMalloc((void **) &num_recs_d, sizeof(int)));
    HANDLE_ERR(cudaMalloc((void **) &dist_d, sizeof(double) * csr->vert_count * num_source_words));

    // Copy source word idxs to device arr
    HANDLE_ERR(cudaMemcpy (source_idxs_d, source_word_idxs.data(), sizeof(int) * source_word_idxs.size(), cudaMemcpyHostToDevice));
    // Copy csr beg_pos arry into device arry
    HANDLE_ERR(cudaMemcpy (beg_pos_d, csr->beg_pos, sizeof(long) * (csr->vert_count + 1), cudaMemcpyHostToDevice));
    // Copy csr csr arry into device arry
    HANDLE_ERR(cudaMemcpy (csr_d, csr->csr, sizeof(long) * csr->edge_count, cudaMemcpyHostToDevice));
    // Copy csr weight arry into device arry
    HANDLE_ERR(cudaMemcpy (weight_d, csr->weight, sizeof(double) * csr->edge_count, cudaMemcpyHostToDevice));
    // Modied number of recs depending on size calculated in kernel
    HANDLE_ERR(cudaMemcpy (num_recs_d, &num_recs, sizeof(int), cudaMemcpyHostToDevice));

    double starttime = wtime();
    shortest_paths_kernel <<< 1, 1 >>> (source_idxs_d, num_source_words, beg_pos_d, csr_d, weight_d, csr->vert_count, dist_d);
    cudaDeviceSynchronize();
    collective_closest_kernel <<< 128, 128 >>> (dist_d, num_source_words, csr->vert_count, word_ids_d, dists_d);
    cudaDeviceSynchronize();
    double endtime = wtime();

    // Copy back closest_words
    double *dists = (double *)malloc(sizeof(double*) * csr->vert_count);
    int *ids = (int *)malloc(sizeof(int*) * csr->vert_count);
    HANDLE_ERR(cudaMemcpy (dists, dists_d, sizeof(double) * csr->vert_count, cudaMemcpyDeviceToHost));
    HANDLE_ERR(cudaMemcpy (ids, word_ids_d, sizeof(int) * csr->vert_count, cudaMemcpyDeviceToHost));

    WordDist **wd = (WordDist**)malloc(sizeof(WordDist*) * csr->vert_count);
    printf("yes\n");
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
    cout << "Algo Time: " << endtime - starttime << endl;
	return 0;	
}
