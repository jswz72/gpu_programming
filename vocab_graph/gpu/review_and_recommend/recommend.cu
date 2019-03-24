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
    int init_num_recs = num_recs;
	
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

    HANDLE_ERR(cudaMalloc((void **) &source_idxs_d, sizeof(int) * source_word_idxs.size()));
    HANDLE_ERR(cudaMalloc((void **) &beg_pos_d, sizeof(long) * csr->vert_count));
    HANDLE_ERR(cudaMalloc((void **) &csr_d, sizeof(long) * csr->edge_count));
    HANDLE_ERR(cudaMalloc((void **) &weight_d, sizeof(double) * csr->edge_count));
    // Crate closest_words of size num_rec
    HANDLE_ERR(cudaMalloc((void **) &dists_d, sizeof(double) * num_recs));
    HANDLE_ERR(cudaMalloc((void **) &word_ids_d, sizeof(int) * num_recs));
    HANDLE_ERR(cudaMalloc((void **) &num_recs_d, sizeof(int)));

    // Copy source word idxs to device arr
    HANDLE_ERR(cudaMemcpy (source_idxs_d, source_word_idxs.data(), sizeof(int) * source_word_idxs.size(), cudaMemcpyHostToDevice));
    // Copy csr beg_pos arry into device arry
    HANDLE_ERR(cudaMemcpy (beg_pos_d, csr->beg_pos, sizeof(long) * csr->vert_count, cudaMemcpyHostToDevice));
    // Copy csr csr arry into device arry
    HANDLE_ERR(cudaMemcpy (csr_d, csr->csr, sizeof(long) * csr->edge_count, cudaMemcpyHostToDevice));
    // Copy csr weight arry into device arry
    HANDLE_ERR(cudaMemcpy (weight_d, csr->weight, sizeof(double) * csr->edge_count, cudaMemcpyHostToDevice));
    // Modied number of recs depending on size calculated in kernel
    HANDLE_ERR(cudaMemcpy (num_recs_d, &num_recs, sizeof(int), cudaMemcpyHostToDevice));

    double starttime = wtime();
    recommend_kernel <<< 1, 1>>> (beg_pos_d, csr_d, weight_d, source_idxs_d, num_source_words, csr->vert_count, dists_d, word_ids_d, num_recs_d);
    cudaDeviceSynchronize();
    double endtime = wtime();
    // Copy back closest_words
    HANDLE_ERR(cudaMemcpy (&num_recs, num_recs_d, sizeof(int), cudaMemcpyDeviceToHost));
    double *dists = (double *)malloc(sizeof(double*) * num_recs);
    int *ids = (int *)malloc(sizeof(int*) * num_recs);
    HANDLE_ERR(cudaMemcpy (dists, dists_d, sizeof(double) * num_recs, cudaMemcpyDeviceToHost));
    HANDLE_ERR(cudaMemcpy (ids, word_ids_d, sizeof(int) * num_recs, cudaMemcpyDeviceToHost));

	cout << "\nLearning recommendations :" << endl;
	for (int i = 0; i < num_recs; i++) {
		cout << words[ids[i]] << " (Value: "
			<< dists[i] << ")" << endl;
	}
    if (num_recs < init_num_recs) {
        cout << "End" << endl;
    }
    cout << "Algo Time: " << endtime - starttime << endl;
	return 0;	
}
