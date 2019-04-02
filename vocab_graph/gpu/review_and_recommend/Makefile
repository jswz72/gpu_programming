default: recommend_source review_and_recommend_source recommend

review_and_recommend_source:
	nvcc -std=c++11 -lineinfo -c review_and_recommend.cu

recommend_source:
	nvcc -std=c++11 -lineinfo -c recommend.cu

recommend:
	nvcc -o recommend.bin recommend.o review_and_recommend.o 



