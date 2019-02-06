SRC=$(wildcard *.cu)

default: $(SRC)
	nvcc -o $@ $^
