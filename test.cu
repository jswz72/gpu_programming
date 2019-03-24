#include <stdio.h>

int main(int argc, char **argv) {
    int nDevices;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Capabilities: %d.%d\n", prop.major, prop.minor);
    printf("Global mem: %lu\n", prop.totalGlobalMem / 1024 / 1024 / 1024);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    return 0;
}
