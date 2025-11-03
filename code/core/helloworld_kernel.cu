#include <stdio.h>

// __global__ = this runs on GPU
__global__ void helloKernel() {
    printf("Hello from thread %d!\n", threadIdx.x);
}

int main() {
    // Launch 10 threads on GPU
    helloKernel<<<1, 10>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    return 0;
}
