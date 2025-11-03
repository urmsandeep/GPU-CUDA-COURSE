%%writefile hello.cu
#include <stdio.h>

// This function runs on the GPU
__global__ void helloKernel() {
    printf("Hello from thread %d!\n", threadIdx.x);
}

int main() {
    // Launch 10 threads on the GPU
    printf("Launching kernel from CPU...\n");
    
    helloKernel<<<1, 10>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    printf("GPU finished!\n");
    return 0;
}
