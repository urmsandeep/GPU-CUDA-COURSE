%%writefile hello.cu
#include <stdio.h>

// 
__global__ void helloKernel() {
    printf("Hello from thread %d!\n", threadIdx.x);
}

int main() {
    // Check if CUDA device is available
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    // Set larger printf buffer BEFORE launching kernel
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*1024*10);
    
    printf("Launching kernel from CPU...\n");
    
    helloKernel<<<1, 10>>>();
    
    // Check for launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(launchErr));
        return 1;
    }
    
    // Wait and check for execution errors
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(syncErr));
        return 1;
    }
    
    // Force flush - this is key!
    cudaDeviceReset();
    
    printf("GPU finished!\n");
    return 0;
}
