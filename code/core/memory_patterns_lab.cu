#include <stdio.h>
#include <stdlib.h>

// Kernel 1: Print thread and warp info
__global__ void printThreadInfo() {
    int tid = blockIdx.x * blockDim.x 
            + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (tid < 64) {
        printf("Block %d | Thread %2d | "
               "Global %3d | Warp %d | Lane %2d\n",
               blockIdx.x, threadIdx.x, 
               tid, warp_id, lane_id);
    }
}

// Kernel 2: Sequential access (GOOD)
__global__ void sequentialAccess(
    int *input, int *output, int N) {
    
    int tid = blockIdx.x * blockDim.x 
            + threadIdx.x;
    
    if (tid < N) {
        output[tid] = input[tid] * 2;
    }
}

// Kernel 3: Strided access (BAD)
__global__ void stridedAccess(
    int *input, int *output, int N) {
    
    int tid = blockIdx.x * blockDim.x 
            + threadIdx.x;
    
    int stride = 32;
    int index = tid * stride;
    
    if (index < N) {
        output[index] = input[index] * 2;
    }
}

int main() {
    printf("=== LAB: Memory Access Patterns ===\n\n");
    
    // Part 1: Visualize structure
    printf("PART 1: Thread & Warp Info\n");
    printf("---------------------------\n");
    printThreadInfo<<<2, 64>>>();
    cudaDeviceSynchronize();
    
    printf("\n\n");
    
    // Part 2: Performance comparison
    int N = 1000000;
    size_t bytes = N * sizeof(int);
    
    // Allocate memory
    int *h_input = (int*)malloc(bytes);
    int *h_output = (int*)malloc(bytes);
    
    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }
    
    int *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes,
               cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // Test sequential access
    printf("PART 2: Performance Comparison\n");
    printf("-------------------------------\n");
    cudaMemset(d_output, 0, bytes);
    
    cudaEventRecord(start);
    sequentialAccess<<<blocks, threads>>>(
        d_input, d_output, N
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("Sequential Access: %.3f ms\n", ms);
    float seq_time = ms;
    
    // Verify
    cudaMemcpy(h_output, d_output, bytes,
               cudaMemcpyDeviceToHost);
    printf("  First 5: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    
    // Test strided access
    cudaMemset(d_output, 0, bytes);
    
    cudaEventRecord(start);
    stridedAccess<<<blocks, threads>>>(
        d_input, d_output, N
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("\nStrided Access: %.3f ms\n", ms);
    printf("  Slowdown: %.2fx\n", ms / seq_time);
    
    // Verify
    cudaMemcpy(h_output, d_output, bytes,
               cudaMemcpyDeviceToHost);
    printf("  First 5: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", h_output[i * 32]);
    }
    printf("\n");
    
    printf("\n=== Analysis ===\n");
    printf("Why is strided %.1fx slower?\n", 
           ms / seq_time);
    printf("• Sequential: Warp accesses "
           "consecutive memory\n");
    printf("  → 1 memory transaction per warp\n");
    printf("• Strided: Warp accesses scattered "
           "memory\n");
    printf("  → 32 memory transactions per warp\n");
    printf("  → Up to 32x slower!\n");
    
    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
