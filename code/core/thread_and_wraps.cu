// Visualize how threads are organized into warps
__global__ void printThreadInfo() {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x 
            + threadIdx.x;
    
    // Calculate warp ID
    // (Hardware groups every 32 threads)
    int warp_id = tid / 32;
    
    // Position within warp (0-31)
    int lane_id = tid % 32;
    
    // Print info for first 2 warps only
    if (tid < 64) {
        printf("Block %d | Thread %d | "
               "Global ID: %d | "
               "Warp: %d | Lane: %d\n",
               blockIdx.x, 
               threadIdx.x,
               tid,
               warp_id,
               lane_id);
    }
}

int main() {
    printf("Launching 2 blocks × 64 threads\n");
    printf("Total: 128 threads = 4 warps\n\n");
    
    printThreadInfo<<<2, 64>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
