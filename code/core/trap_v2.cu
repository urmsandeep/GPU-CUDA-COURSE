__global__ void trap_kernel_v2(
    double a, double h, int n,
    double* block_sums) {
    
    __shared__ double cache[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Each thread computes its trapezoid
    double local_sum = 0.0;
    if (gid < n) {
        double x = a + gid * h;
        local_sum = f(x);
    }
    
    // Load into shared memory
    cache[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; 
         stride > 0; 
         stride /= 2) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes block result
    if (tid == 0) {
        block_sums[blockIdx.x] = cache[0];
    }
}

int main() {
    int n = 1000000;
    double a = 0.0, b = 10.0;
    double h = (b - a) / n;
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    double *d_block_sums;
    cudaMalloc(&d_block_sums, 
               blocks * sizeof(double));
    
    trap_kernel_v2<<<blocks, threads>>>(
        a, h, n, d_block_sums
    );
    
    // Copy back only block sums
    double *h_block_sums = 
        new double[blocks];
    cudaMemcpy(h_block_sums, d_block_sums,
               blocks * sizeof(double),
               cudaMemcpyDeviceToHost);
    
    // CPU reduces only ~4000 values
    double sum = 0.0;
    for (int i = 0; i < blocks; i++) {
        sum += h_block_sums[i];
    }
    
    // Add endpoint corrections
    sum += (f(a) + f(b)) / 2.0;
    sum -= (f(a + (n-1)*h));
    
    double result = h * sum;
    printf("Result: %.6f\n", result);
    
    delete[] h_block_sums;
    cudaFree(d_block_sums);
    return 0;
}
