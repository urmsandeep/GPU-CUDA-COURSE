// First kernel: compute trapezoids 
// and reduce within blocks
__global__ void trap_kernel_v3_step1(
    double a, double h, int n,
    double* block_sums) {
    
    __shared__ double cache[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    double local_sum = 0.0;
    
    // Grid-stride loop for large n
    for (int i = gid; i < n; 
         i += blockDim.x * gridDim.x) {
        double x = a + i * h;
        local_sum += f(x);
    }
    
    cache[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; 
         stride > 0; stride /= 2) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_sums[blockIdx.x] = cache[0];
    }
}

// Second kernel: reduce block sums
__global__ void reduce_sums(
    double* input, 
    double* output,
    int n) {
    
    __shared__ double cache[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    cache[tid] = (gid < n) ? input[gid] : 0.0;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; 
         stride > 0; stride /= 2) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = cache[0];
    }
}

int main() {
    int n = 1000000;
    double a = 0.0, b = 10.0;
    double h = (b - a) / n;
    
    int threads = 256;
    int blocks = 512;  // Fixed grid size
    
    double *d_sums1, *d_sums2;
    cudaMalloc(&d_sums1, blocks * sizeof(double));
    cudaMalloc(&d_sums2, sizeof(double));
    
    // Step 1: Compute and reduce
    trap_kernel_v3_step1<<<blocks, threads>>>(
        a, h, n, d_sums1
    );
    
    // Step 2: Final reduction
    reduce_sums<<<1, threads>>>(
        d_sums1, d_sums2, blocks
    );
    
    // Copy back ONE value
    double gpu_sum;
    cudaMemcpy(&gpu_sum, d_sums2, 
               sizeof(double),
               cudaMemcpyDeviceToHost);
    
    // Final adjustment on CPU
    gpu_sum += (f(a) + f(b)) / 2.0;
    gpu_sum -= f(a + (n-1) * h);
    
    double result = h * gpu_sum;
    printf("Result: %.10f\n", result);
    
    cudaFree(d_sums1);
    cudaFree(d_sums2);
    return 0;
}
