__device__ double f(double x) {
    return x * x;
}

__global__ void trap_kernel_v1(
    double a, double h, int n,
    double* local_sums) {
    
    int tid = blockIdx.x * blockDim.x 
            + threadIdx.x;
    
    if (tid < n) {
        double x = a + tid * h;
        local_sums[tid] = f(x);
    }
}

int main() {
    int n = 1000000;
    double a = 0.0, b = 10.0;
    double h = (b - a) / n;
    
    double *d_sums;
    cudaMalloc(&d_sums, n * sizeof(double));
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    trap_kernel_v1<<<blocks, threads>>>(
        a, h, n, d_sums
    );
    
    // Copy results back
    double *h_sums = new double[n];
    cudaMemcpy(h_sums, d_sums, 
               n * sizeof(double),
               cudaMemcpyDeviceToHost);
    
    // CPU reduction (endpoints weighted)
    double sum = (f(a) + f(b)) / 2.0;
    for (int i = 1; i < n - 1; i++) {
        sum += h_sums[i];
    }
    
    double result = h * sum;
    printf("Result: %.6f\n", result);
    
    delete[] h_sums;
    cudaFree(d_sums);
    return 0;
