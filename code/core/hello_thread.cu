%%writefile hello.cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
