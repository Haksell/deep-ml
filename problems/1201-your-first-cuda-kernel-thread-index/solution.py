#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

std::vector<int> global_thread_indices(int n) {
    int bytes = sizeof(int) * n;
    int* arr_h = (int*)malloc(bytes);
    int* arr_d;

    cudaMalloc(&arr_d, bytes);
    int blockDim = 256;
    int gridDim = (blockDim + 255) / 256;
    index_kernel<<<gridDim, blockDim>>>(arr_d, n);
    cudaMemcpy(arr_h, arr_d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(arr_d);
    return std::vector(arr_h, arr_h + n);
}
