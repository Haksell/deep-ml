#include <cuda_runtime.h>
#include <vector>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

std::vector<float> vector_add(const std::vector<float>& a_h, const std::vector<float>& b_h) {
    int n = a_h.size();
    int bytes = sizeof(float) * n;

    float *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, bytes);
    cudaMalloc(&b_d, bytes);
    cudaMalloc(&c_d, bytes);

    cudaMemcpy(a_d, a_h.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), bytes, cudaMemcpyHostToDevice);

    int blockDim = 256;
    int gridDim = (n + blockDim - 1) / blockDim;
    add_kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, n);

    std::vector<float> result(n);
    cudaMemcpy(result.data(), c_d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return result;
}
