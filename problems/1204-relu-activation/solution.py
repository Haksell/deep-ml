#include <cuda_runtime.h>
#include <vector>

__global__ void relu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0 ? x[i] : 0;
}

std::vector<float> relu(const std::vector<float>& x_h) {
    int n = x_h.size();
    int bytes = sizeof(float) * n;

    float *x_d, *y_d;
    cudaMalloc(&x_d, bytes);
    cudaMalloc(&y_d, bytes);
    cudaMemcpy(x_d, x_h.data(), bytes, cudaMemcpyHostToDevice);

    int blockDim = 256;
    int gridDim = (n + blockDim - 1) / blockDim;
    relu_kernel<<<gridDim, blockDim>>>(x_d, y_d, n);

    std::vector<float> result(n);
    cudaMemcpy(result.data(), y_d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(x_d);
    cudaFree(y_d);
    return result;
}
