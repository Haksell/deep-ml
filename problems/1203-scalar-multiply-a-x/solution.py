#include <cuda_runtime.h>
#include <vector>

__global__ void scale_kernel(const float* x, float a, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i];
}

std::vector<float> scalar_multiply(const std::vector<float>& x_h, float a) {
    int n = x_h.size();
    int bytes = sizeof(float) * n;

    float *x_d, *y_d;
    cudaMalloc(&x_d, bytes);
    cudaMalloc(&y_d, bytes);
    cudaMemcpy(x_d, x_h.data(), bytes, cudaMemcpyHostToDevice);

    int blockDim = 256;
    int gridDim = (n + blockDim - 1) / blockDim;
    scale_kernel<<<gridDim, blockDim>>>(x_d, a, y_d, n);

    std::vector<float> result(n);
    cudaMemcpy(result.data(), y_d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(x_d);
    cudaFree(y_d);
    return result;
}
