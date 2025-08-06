#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

__device__ float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

__global__ void compute_gradients(const float* X, const float* y, const float* beta,
    float* grad, int n_samples, int n_features) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_features) return;

    float g = 0.0f;
    for (int i = 0; i < n_samples; ++i) {
        float dot = 0.0f;
        for (int j = 0; j < n_features; ++j) {
            dot += X[i * n_features + j] * beta[j];
        }
        float p = sigmoid(dot);
        g += (p - y[i]) * X[i * n_features + tid];
    }
    grad[tid] = g / n_samples;
}

void solve(const float* X_host, const float* y_host, float* beta_host, int n_samples, int n_features) {
    float* X_dev, * y_dev, * beta_dev, * grad_dev;

    cudaMalloc(&X_dev, n_samples * n_features * sizeof(float));
    cudaMalloc(&y_dev, n_samples * sizeof(float));
    cudaMalloc(&beta_dev, n_features * sizeof(float));
    cudaMalloc(&grad_dev, n_features * sizeof(float));

    cudaMemcpy(X_dev, X_host, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_dev, y_host, n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_dev, beta_host, n_features * sizeof(float), cudaMemcpyHostToDevice);

    const int iterations = 100;
    const float lr = 0.1f;

    for (int it = 0; it < iterations; ++it) {
        compute_gradients << <(n_features + 255) / 256, 256 >> > (
            X_dev, y_dev, beta_dev, grad_dev, n_samples, n_features);

        // Cập nhật beta
        std::vector<float> grad_host(n_features);
        cudaMemcpy(grad_host.data(), grad_dev, n_features * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < n_features; ++i) {
            beta_host[i] -= lr * grad_host[i];
        }

        cudaMemcpy(beta_dev, beta_host, n_features * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaFree(X_dev);
    cudaFree(y_dev);
    cudaFree(beta_dev);
    cudaFree(grad_dev);
}
