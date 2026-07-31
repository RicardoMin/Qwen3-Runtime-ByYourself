#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void check_cuda(
    cudaError_t status,
    const char* expression,
    const char* file,
    int line
) {
    if (status == cudaSuccess) {
        return;
    }

    std::cerr
        << "CUDA error at " << file << ':' << line << '\n'
        << "call: " << expression << '\n'
        << "reason: " << cudaGetErrorString(status) << '\n';

    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

__global__ void vector_add(
    const float* lhs,
    const float* rhs,
    float* output,
    int size
) {
    const int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = lhs[index] + rhs[index];
    }
}

}  // namespace

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No CUDA device is visible.\n";
        return EXIT_FAILURE;
    }

    // CUDA_VISIBLE_DEVICES=6 后，物理 GPU 6 会映射为 cuda:0
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp device_property{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_property, 0));

    constexpr int element_count = 1024;
    constexpr int threads_per_block = 256;

    constexpr std::size_t bytes =
        element_count * sizeof(float);

    std::vector<float> lhs(element_count);
    std::vector<float> rhs(element_count);
    std::vector<float> output(element_count);

    for (int i = 0; i < element_count; ++i) {
        lhs[i] = static_cast<float>(i);
        rhs[i] = static_cast<float>(2 * i);
    }

    float* device_lhs = nullptr;
    float* device_rhs = nullptr;
    float* device_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&device_lhs), bytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&device_rhs), bytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&device_output), bytes));

    CUDA_CHECK(cudaMemcpy(
        device_lhs,
        lhs.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        device_rhs,
        rhs.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    const int block_count =
        (element_count + threads_per_block - 1)
        / threads_per_block;

    vector_add<<<block_count, threads_per_block>>>(
        device_lhs,
        device_rhs,
        device_output,
        element_count
    );

    // kernel 异步启动，需要分别检查启动错误和执行错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        output.data(),
        device_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < element_count; ++i) {
        const float expected = lhs[i] + rhs[i];

        if (std::abs(output[i] - expected) > 1e-6F) {
            std::cerr
                << "Verification failed at index " << i
                << ", expected " << expected
                << ", got " << output[i] << '\n';

            return EXIT_FAILURE;
        }
    }

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_rhs));
    CUDA_CHECK(cudaFree(device_lhs));

    std::cout << "CUDA smoke test passed.\n";
    std::cout << "GPU: " << device_property.name << '\n';
    std::cout
        << "Compute capability: "
        << device_property.major << '.'
        << device_property.minor << '\n';

    return EXIT_SUCCESS;
}
