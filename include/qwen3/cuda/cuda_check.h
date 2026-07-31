#pragma once

#include <cuda_runtime_api.h>

namespace qwen3::cuda
{
    void check(cudaError_t status, const char *expression, const char *file, int line);

}   // namespace qwen3::cuda

#define QWEN3_CUDA_CHECK(expression) ::qwen3::cuda::check((expression), #expression, __FILE__, __LINE__)