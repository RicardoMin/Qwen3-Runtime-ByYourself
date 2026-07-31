#include "qwen3/cuda/cuda_check.h"

#include <sstream>
#include <stdexcept>

namespace qwen3::cuda
{
    void check(cudaError_t status, const char *expression, const char *file, int line)
    {
        if (status == cudaSuccess) {
            return;
        }

        std::ostringstream message;

        message << "CUDA call failed: " << expression << '\n'
                << cudaGetErrorName(status) << ": "
                << cudaGetErrorString(status) << '\n'
                << "at " << file << ':' << line;

        throw std::runtime_error(message.str());
    }
}   // namespace qwen3:cuda