#include "qwen3/ops/argmax.h"

#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/cuda/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qwen3::ops
{
    namespace
    {
        const int kThreadsPerBlock = 256;

        constexpr std::int64_t kMaximumBlockCount = 65535;

        constexpr std::int32_t kInvalidIndex = 2147483647;

        __device__ __forceinline__
        bool candidate_is_better(float candidate_value, std::int32_t candidate_index, float current_value, std::int32_t current_index)
        {
            return candidate_value > current_value || (candidate_value == current_value && candidate_index < current_index);
        }

        __global__
        void argmax_bfloat16_kernel(const __nv_bfloat16 *input, std::int32_t *output, std::int64_t row_count, std::int64_t row_stride, std::int64_t search_size)
        {
            __shared__ float shared_values[kThreadsPerBlock];

            __shared__ std::int32_t shared_indices[kThreadsPerBlock];

            // 一个 block 处理一行
            // block 数不足时继续处理后面的行
            for (std::int64_t row = static_cast<std::int64_t>(blockIdx.x); row < row_count; row += static_cast<std::int64_t>(gridDim.x))
            {
                float best_value = -CUDART_INF_F;

                std::int32_t best_index = kInvalidIndex;

                const std::int64_t row_offset = row * row_stride;

                // 每个线程负责这一行中的若干元素
                for (std::int64_t column = static_cast<std::int64_t>(threadIdx.x); column < search_size; column += static_cast<std::int64_t>(blockDim.x))
                {
                    float candidate_value = __bfloat162float(input[row_offset + column]);

                    // 正常模型输出不应该包含 NaN
                    // 为了让行为保持稳定，这里把 NaN 当作负无穷处理
                    if (candidate_value != candidate_value) {
                        candidate_value = -CUDART_INF_F;
                    }

                    const auto candidate_index = static_cast<std::int32_t>(column);

                    if (candidate_is_better(candidate_value, candidate_index, best_value, best_index)) {
                        best_value = candidate_value;
                        best_index = candidate_index;
                    }
                }

                shared_values[threadIdx.x] = best_value;

                shared_indices[threadIdx.x] = best_index;

                __syncthreads();

                // 256 -> 128 -> 64 -> ... -> 1
                for (int stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1)
                {
                    if (threadIdx.x < stride) {
                        const int other = threadIdx.x + stride;

                        if (candidate_is_better(shared_values[other], shared_indices[other], shared_values[threadIdx.x], shared_indices[threadIdx.x]))
                        {
                            shared_values[threadIdx.x] = shared_values[other];

                            shared_indices[threadIdx.x] = shared_indices[other];
                        }
                    }

                    __syncthreads();
                }

                if (threadIdx.x == 0) {
                    output[row] = shared_indices[0];
                }

                // 防止下一个 row 提前覆盖共享内存
                __syncthreads();
            }
        }

        void validate_argmax_input(const Tensor &input)
        {
            if (input.dtype() !=
                DType::BFloat16)
            {
                throw std::invalid_argument(
                    "argmax input must have "
                    "BFloat16 dtype");
            }

            if (!input.device().is_cuda())
            {
                throw std::invalid_argument(
                    "argmax input must be on CUDA");
            }

            if (input.shape().rank() == 0)
            {
                throw std::invalid_argument(
                    "argmax input cannot be scalar");
            }

            const std::size_t last_dimension =
                input.shape().rank() - 1;

            const std::int64_t reduction_size =
                input.shape()[last_dimension];

            if (reduction_size <= 0)
            {
                throw std::invalid_argument(
                    "argmax reduction dimension "
                    "must be positive");
            }

            if (reduction_size >
                static_cast<std::int64_t>(
                    std::numeric_limits<
                        std::int32_t>::max()))
            {
                throw std::invalid_argument(
                    "argmax reduction dimension "
                    "exceeds Int32 index range");
            }
        }

        Shape make_argmax_output_shape(const Shape &input_shape, bool keep_dimension)
        {
            std::vector<Shape::Dimension> output_dimensions = input_shape.dimensions();

            if (keep_dimension) {
                output_dimensions.back() = 1;
            }
            else {
                output_dimensions.pop_back();
            }

            return Shape{std::move(output_dimensions)};
        }
    }   // namespace

    Tensor argmax_last_dimension(const Tensor &input, bool keep_dimension, std::int64_t valid_reduction_size)
    {
        validate_argmax_input(input);

        const std::size_t last_dimension = input.shape().rank() - 1;

        const std::int64_t row_stride = input.shape()[last_dimension];

        if (valid_reduction_size < 0 || valid_reduction_size > row_stride) 
        {
            throw std::invalid_argument("valid reduction size is outsize the last dimension");
        }

        const std::int64_t search_size = valid_reduction_size == 0 ? row_stride : valid_reduction_size;

        const std::int64_t row_count = input.numel() / row_stride;

        Tensor output = Tensor::empty(make_argmax_output_shape(input.shape(), keep_dimension), DType::Int32, input.device());

        // 例如输入 shape=[0, V]
        if (row_count == 0) {
            return output;
        }

        int previous_device = 0;

        QWEN3_CUDA_CHECK(cudaGetDevice(&previous_device));

        const int target_device = input.device().index();

        const bool changed_device = previous_device != target_device;

        if (changed_device) {
            QWEN3_CUDA_CHECK(cudaSetDevice(target_device));
        }

        const int block_count = static_cast<int>(std::min(row_count, kMaximumBlockCount));
        
        argmax_bfloat16_kernel<<<block_count, kThreadsPerBlock>>>(static_cast<const __nv_bfloat16 *>(input.data()),
            static_cast<int32_t *>(output.data()), row_count, row_stride, search_size);

        const cudaError_t launch_status = cudaGetLastError();

        cudaError_t restore_status = cudaSuccess;

        if (changed_device) {
            restore_status = cudaSetDevice(previous_device);
        }

        QWEN3_CUDA_CHECK(launch_status);
        QWEN3_CUDA_CHECK(restore_status);

        return output;
    }
}