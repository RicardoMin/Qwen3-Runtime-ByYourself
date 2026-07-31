#include "qwen3/ops/rms_norm.h"

#include "qwen3/core/dtype.h"
#include "qwen3/cuda/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace qwen3::ops
{
    namespace
    {
        constexpr int kThreadsPerBlock = 256;
        constexpr int kWarpSize = 32;
        constexpr std::int64_t kMaximumBlockCount = 65535;

        __device__ float warp_reduce_sum(float value)
        {
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
            {
                value += __shfl_down_sync(0xffffffffU, value, offset);
            }

            return value;
        }

        __device__ float block_reduce_sum(float value)
        {
            // 最多支持 1024 个线程，即 32 个 warp
            __shared__ float warp_sums[32];

            const int lane_index = threadIdx.x % kWarpSize;

            const int warp_index = threadIdx.x / kWarpSize;

            value = warp_reduce_sum(value);

            // 每个 warp 的第一个线程保存该 warp 的结果
            if (lane_index == 0)
            {
                warp_sums[warp_index] = value;
            }

            __syncthreads();

            const int warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;

            float block_sum = 0.0F;

            // 只让第一个 warp 汇总所有 warp 的结果
            if (warp_index == 0)
            {
                if (lane_index < warp_count)
                {
                    block_sum = warp_sums[lane_index];
                }

                block_sum = warp_reduce_sum(block_sum);
            }

            return block_sum;
        }

        __global__ void rms_norm_bfloat16_kernel(
            const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
            __nv_bfloat16 *output, std::int64_t row_count,
            std::int64_t normalized_size, float epsilon)
        {
            __shared__ float inverse_rms;

            // 一个 block 处理一行
            // block 不够时通过 grid-stride 继续处理后面的行
            for (std::int64_t row = static_cast<std::int64_t>(blockIdx.x); row < row_count; row += static_cast<std::int64_t>(gridDim.x))
            {
                const std::int64_t row_offset = row * normalized_size;

                float local_square_sum = 0.0F;

                // 每个线程负责最后一维中的若干元素
                for (std::int64_t column = threadIdx.x; column < normalized_size; column += blockDim.x)
                {
                    const float value = __bfloat162float(input[row_offset + column]);

                    local_square_sum += value * value;
                }

                const float square_sum = block_reduce_sum(local_square_sum);

                if (threadIdx.x == 0)
                {
                    const float mean_square = square_sum / static_cast<float>(normalized_size);

                    inverse_rms = rsqrtf(mean_square + epsilon);
                }

                __syncthreads();

                for (std::int64_t column = threadIdx.x; column < normalized_size; column += blockDim.x)
                {
                    const float input_value = __bfloat162float(input[row_offset + column]);

                    // Hugging Face Qwen3 RMSNorm 的顺序是：
                    //
                    // 1. FP32 计算归一化
                    // 2. 转回输入 dtype（BF16）
                    // 3. 再乘 BF16 weight
                    //
                    // 因此这里保留这一轮 BF16 舍入
                    const __nv_bfloat16 normalized = __float2bfloat16_rn(input_value * inverse_rms);

                    const float normalized_value = __bfloat162float(normalized);

                    const float weight_value = __bfloat162float(weight[column]);

                    output[row_offset + column] = __float2bfloat16_rn(normalized_value * weight_value);
                }

                // 防止下次循环覆盖 inverse_rms 时，
                // 当前行还有线程尚未读取完
                __syncthreads();
            }
        }

        void validate_rms_norm_arguments(const Tensor &input, const Tensor &weight, float epsilon)
        {
            if (input.dtype() != DType::BFloat16)
            {
                throw std::invalid_argument("RMSNorm input must have BFloat16 dtype");
            }

            if (weight.dtype() != DType::BFloat16)
            {
                throw std::invalid_argument(
                    "RMSNorm weight must have "
                    "BFloat16 dtype");
            }

            if (!input.device().is_cuda() || !weight.device().is_cuda())
            {
                throw std::invalid_argument(
                    "RMSNorm currently requires "
                    "CUDA tensors");
            }

            if (input.device() != weight.device())
            {
                throw std::invalid_argument(
                    "RMSNorm input and weight must "
                    "be on the same CUDA device");
            }

            if (input.shape().rank() == 0)
            {
                throw std::invalid_argument(
                    "RMSNorm input cannot be scalar");
            }

            if (weight.shape().rank() != 1)
            {
                throw std::invalid_argument(
                    "RMSNorm weight must be "
                    "one-dimensional");
            }

            const std::int64_t normalized_size = weight.shape()[0];

            if (normalized_size <= 0)
            {
                throw std::invalid_argument(
                    "RMSNorm normalized size must be positive");
            }

            const std::int64_t packed_width =
                input.shape()[input.shape().rank() - 1];

            if (packed_width <= 0)
            {
                throw std::invalid_argument(
                    "RMSNorm input last dimension must be positive");
            }

            // 普通 hidden state：packed_width == normalized_size
            //
            // Q/K packed heads：
            // packed_width == num_heads * normalized_size
            //
            // 例如：
            // Q [B, S, 2048]，q_norm [128]
            // 2048 / 128 = 16 个 Q head
            if (packed_width % normalized_size != 0)
            {
                throw std::invalid_argument(
                    "RMSNorm input last dimension must be divisible "
                    "by the weight size");
            }

            if (input.numel() % normalized_size != 0)
            {
                throw std::invalid_argument(
                    "RMSNorm input element count must be divisible "
                    "by the weight size");
            }

            if (!std::isfinite(epsilon) || epsilon <= 0.0F)
            {
                throw std::invalid_argument(
                    "RMSNorm epsilon must be "
                    "finite and positive");
            }
        }
    } // namespace

    Tensor rms_norm(const Tensor &input, const Tensor &weight, float epsilon)
    {
        validate_rms_norm_arguments(input, weight, epsilon);

        const std::int64_t normalized_size = weight.shape()[0];

        const std::int64_t row_count = input.numel() / normalized_size;

        Tensor output = qwen3::Tensor::empty(input.shape(), input.dtype(), input.device());

        if (row_count == 0)
        {
            return output;
        }

        int previous_device = 0;

        QWEN3_CUDA_CHECK(cudaGetDevice(&previous_device));

        const int target_device = input.device().index();

        const bool changed_device = previous_device != target_device;

        if (changed_device)
        {
            QWEN3_CUDA_CHECK(cudaSetDevice(target_device));
        }

        const int block_count = static_cast<int>(std::min(row_count, kMaximumBlockCount));

        rms_norm_bfloat16_kernel<<<block_count, kThreadsPerBlock>>>(
            static_cast<const __nv_bfloat16 *>(input.data()),
            static_cast<const __nv_bfloat16 *>(weight.data()),
            static_cast<__nv_bfloat16 *>(output.data()),
            row_count, normalized_size, epsilon);

        const cudaError_t launch_status = cudaGetLastError();

        cudaError_t restore_status = cudaSuccess;

        if (changed_device)
        {
            restore_status = cudaSetDevice(previous_device);
        }

        QWEN3_CUDA_CHECK(launch_status);
        QWEN3_CUDA_CHECK(restore_status);

        return output;
    }
} // namespace qwen::ops