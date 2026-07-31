#include "qwen3/ops/rope.h"

#include "qwen3/cuda/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qwen3::ops
{
    namespace
    {
        constexpr int kThreadsPerBlock = 256;

        __global__ void apply_rope_kernel(
            __nv_bfloat16 *values,
            const float *rope_cache,
            std::int64_t batch_size,
            std::int64_t sequence_length,
            std::int32_t num_heads,
            std::int64_t head_dim,
            std::int64_t half_dim,
            std::int64_t start_position,
            std::int64_t total_pairs)
        {
            const std::int64_t first_index = 
                static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

            const std::int64_t grid_stride =
                static_cast<std::int64_t>(gridDim.x) * blockDim.x;

            for (std::int64_t linear_index = first_index;
                linear_index < total_pairs; linear_index += grid_stride)
            {
                std::int64_t remaining = linear_index;

                const std::int64_t pair_index = remaining % half_dim;
                remaining /= half_dim;

                const std::int64_t head_index = remaining % num_heads;
                remaining /= num_heads;

                const std::int64_t token_index = remaining % sequence_length;
                remaining /= sequence_length;

                const std::int64_t batch_index = remaining;

                // values 的逻辑布局为：
                // [batch, sequence, head, head_dim]
                const std::int64_t head_offset = 
                    ((batch_index * sequence_length + token_index) *
                    num_heads + head_index) * head_dim;
                
                const std::int64_t first_half_offset = head_offset + pair_index;

                const std::int64_t second_half_offset = head_offset + half_dim + pair_index;

                const std::int64_t position = start_position + token_index;

                const std::int64_t cache_offset = (position * half_dim + pair_index) * 2;

                const float cosine = rope_cache[cache_offset];

                const float sine = rope_cache[cache_offset + 1];

                // 必须先把两个原始值都读取出来
                // 否则原地写回第一个值后会破坏第二个值的计算
                const float first = __bfloat162float(values[first_half_offset]);

                const float second = __bfloat162float(values[second_half_offset]);

                const float rotated_first = first * cosine - second * sine;

                const float rotated_second = second * cosine + first * sine;

                values[first_half_offset] = __float2bfloat16_rn(rotated_first);

                values[second_half_offset] = __float2bfloat16_rn(rotated_second);
            }    
        }

        [[nodiscard]]
        std::int64_t checked_projection_size(std::int32_t num_heads, std::int64_t head_dim)
        {
            if (num_heads <= 0) {
                throw std::invalid_argument("RoPE head count must be positive");
            }

            if (head_dim > std::numeric_limits<std::int64_t>::max() / num_heads) {
                throw std::overflow_error("RoPE projection size overflow");
            }

            return static_cast<std::int64_t>(num_heads) * head_dim;
        }

        void validate_rope_inputs(const Tensor &query, const Tensor &key, const Tensor &rope_cache,
        std::int32_t num_query_heads, std::int32_t num_key_value_heads, std::int64_t start_position)
        {
            if (query.dtype() != DType::BFloat16 || key.dtype() != DType::BFloat16)
            {
                throw std::invalid_argument(
                    "RoPE query and key must use bfloat16");
            }

            if (rope_cache.dtype() != DType::Float32)
            {
                throw std::invalid_argument(
                    "RoPE cache must use float32");
            }

            if (!query.device().is_cuda() ||
                !key.device().is_cuda() ||
                !rope_cache.device().is_cuda())
            {
                throw std::invalid_argument(
                    "RoPE tensors must be on CUDA");
            }

            if (query.device() != key.device() ||
                query.device() != rope_cache.device())
            {
                throw std::invalid_argument(
                    "RoPE tensors must be on the same CUDA device");
            }

            if (query.shape().rank() != 3 ||
                key.shape().rank() != 3)
            {
                throw std::invalid_argument(
                    "RoPE query and key must have rank 3");
            }

            if (rope_cache.shape().rank() != 3 ||
                rope_cache.shape()[2] != 2)
            {
                throw std::invalid_argument(
                    "RoPE cache shape must be "
                    "[max_positions, head_dim / 2, 2]");
            }

            if (query.shape()[0] != key.shape()[0] ||
                query.shape()[1] != key.shape()[1])
            {
                throw std::invalid_argument(
                    "RoPE query and key batch/sequence "
                    "dimensions must match");
            }

            const std::int64_t half_dim =
                rope_cache.shape()[1];

            if (half_dim <= 0 ||
                half_dim >
                    std::numeric_limits<std::int64_t>::max() / 2)
            {
                throw std::invalid_argument(
                    "RoPE cache contains an invalid head dimension");
            }

            const std::int64_t head_dim = half_dim * 2;

            const std::int64_t expected_query_size =
                checked_projection_size(
                    num_query_heads, head_dim);

            const std::int64_t expected_key_size =
                checked_projection_size(
                    num_key_value_heads, head_dim);

            if (query.shape()[2] != expected_query_size)
            {
                throw std::invalid_argument(
                    "RoPE query projection size does not match "
                    "num_query_heads * head_dim");
            }

            if (key.shape()[2] != expected_key_size)
            {
                throw std::invalid_argument(
                    "RoPE key projection size does not match "
                    "num_key_value_heads * head_dim");
            }

            if (start_position < 0)
            {
                throw std::invalid_argument(
                    "RoPE start_position cannot be negative");
            }

            const std::int64_t sequence_length =
                query.shape()[1];

            const std::int64_t maximum_positions =
                rope_cache.shape()[0];

            // 使用减法检查，避免 start_position +
            // sequence_length 溢出。
            if (start_position > maximum_positions ||
                sequence_length >
                    maximum_positions - start_position)
            {
                throw std::out_of_range(
                    "RoPE position range exceeds the cache");
            }
        }

        [[nodiscard]]
        cudaError_t launch_rope(Tensor &tensor, const Tensor &rope_cache,
                                std::int32_t num_heads, std::int64_t start_position)
        {
            const std::int64_t batch_size = tensor.shape()[0];

            const std::int64_t sequence_length = tensor.shape()[1];

            const std::int64_t half_dim = rope_cache.shape()[1];

            const std::int64_t head_dim = half_dim * 2;

            const std::int64_t total_pairs = 
                batch_size * sequence_length * static_cast<std::int64_t>(num_heads) * half_dim;
            
            if (total_pairs == 0) {
                return cudaSuccess;
            }

            const std::int64_t wanted_blocks = (total_pairs + kThreadsPerBlock - 1) / kThreadsPerBlock;

            // 使用 grid-stride loop，因此不需要为每个元素都创建一个 block
            const int block_count = static_cast<int>(
                std::min<std::int64_t>(wanted_blocks, 65535)
            );

            apply_rope_kernel<<<block_count, kThreadsPerBlock>>>(
                static_cast<__nv_bfloat16 *>(tensor.data()),
                static_cast<const float *>(rope_cache.data()),
                batch_size, sequence_length, num_heads,
                head_dim, half_dim, start_position, total_pairs
            );

            return cudaGetLastError();
        }
    }   // namespace

    Tensor build_rope_cache(
        std::int64_t max_positions, std::int32_t head_dim,
        float rope_theta, Device device
    )
    {
        if (max_positions <= 0) {
            throw std::invalid_argument("RoPE max_positions must be positive");
        }

        if (head_dim <= 0 || head_dim % 2 != 0) {
            throw std::invalid_argument("RoPE head_dim must be a positive even number");
        }

        if (!std::isfinite(rope_theta) || rope_theta <= 0.0F) {
            throw std::invalid_argument("RoPE theta must be finite and positive");
        }

        const std::int64_t half_dim = head_dim / 2;

        Tensor host_cache = Tensor::empty(Shape{max_positions, half_dim, 2}, DType::Float32, Device::cpu());

        auto *cache_data = static_cast<float *>(host_cache.data());

        // 每个 pair 只计算一次逆频率
        std::vector<double> inverse_frequencies(static_cast<std::size_t>(half_dim));

        for (std::int64_t pair = 0; pair < half_dim; ++pair)
        {
            const double exponent = (2.0 * static_cast<double>(pair)) / static_cast<double>(head_dim);

            inverse_frequencies[static_cast<std::size_t>(pair)]
                = std::pow(static_cast<double>(rope_theta), - exponent);
        }

        for (std::int64_t position = 0; position < max_positions; ++position)
        {
            for (std::int64_t pair = 0; pair < half_dim; ++pair)
            {
                const double angle = static_cast<double>(position)
                    * inverse_frequencies[static_cast<std::size_t>(pair)];
                
                const std::size_t offset = static_cast<std::size_t>((position * half_dim + pair) * 2);

                cache_data[offset] = static_cast<float>(std::cos(angle));
                cache_data[offset + 1] = static_cast<float>(std::sin(angle));
            }
        }

        if (device.is_cpu()) {
            return host_cache;
        }
        
        return host_cache.copy_to(device);
    }

    void apply_rope_inplace(
        Tensor &query, Tensor &key,
        const Tensor &rope_cache, std::int32_t num_query_heads,
        std::int32_t num_key_value_heads, std::int64_t start_position
    )
    {
        validate_rope_inputs(query, key, rope_cache, 
            num_query_heads, num_key_value_heads, start_position);
        
        int previous_device = 0;

        QWEN3_CUDA_CHECK(cudaGetDevice(&previous_device));

        const int target_device = query.device().index();

        const bool changed_device = previous_device != target_device;

        if (changed_device) {
            QWEN3_CUDA_CHECK(cudaSetDevice(target_device));
        }

        cudaError_t operation_status =
            launch_rope(query, rope_cache, num_query_heads, start_position);

        if (operation_status == cudaSuccess)
        {
            operation_status = launch_rope(key, rope_cache, num_key_value_heads, start_position);
        }

        cudaError_t restore_status = cudaSuccess;

        if (changed_device) {
            restore_status = cudaSetDevice(previous_device);
        }

        QWEN3_CUDA_CHECK(operation_status);
        QWEN3_CUDA_CHECK(restore_status);
    }   //namespace qwen3::ops
}