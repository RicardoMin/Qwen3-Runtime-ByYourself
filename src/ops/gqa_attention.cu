#include "qwen3/ops/gqa_attention.h"

#include "qwen3/cuda/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qwen3::ops
{
    namespace
    {
        constexpr int kThreadsPerBlock = 256;

        __global__ void grouped_query_attention_kernel(
            const __nv_bfloat16 *query, const __nv_bfloat16 *key_cache, const __nv_bfloat16 *value_cache,
            __nv_bfloat16 *output, std::int64_t batch_size, std::int64_t sequence_length,
            std::int64_t cached_length, std::int32_t num_query_heads, std::int32_t num_key_value_heads,
            std::int64_t head_dim, std::int64_t start_position, std::int64_t key_batch_stride,
            std::int64_t key_sequence_stride, std::int64_t key_head_stride, std::int64_t key_dimension_stride,
            std::int64_t value_batch_stride, std::int64_t value_sequence_stride, std::int64_t value_head_stride,
            std::int64_t value_dimension_stride, float scale
        )
        {
            const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);

            const std::int64_t rows_per_batch = sequence_length * num_query_heads;

            const std::int64_t batch_index = row / rows_per_batch;

            const std::int64_t remaining = row % rows_per_batch;

            const std::int64_t query_position = remaining / num_query_heads;

            const std::int32_t query_head = static_cast<std::int32_t>(remaining % num_query_heads);

            const std::int32_t query_heads_per_kv_head = num_query_heads / num_key_value_heads;

            const std::int32_t kv_head = query_head / query_heads_per_kv_head;

            const std::int64_t query_head_offset = ((batch_index * sequence_length + query_position) * num_query_heads + query_head) * head_dim;

            const int thread_index = threadIdx.x;

            // 当前实现一个线程负责一个输出维度
            // Qwen3 Dense 的 head_dim 是 128
            // 因此 256 个线程足够
            float query_value = 0.0F;
            float output_accumulator = 0.0F;

            if (thread_index < head_dim)
            {
                query_value = __bfloat162float(query[query_head_offset + thread_index]);
            }

            __shared__ float reduction[kThreadsPerBlock];

            __shared__ float running_maximum;
            __shared__ float normalizer;
            __shared__ float old_weight;
            __shared__ float new_weight;

            if (thread_index == 0) 
            {
                running_maximum = -CUDART_INF_F;
                normalizer = 0.0F;
                old_weight = 0.0F;
                new_weight = 0.0F;
            }

            __syncthreads();

            // 对于当前 chunk 中第 query_position 个 token：
            //
            // 全局位置 = start_position + query_position
            //
            // 它最多只能看到：
            // [0, start_position + query_position]
            const std::int64_t visible_key_count = start_position + query_position + 1;

            for (std::int64_t key_position = 0; key_position < visible_key_count; ++key_position)
            {
                float partial_dot = 0.0F;

                if (thread_index < head_dim)
                {
                    const std::int64_t key_offset = batch_index * key_batch_stride + 
                    key_position * key_sequence_stride +
                    static_cast<std::int64_t>(kv_head) * key_head_stride +
                    static_cast<std::int64_t>(thread_index) * key_dimension_stride;

                    const float key_value = __bfloat162float(key_cache[key_offset]);

                    partial_dot = query_value * key_value;
                }

                reduction[thread_index] = partial_dot;

                __syncthreads();

                // 一个 block 内归约得到 Q * K
                for (int offset = kThreadsPerBlock / 2; offset > 0; offset /= 2)
                {
                    if (thread_index < offset) {
                        reduction[thread_index] += reduction[thread_index + offset];
                    }

                    __syncthreads();
                }

                if (thread_index == 0) {
                    const float score = reduction[0] * scale;

                    const float new_maximum = fmaxf(running_maximum, score);

                    // 历史积累结果需要缩放到新的最大值
                    old_weight = expf(running_maximum - new_maximum);

                    // 当前 score 对应的权重
                    new_weight = expf(score - new_maximum);

                    normalizer = normalizer * old_weight + new_weight;

                    running_maximum = new_maximum;
                }

                __syncthreads();

                if (thread_index < head_dim) 
                {
                    const std::int64_t value_offset = batch_index * value_batch_stride +
                    key_position * value_sequence_stride +
                    static_cast<std::int64_t>(kv_head) * value_head_stride +
                    static_cast<std::int64_t>(thread_index) * value_dimension_stride;

                    const float current_value = __bfloat162float(value_cache[value_offset]);

                    output_accumulator = output_accumulator * old_weight + current_value * new_weight;
                }

                // 下一次循环开头的 __suncthreads()
                // 会保证本轮 accumulator 更新完成
            }

            if (thread_index < head_dim)
            {
                output[query_head_offset + thread_index] = 
                    __float2bfloat16_rn(output_accumulator / normalizer);
            }
        }

        void validate_attention_input(
            const Tensor &query, const TensorView &key_cache, const TensorView &value_cache,
            std::int32_t num_query_heads, std::int64_t start_position)
        {
            if (query.dtype() != DType::BFloat16 ||
                key_cache.dtype() != DType::BFloat16 ||
                value_cache.dtype() != DType::BFloat16)
            {
                throw std::invalid_argument(
                    "GQA currently requires bfloat16 tensors");
            }

            if (!query.device().is_cuda() ||
                !key_cache.device().is_cuda() ||
                !value_cache.device().is_cuda())
            {
                throw std::invalid_argument(
                    "GQA tensors must be on CUDA");
            }

            if (query.device() != key_cache.device() ||
                query.device() != value_cache.device())
            {
                throw std::invalid_argument(
                    "GQA tensors must be on the same CUDA device");
            }

            if (query.shape().rank() != 3)
            {
                throw std::invalid_argument(
                    "GQA query must have rank 3");
            }

            if (key_cache.shape().rank() != 4 ||
                value_cache.shape().rank() != 4)
            {
                throw std::invalid_argument(
                    "GQA cache views must have rank 4");
            }

            if (key_cache.shape() !=
                value_cache.shape())
            {
                throw std::invalid_argument(
                    "GQA key/value cache shapes must match");
            }

            const std::int64_t batch_size =
                query.shape()[0];

            const std::int64_t sequence_length =
                query.shape()[1];

            const std::int64_t cache_batch_size =
                key_cache.shape()[0];

            const std::int64_t cached_length =
                key_cache.shape()[1];

            const std::int64_t num_key_value_heads =
                key_cache.shape()[2];

            const std::int64_t head_dim =
                key_cache.shape()[3];

            if (batch_size <= 0 ||
                sequence_length <= 0)
            {
                throw std::invalid_argument(
                    "GQA batch and sequence dimensions "
                    "must be positive");
            }

            if (batch_size != cache_batch_size)
            {
                throw std::invalid_argument(
                    "GQA query/cache batch sizes must match");
            }

            if (num_query_heads <= 0 ||
                num_key_value_heads <= 0)
            {
                throw std::invalid_argument(
                    "GQA head counts must be positive");
            }

            if (num_query_heads %
                    num_key_value_heads !=
                0)
            {
                throw std::invalid_argument(
                    "GQA query heads must be divisible by "
                    "key/value heads");
            }

            if (head_dim <= 0 ||
                head_dim > kThreadsPerBlock)
            {
                throw std::invalid_argument(
                    "GQA head_dim must be between 1 and 256");
            }

            const std::int64_t expected_query_size =
                static_cast<std::int64_t>(
                    num_query_heads) *
                head_dim;

            if (query.shape()[2] !=
                expected_query_size)
            {
                throw std::invalid_argument(
                    "GQA query projection size is incorrect");
            }

            if (start_position < 0)
            {
                throw std::invalid_argument(
                    "GQA start_position cannot be negative");
            }

            // 调用 GQA 前，当前 K/V 应该已经 append。
            //
            // 因此：
            // cached_length = start_position + sequence_length
            if (start_position > cached_length ||
                sequence_length !=
                    cached_length -
                        start_position)
            {
                throw std::invalid_argument(
                    "GQA cache length must equal "
                    "start_position + sequence_length");
            }

            const std::int64_t total_rows =
                batch_size *
                sequence_length *
                num_query_heads;

            if (total_rows >
                std::numeric_limits<int>::max())
            {
                throw std::overflow_error(
                    "GQA CUDA grid is too large");
            }
        }
    }// namespace

    Tensor grouped_query_attention(
        const Tensor &query, const TensorView &key_cache, const TensorView &value_cache,
        std::int32_t num_query_heads, std::int64_t start_position
    )
    {
        validate_attention_input(query, key_cache, value_cache, num_query_heads, start_position);

        const std::int64_t batch_size = query.shape()[0];

        const std::int64_t sequence_length = query.shape()[1];

        const std::int64_t cached_length = key_cache.shape()[1];

        const auto num_key_value_heads = static_cast<std::int32_t>(key_cache.shape()[2]);
        
        const std::int64_t head_dim = key_cache.shape()[3];

        const std::int64_t total_rows = batch_size * sequence_length * num_query_heads;

        Tensor output = Tensor::empty(query.shape(), DType::BFloat16, query.device());

        const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

        int previous_device = 0;

        QWEN3_CUDA_CHECK(cudaGetDevice(&previous_device));

        const bool changed_device = previous_device != query.device().index();

        if (changed_device) {
            QWEN3_CUDA_CHECK(cudaSetDevice(query.device().index()));
        }

        grouped_query_attention_kernel<<<static_cast<int>(total_rows), kThreadsPerBlock>>>
        (static_cast<const __nv_bfloat16 *>(query.data()), static_cast<const __nv_bfloat16 *>(key_cache.data()),
        static_cast<const __nv_bfloat16 *>(value_cache.data()), static_cast<__nv_bfloat16 *>(output.data()),
        batch_size, sequence_length, cached_length, num_query_heads, num_key_value_heads, head_dim, start_position,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3), scale);

        const cudaError_t operation_status = cudaGetLastError();

        cudaError_t resotre_status = cudaSuccess;

        if (changed_device) {
            resotre_status = cudaSetDevice(previous_device);
        }

        QWEN3_CUDA_CHECK(operation_status);
        QWEN3_CUDA_CHECK(resotre_status);

        return output;
    }
}   // namespace qwen3::ops