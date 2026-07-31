#pragma once

#include "qwen3/core/tensor.h"

#include <cstdint>

namespace qwen3::ops
{
    // 生成形状为 [max_positions, head_dim / 2, 2] 的 FP32 RoPE 缓存。
    //
    // cache[position, pair, 0] = cos(angle)
    // cache[position, pair, 1] = sin(angle)
    //
    // 对于原版 Qwen3 Dense，这个缓存可以被所有 Transformer 层共享。
    [[nodiscard]]
    Tensor build_rope_cache(std::int64_t max_positions, std::int32_t head_dim,
                            float rope_theta, Device device);

    // 原地旋转 query 和 key。
    //
    // query: [batch_size, sequence_length,
    //         num_query_heads * head_dim]
    //
    // key:   [batch_size, sequence_length,
    //         num_key_value_heads * head_dim]
    //
    // start_position:
    //   当前这批 token 在整个序列中的起始位置。
    void apply_rope_inplace(
        Tensor &query, Tensor &key,
        const Tensor &rope_cache,
        std::int32_t num_query_heads,
        std::int32_t num_key_value_heads,
        std::int64_t start_position
    );
}   // namespace qwen3