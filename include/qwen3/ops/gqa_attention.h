#pragma once

#include "qwen3/core/tensor.h"
#include "qwen3/core/tensor_view.h"

#include <cstdint>

namespace qwen3::ops
{
    // query:
    // [batch_size, sequence_length,
    //  num_query_heads * head_dim]
    //
    // key_cache/value_cache:
    // [batch_size, cached_length,
    //  num_key_value_heads, head_dim]
    //
    // 返回：
    // [batch_size, sequence_length,
    //  num_query_heads * head_dim]
    [[nodiscard]]
    Tensor grouped_query_attention(
        const Tensor &query,
        const TensorView &key_cache,
        const TensorView &value_cache,
        std::int32_t num_query_heads,
        std::int64_t start_position
    );
}   // namespace qwen3::ops