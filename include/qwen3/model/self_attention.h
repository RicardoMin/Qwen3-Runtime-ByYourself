#pragma once

#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cublas_handle.h"
#include "qwen3/model/kv_cache.h"
#include "qwen3/model/weights.h"

#include <cstdint>

namespace qwen3
{
    // 对应 PyTorch Qwen3Attention.forward。
    //
    // normalized_hidden_states:
    // [batch_size, sequence_length, hidden_size]
    //
    // 返回：
    // [batch_size, sequence_length, hidden_size]

    [[nodiscard]]
    Tensor qwen3_self_attention(const Tensor &normalizer_hidden_states, const Qwen3LayerWeights &weights,
                            const ModelConfig &config, const Tensor &rope_cache, KvCache &kv_cache,
                            std::int32_t layer_index, std::int64_t start_position, cuda::CublasHandle &cublas_handle);
}   //namespace qwen3