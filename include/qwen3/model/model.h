#pragma once

#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cublas_handle.h"
#include "qwen3/model/config.h"
#include "qwen3/model/kv_cache.h"
#include "qwen3/model/weights.h"

#include <cstdint>

namespace qwen3
{
    // 输入：
    // token_ids [batch_size, sequence_length]，Int32，CUDA
    //
    // 输出：
    // final_hidden_states
    // [batch_size, sequence_length, hidden_size]，BF16，CUDA
    [[nodiscard]]
    Tensor qwen3_model_forward(const Tensor &token_ids, const Qwen3Weights &, const ModelConfig &config,
                                const Tensor &rope_cache, KvCache &kv_cache, std::int64_t start_position, cuda::CublasHandle &cublas_handle);
}