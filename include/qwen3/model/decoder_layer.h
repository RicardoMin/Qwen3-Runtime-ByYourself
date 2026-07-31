#pragma once

#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cublas_handle.h"
#include "qwen3/model/config.h"
#include "qwen3/model/kv_cache.h"
#include "qwen3/model/weights.h"

#include <cstdint>

namespace qwen3
{
    // hidden_status 的所有权被转移进函数
    //
    // 输入，输出
    // [batch_size, sequence_length, hidden_size]
    [[nodiscard]]
    Tensor qwen3_decoder_layer(Tensor hidden_states, const Qwen3LayerWeights &weights, const ModelConfig &config, const Tensor &rope_cache,
                                KvCache &kv_cache, std::int32_t layer_index, std::int64_t start_position, cuda::CublasHandle &cublas_handle);
}