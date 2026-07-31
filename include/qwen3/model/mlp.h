#pragma once

#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cublas_handle.h"
#include "qwen3/model/config.h"
#include "qwen3/model/weights.h"

namespace qwen3
{
    // 输入必须已经经过 post_attention_layernorm。
    //
    // 输入：
    //   [batch_size, sequence_length, hidden_size]
    //
    // 输出：
    //   [batch_size, sequence_length, hidden_size]
    //
    // 本函数不执行 RMSNorm，也不执行残差相加。
    [[nodiscard]]
    Tensor qwen3_mlp(const Tensor &normalized_hidden_states, const Qwen3LayerWeights &weights,
                    const ModelConfig &config, cuda::CublasHandle &cublas_handle);
}