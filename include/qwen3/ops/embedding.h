#pragma once

#include "qwen3/core/tensor.h"

namespace qwen3::ops
{
    // token_ids:
    //   dtype  = Int32
    //   device = CUDA
    //   shape  = [batch, sequence]，也支持一般的非标量形状
    //
    // embedding_weight:
    //   dtype  = BFloat16
    //   device = 与 token_ids 相同的 CUDA 设备
    //   shape  = [vocab_size, hidden_size]
    //
    // 返回：
    //   shape = token_ids.shape + [hidden_size]
    //   dtype = BFloat16
    //   device = CUDA
    [[nodiscard]]
    Tensor embedding_lookup(const Tensor &token_ids, const Tensor &embedding_weight);

}   // namespace qwen::ops  