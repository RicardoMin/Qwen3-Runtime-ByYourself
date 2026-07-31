#pragma once

#include "qwen3/core/tensor.h"

namespace qwen3::ops
{
    // 沿最后一个维度求最大值所在的索引。
    //
    // input:
    //   dtype  = BFloat16
    //   device = CUDA
    //   shape  = [..., reduction_size]
    //
    // keep_dimension == false:
    //   [B, V] -> [B]
    //
    // keep_dimension == true:
    //   [B, V] -> [B, 1]
    //
    // output:
    //   dtype  = Int32
    //   device = 与 input 相同
    [[nodiscard]]
    Tensor argmax_last_dimension(const Tensor &input, bool keep_dimension = false, std::int64_t valid_reduction_size = 0);
}   // namespace qwen3::ops