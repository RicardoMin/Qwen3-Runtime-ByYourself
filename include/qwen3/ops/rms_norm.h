#pragma once

#include "qwen3/core/tensor.h"

namespace qwen3::ops
{
    // 对 input 的最后一维执行 RMSNorm。
    //
    // input:
    //   dtype  = BFloat16
    //   device = CUDA
    //   shape  = [..., hidden_size]
    //
    // weight:
    //   dtype  = BFloat16
    //   device = 与 input 相同
    //   shape  = [hidden_size]
    //
    // output:
    //   dtype、device、shape 与 input 相同。
    [[nodiscard]]
    Tensor rms_norm(const Tensor &input, const Tensor &weight, float epsilon);
}   // namespace qwen3::ops