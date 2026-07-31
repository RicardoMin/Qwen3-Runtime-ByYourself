#pragma once

#include "qwen3/core/tensor.h"

namespace qwen3::ops
{
    // destination += source
    //
    // 两个 Tensor 必须：
    // 1. 都是 CUDA BF16
    // 2. shape 完全相同
    // 3. 位于同一张 GPU
    void add_inplace(Tensor &destination, const Tensor &source);

    // output = silu(gate) * up
    //
    // SiLU(x) = x / (1 + exp(-x))
    //
    // gate 和 up 必须具有相同的 shape、dtype 和 device。
    [[nodiscard]]
    Tensor silu_multiply(const Tensor &gate, const Tensor &up);
}   // namespace qwen3::ops