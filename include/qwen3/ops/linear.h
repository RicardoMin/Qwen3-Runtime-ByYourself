#pragma once

#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cublas_handle.h"

namespace qwen3::ops
{
    // input:
    //   shape = [..., in_features]
    //
    // weight:
    //   shape = [out_features, in_features]
    //
    // output:
    //   shape = [..., out_features]
    //
    // 数学运算：
    //   output = input @ weight.T
    [[nodiscard]]
    Tensor linear(const Tensor &input, const Tensor &weight, cuda::CublasHandle &handle);
}   // namespace::qwen3::ops