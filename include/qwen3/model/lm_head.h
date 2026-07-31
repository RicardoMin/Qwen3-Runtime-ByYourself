#pragma once

#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cublas_handle.h"

namespace qwen3
{
    // 输入：
    //   [batch_size, sequence_length, hidden_size]
    //
    // 输出：
    //   [batch_size, hidden_size]
    //
    // 只复制每个 batch 的最后一个 token。
    [[nodiscard]]
    Tensor select_last_token_hidden(const Tensor &hidden_states);

    // final_hidden_states:
    //   [batch_size, sequence_length, hidden_size]
    //
    // output_weight:
    //   [vocab_size, hidden_size]
    //
    // output:
    //   [batch_size, vocab_size]
    [[nodiscard]]
    Tensor qwen3_lm_head(const Tensor &final_hidden_states, const Tensor &output_weight, cuda::CublasHandle &cublas_handle);
}