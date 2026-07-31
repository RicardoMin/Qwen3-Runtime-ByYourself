#pragma once

#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cublas_handle.h"
#include "qwen3/generation/sampler.h"
#include "qwen3/model/config.h"
#include "qwen3/model/kv_cache.h"
#include "qwen3/model/weights.h"

#include <cstdint>
#include <vector>

namespace qwen3::generation
{
    enum class StopReason
    {
        MaxNewTokens,
        StopToken,
    };

    struct GenerationOptions final
    {
        // 最多生成多少个新 token，不包含 prompt
        std::int64_t max_new_tokens = 128;

        // 例如 Qwen3 常用：
        // 151645：<|im_end|>
        // 151543: <|endoftext|>
        std::vector<std::int32_t> stop_token_ids;
    };

    struct GenerationResult final
    {
        // 这里只保存新生成的 token，不包含 prompt
        // 如果遇到 stop token，stop token 本身也会包含在结果中
        std::vector<std::int32_t> token_ids;

        StopReason stop_reason = StopReason::MaxNewTokens;
    };

    // 第一版只支持 batch_size == 1
    //
    // 要求传入的是一个空的 KV Cache，也就是每层 cached_Length 都为0
    [[nodiscard]]
    GenerationResult generate_token_ids(const Tensor &prompt_token_ids, const Qwen3Weights &weights, const ModelConfig &conifg,
    const Tensor &rope_cache, KvCache &kv_cache, cuda::CublasHandle &cublas_handle, Sampler &sampler, const GenerationOptions &options);
}