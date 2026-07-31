#pragma once

#include "qwen3/core/tensor.h"

#include <cstdint>
#include <random>

namespace qwen3::generation
{
    enum class SamplingMode
    {
        Greedy,
        Sample,
    };

    struct SamplingConfig final
    {
        SamplingMode mode = SamplingMode::Sample;

        float temperature = 0.6F;

        // 0 表示禁用 Top-k
        std::int32_t top_k = 20;

        // 1.0 表示禁用 Top-p
        float top_p = 0.95F;

        std::uint64_t seed = 42;

        // 允许参与采样的有效词表前缀
        // 0 表示使用 logits 完整的最后一维
        std::int64_t valid_vocab_size = 0;
    };

    class Sampler final
    {
    public:
        explicit Sampler(SamplingConfig config);

        // logits:
        //   dtype  = BFloat16
        //   device = CUDA
        //   shape  = [batch_size, vocab_size]
        //
        // 返回：
        //   dtype  = Int32
        //   device = 与 logits 相同
        //   shape  = [batch_size, 1]
        [[nodiscard]]
        Tensor sample(const Tensor &logits);

        // 重新设置随机数种子
        // 设置相同种子后可以重放采样序列
        void reseed(std::uint64_t seed);

    private:
        SamplingConfig config_;

        // 必须作为长期状态保存，不能每次 sample 都重新创建并使用同一个 seed
        std::mt19937_64 random_engine_;
    };

}   // namespace qwen3::gerenation