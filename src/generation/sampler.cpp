#include "qwen3/generation/sampler.h"

#include "qwen3/core/device.h"
#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/ops/argmax.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qwen3::generation
{
    namespace
    {
        struct Candidate final
        {
            double score = 0.0;
            std::int32_t token_id = 0;
        };

        bool candidate_is_better(const Candidate &lhs, const Candidate &rhs)
        {
            if (lhs.score == rhs.score) {
                return lhs.token_id < rhs.token_id;
            }
            return lhs.score > rhs.score;
        }

        float bfloat16_to_float(std::uint16_t value)
        {
            const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;

            float result = 0.0F;

            std::memcpy(&result, &bits, sizeof(result));

            return result;
        }

        double next_uniform(std::mt19937_64 &random_engine)
        {
            // 取随机数的高 53 位，映射到 [0, 1)
            //
            // 不使用 uniform_real_distribution
            // 避免不同标准库实现造成序列差异
            constexpr double scale = 1.0 / 9007199254740992.0;

            return static_cast<double>(random_engine() >> 11U) * scale;
        }

        void validate_sampling_config(const SamplingConfig &config)
        {
              switch (config.mode)
            {
            case SamplingMode::Greedy:
                return;

            case SamplingMode::Sample:
                break;

            default:
                throw std::invalid_argument(
                    "unknown sampling mode");
            }

            if (!std::isfinite(
                    config.temperature) ||
                config.temperature <= 0.0F)
            {
                throw std::invalid_argument(
                    "sampling temperature must "
                    "be finite and positive");
            }

            if (config.top_k < 0)
            {
                throw std::invalid_argument(
                    "sampling top_k cannot "
                    "be negative");
            }

            if (!std::isfinite(config.top_p) ||
                config.top_p <= 0.0F ||
                config.top_p > 1.0F)
            {
                throw std::invalid_argument(
                    "sampling top_p must be in "
                    "the interval (0, 1]");
            }
        }

        void validate_logits(const Tensor &logits)
        {
            if (logits.dtype() !=
                DType::BFloat16)
            {
                throw std::invalid_argument(
                    "sampler logits must have "
                    "BFloat16 dtype");
            }

            if (!logits.device().is_cuda())
            {
                throw std::invalid_argument(
                    "sampler logits must be "
                    "on CUDA");
            }

            if (logits.shape().rank() != 2)
            {
                throw std::invalid_argument(
                    "sampler logits must have "
                    "shape [batch_size, "
                    "vocab_size]");
            }

            if (logits.shape()[0] <= 0)
            {
                throw std::invalid_argument(
                    "sampler batch size must "
                    "be positive");
            }

            if (logits.shape()[1] <= 0)
            {
                throw std::invalid_argument(
                    "sampler vocabulary size "
                    "must be positive");
            }
        }

        std::int32_t smaple_one_row(const std::uint16_t *row_logits, std::int64_t vocab_size, const SamplingConfig &config, std::mt19937_64 &random_engine)
        {
            std::vector<Candidate> candidates;

            candidates.reserve(static_cast<std::size_t>(vocab_size));

            // 1. Temperature
            for (std::int64_t token_id = 0; token_id < vocab_size; ++token_id)
            {
                const float logit = bfloat16_to_float(row_logits[token_id]);

                // NaN 或 Inf 通常意味着上游算子窜在问题
                // 不在采样器里静默掩盖
                if (!std::isfinite(logit)) {
                    throw std::runtime_error("temperature scaling produced a non-finite score");
                }

                const double score = static_cast<double>(logit) / static_cast<double>(config.temperature);

                candidates.push_back(Candidate{score, static_cast<std::int32_t>(token_id)});
            }

            // 2. Top-k
            //
            // Hugging Face 的 Top-k 会保留所有 score >= 第 k 大 score 的 token
            //
            // 因此如果第 k 名存在并列，最终保留数量可能略大于k
            if (config.top_k > 0)
            {
                const std::size_t requested_top_k = std::min(static_cast<std::size_t>(config.top_k), candidates.size());

                if (requested_top_k < candidates.size()) 
                {
                    auto kth_position = candidates.begin() + static_cast<std::ptrdiff_t>(requested_top_k - 1);

                    std::nth_element(candidates.begin(), kth_position, candidates.end(), candidate_is_better);

                    const double kth_score = kth_position->score;

                    const auto retained_end = std::remove_if(candidates.begin(), candidates.end(), [kth_score](const Candidate &candidate) {
                        return candidate.score < kth_score;
                    });

                    candidates.erase(retained_end, candidates.end());

                }
            }

            // Top-需要从高到低累加概率
            std::sort(candidates.begin(), candidates.end(), candidate_is_better);

            // 3. 稳定 Softmax
            //
            // score 已经除过 temperature
            const double maximum_score = candidates.front().score;

            std::vector<double> weights(candidates.size());

            double total_weight = 0.0;

            for (std::size_t index = 0; index < candidates.size(); ++index)
            {
                const double weight = std::exp(candidates[index].score - maximum_score);

                weights[index] = weight;
                total_weight += weight;
            }

            if (!std::isfinite(total_weight) || total_weight <= 0.0) {
                throw std::runtime_error("sampler softmax weight sum is invalid");
            }

            // 4.Top-p
            //
            // 找到累计权重第一次达到 top_p * total_weight 的位置
            const double top_p_threshold = static_cast<double>(config.top_p) * total_weight;

            std::size_t retained_count = candidates.size();

            double retained_weight = 0.0;

            for (std::size_t index = 0; index < candidates.size(); ++index)
            {
                retained_weight += weights[index];

                if (retained_weight >= top_p_threshold) {
                    retained_count = index + 1;
                    break;
                }
            }

            // 5. 按保留候选的权重随机抽取
            //
            // 直接再 [0. retained_weight] 中抽样
            // 等价于先把保留概率重新归一化
            const double random_value = next_uniform(random_engine) * retained_weight;

            double cumulative_weight = 0.0;

            for (std::size_t index = 0; index < retained_count; ++index)
            {
                cumulative_weight += weights[index];

                if (random_value < cumulative_weight) {
                    return candidates[index].token_id;
                }
            }

            // 处理最后可能出现的浮点舍入边界
            return candidates[retained_count - 1].token_id;
        }

        std::int64_t resolve_valid_vocab_size(const Tensor &logits, const SamplingConfig &config)
        {
            const std::int64_t logits_vocab_size = logits.shape()[1];

            if (config.valid_vocab_size < 0) {
                throw std::invalid_argument("valid_vocab_size cannot be negative");
            }

            if (config.valid_vocab_size == 0) {
                return logits_vocab_size;
            }

            if (config.valid_vocab_size > logits_vocab_size) {
                throw std::invalid_argument("valid_vocab_size exceeds logits vocabulary size");
            }

            return config.valid_vocab_size;
        }

    }   // namespace

    Sampler::Sampler(SamplingConfig config)
        : config_(config)
        , random_engine_(config.seed)
    {
        validate_sampling_config(config_);
    }

    Tensor Sampler::sample(const Tensor &logits)
    {
        validate_logits(logits);

        const std::int64_t batch_size = logits.shape()[0];

        // 物理行跨度仍然是模型的完整 vocab_size
        const std::int64_t logits_vocab_size = logits.shape()[1];

        // 真正允许参与选择的 token 数量
        const std::int64_t valid_vocab_size = resolve_valid_vocab_size(logits, config_);

        // Greedy 完全留在 GPU
        if (config_.mode == SamplingMode::Greedy) {
            return ops::argmax_last_dimension(logits, true, valid_vocab_size);
        }

        // correctness-first
        // 先将 BF16 Logits 同步复制到 CPU
        Tensor cpu_logits = logits.copy_to(Device::cpu());

        Tensor cpu_token_ids = Tensor::empty(Shape{batch_size, 1}, DType::Int32, Device::cpu());

        const auto *logit_values = static_cast<const std::uint16_t *>(cpu_logits.data());

        auto *token_values = static_cast<std::int32_t *>(cpu_token_ids.data());

        for (std::int64_t batch = 0; batch < batch_size; ++batch)
        {                                                               // 行偏移必须使用物理词表大小
            token_values[batch] = smaple_one_row(logit_values + batch * logits_vocab_size, 
            // 只遍历有效 token
                valid_vocab_size, config_, random_engine_);
        }

        // 返回 CUDA[B, 1] 可以直接作为下一轮 model_forward 的 token_ids
        return cpu_token_ids.copy_to(logits.device());
    }

    void Sampler::reseed(std::uint64_t seed)
    {
        config_.seed = seed;
        random_engine_.seed(seed);
    }
}   // namespace qwen3::generation