#pragma once

#include "qwen3/core/device.h"
#include "qwen3/core/tensor.h"
#include "qwen3/io/safetensors_checkpoint.h"
#include "qwen3/model/config.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace qwen3
{
    // 一个 Transformer 层包含的全部权重
    struct Qwen3LayerWeights final
    {
        Tensor input_layernorm;

        Tensor q_proj;
        Tensor k_proj;
        Tensor v_proj;
        Tensor o_proj;

        Tensor q_norm;
        Tensor k_norm;

        Tensor post_attention_layernorm;

        Tensor gate_proj;
        Tensor up_proj;
        Tensor down_proj;
    };

    class Qwen3Weights final
    {
    public:
        [[nodiscard]]
        static Qwen3Weights load(SafetensorsCheckpoint &checkpoint, const ModelConfig &config, Device target_device);

        [[nodiscard]]
        const Tensor &token_embedding() const noexcept;

        [[nodiscard]]
        const Tensor &final_norm() const noexcept;

        [[nodiscard]]
        const Qwen3LayerWeights &layer(std::size_t layer_index) const;

        [[nodiscard]]
        std::size_t layer_count() const noexcept;

        // 不同模型之间的output权重不一样
        // 如果 config.tie_word_embeddings == true
        // 返回 token_embedding
        // 否则返回独立加载的 lm_head.weight
        [[nodiscard]]
        const Tensor &output_weight() const noexcept;

        [[nodiscard]]
        bool has_independent_output_weight() const noexcept;

    private:
        Qwen3Weights(Tensor token_embeddings, std::vector<Qwen3LayerWeights> layers, Tensor final_norm, std::optional<Tensor> lm_head);

    private:
        Tensor token_embedding_;
        std::vector<Qwen3LayerWeights> layers_;
        Tensor final_norm_;

        std::optional<Tensor> lm_head_;
    };
}   // namespace qwen3