#include "qwen3/model/weights.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

namespace qwen3
{
    namespace
    {
        std::size_t to_size(std::int32_t value)
        {
            if (value < 0) {
                throw std::invalid_argument("model configuration value cannot be negative");
            }

            return static_cast<std::size_t>(value);
        }

        std::size_t checked_product(std::size_t lhs, std::size_t rhs)
        {
            if (lhs != 0 && rhs > static_cast<std::size_t>(-1) / lhs) {
                throw std::overflow_error("model weight element count overflow");
            }
            
            return lhs * rhs;
        }

        void append_weight(std::vector<std::string> &names, const SafetensorsCheckpoint &checkpoint, std::string name, std::size_t expected_size)
        {
            const SafetensorsTensorInfo &info = checkpoint.tensor_info(name);

            if (info.dtype != DType::BFloat16) {
                throw std::runtime_error("weight '" + name + "' is not BF16");
            }

            if (info.shape.numel() != expected_size) {
                throw std::runtime_error("weight '" + name + "' has an unexpected number of elements: expected " +
                                        std::to_string(expected_size) + ", got " + std::to_string(info.shape.numel()));
            }

            names.push_back(std::move(name));
        }

        std::string layer_prefix(std::size_t layer_index)
        {
            return "model.layers." + std::to_string(layer_index) + ".";
        }

    }   // namespace

    Qwen3Weights::Qwen3Weights(Tensor token_embedding, std::vector<Qwen3LayerWeights> layers, Tensor final_norm, std::optional<Tensor> lm_head)
        : token_embedding_(std::move(token_embedding))
        , layers_(std::move(layers))
        , final_norm_(std::move(final_norm))
        , lm_head_(std::move(lm_head))
    {
    }

    Qwen3Weights Qwen3Weights::load(SafetensorsCheckpoint &checkpoint, const ModelConfig &config, Device target_device)
    {
        config.validate();

        // 当前权重结构只支持官方 Qwen3 Dense 的无 bias 形式
        // 如果以后支持 attention bias，需要额外加载 q/k/v/o bias
        if (config.attention_bias) {
            throw std::invalid_argument("attention_bias=true is not supported");
        }

        const std::size_t vocab_size = to_size(config.vocab_size);
        const std::size_t hidden_size = to_size(config.hidden_size);
        const std::size_t layer_count = to_size(config.num_hidden_layers);
        const std::size_t intermediate_size = to_size(config.intermediate_size);
        const std::size_t head_dim = to_size(config.head_dim);
        const std::size_t q_projection_size = config.q_projection_size();
        const std::size_t kv_projection_size = config.kv_projection_size();

        std::vector<std::string> names;

        // 每层11个权重，再加embedding 和最终 norm
        names.reserve(2 + layer_count * 11 + (config.tie_word_embeddings ? 0 : 1));

        append_weight(names, checkpoint, "model.embed_tokens.weight", checked_product(vocab_size, hidden_size));

        for (std::size_t layer_index = 0; layer_index < layer_count; ++layer_index)
        {
            const std::string prefix = layer_prefix(layer_index);

            append_weight(names, checkpoint, prefix + "input_layernorm.weight", hidden_size);

            append_weight(names, checkpoint, prefix + "self_attn.q_proj.weight", checked_product(q_projection_size, hidden_size));

            append_weight(names, checkpoint, prefix + "self_attn.k_proj.weight", checked_product(kv_projection_size, hidden_size));

            append_weight(names, checkpoint, prefix + "self_attn.v_proj.weight", checked_product(kv_projection_size, hidden_size));

            append_weight(names, checkpoint, prefix + "self_attn.o_proj.weight", checked_product(q_projection_size, hidden_size));

            append_weight(names, checkpoint, prefix + "self_attn.q_norm.weight", head_dim);

            append_weight(names, checkpoint, prefix + "self_attn.k_norm.weight", head_dim);

            append_weight(names, checkpoint, prefix + "post_attention_layernorm.weight", hidden_size);

            append_weight(names, checkpoint, prefix + "mlp.gate_proj.weight", checked_product(intermediate_size, hidden_size));

            append_weight(names, checkpoint, prefix + "mlp.up_proj.weight", checked_product(intermediate_size, hidden_size));

            append_weight(names, checkpoint, prefix + "mlp.down_proj.weight", checked_product(intermediate_size, hidden_size));
        }

        append_weight(names, checkpoint, "model.norm.weight", hidden_size);

        if (!config.tie_word_embeddings) {
            append_weight(names, checkpoint, "lm_head.weight", checked_product(vocab_size, hidden_size));
        }

        // 这里会走 loader 模块的双缓冲批量上传
        std::vector<Tensor> loaded = checkpoint.load_many(names, target_device);

        const std::size_t expected_tensor_count = 2 + layer_count * 11 + (config.tie_word_embeddings ? 0 : 1);

        if (loaded.size() != expected_tensor_count) {
            throw std::logic_error("SafetensorsLoader returned an unexpected number of tensors");
        }

        std::size_t cursor = 0;

        Tensor token_embedding = std::move(loaded[cursor++]);
        
        std::vector<Qwen3LayerWeights> layers;
        layers.reserve(layer_count);

        for (std::size_t layer_index = 0; layer_index < layer_count; ++layer_index)
        {
            layers.push_back(
                Qwen3LayerWeights{
                    std::move(loaded[cursor++]),    // input_layernorm

                    std::move(loaded[cursor++]),    // q_proj
                    std::move(loaded[cursor++]),    // k_proj
                    std::move(loaded[cursor++]),    // v_proj
                    std::move(loaded[cursor++]),    // o_proj

                    std::move(loaded[cursor++]),    // q_norm
                    std::move(loaded[cursor++]),    // k_norm

                    std::move(loaded[cursor++]),    // post_attention_layernorm

                    std::move(loaded[cursor++]),    // gate_proj
                    std::move(loaded[cursor++]),    // up_proj
                    std::move(loaded[cursor++])     // down_proj
            });
        }

        Tensor final_norm = std::move(loaded[cursor++]);

        std::optional<Tensor> lm_head;

        if (!config.tie_word_embeddings) {
            lm_head.emplace(std::move(loaded[cursor++]));
        }

        if (cursor != loaded.size()) {
            throw std::logic_error("not all loaded tensors were consumed");
        }

        return Qwen3Weights{
            std::move(token_embedding),
            std::move(layers),
            std::move(final_norm),
            std::move(lm_head)
        };
    }

    const Tensor &Qwen3Weights::token_embedding() const noexcept
    {
        return token_embedding_;
    }

    const Tensor &Qwen3Weights::final_norm() const noexcept
    {
        return final_norm_;
    }

    const Qwen3LayerWeights &Qwen3Weights::layer(std::size_t layer_index) const
    {
        return layers_.at(layer_index);
    }

    std::size_t Qwen3Weights::layer_count() const noexcept
    {
        return layers_.size();
    }

    const Tensor &Qwen3Weights::output_weight() const noexcept
    {
        if (lm_head_.has_value()) {
            return *lm_head_;
        }
        return token_embedding_;
    }

    bool Qwen3Weights::has_independent_output_weight() const noexcept
    {
        return lm_head_.has_value();
    }
}   // namespace qwen3