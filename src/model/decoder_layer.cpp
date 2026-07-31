#include "qwen3/model/decoder_layer.h"

#include "qwen3/core/dtype.h"
#include "qwen3/model/mlp.h"
#include "qwen3/model/self_attention.h"
#include "qwen3/ops/elementwise.h"
#include "qwen3/ops/rms_norm.h"

#include <stdexcept>

namespace qwen3
{
    namespace
    {
        void validate_decoder_layer_arguments(const Tensor &hidden_states, const ModelConfig &config,
                                            std::int32_t layer_index, std::int64_t start_position)
        {
            if (hidden_states.dtype() != DType::BFloat16)
            {
                throw std::invalid_argument(
                    "DecoderLayer hidden states must "
                    "have BFloat16 dtype");
            }

            if (!hidden_states.device().is_cuda())
            {
                throw std::invalid_argument(
                    "DecoderLayer hidden states must "
                    "be on CUDA");
            }

            if (hidden_states.shape().rank() != 3)
            {
                throw std::invalid_argument(
                    "DecoderLayer hidden states must have "
                    "shape [batch, sequence, hidden_size]");
            }

            if (hidden_states.shape()[2] !=
                config.hidden_size)
            {
                throw std::invalid_argument(
                    "DecoderLayer hidden size does not "
                    "match ModelConfig");
            }

            if (layer_index < 0 ||
                layer_index >= config.num_hidden_layers)
            {
                throw std::out_of_range(
                    "DecoderLayer layer index is invalid");
            }

            if (start_position < 0)
            {
                throw std::invalid_argument(
                    "DecoderLayer start position "
                    "cannot be negative");
            }
        }
    }   // namespace

    Tensor qwen3_decoder_layer(Tensor hidden_states, const Qwen3LayerWeights &weights, const ModelConfig &config, const Tensor &rope_cache,
                                KvCache &kv_cache, std::int32_t layer_index, std::int64_t start_position, cuda::CublasHandle &cublas_handle)
    {
        validate_decoder_layer_arguments(hidden_states, config, layer_index, start_position);

        // ===================================================================================
        // 1. Self-Attention 分支
        // ===================================================================================
        {
            Tensor normalized_attention_input = ops::rms_norm(hidden_states, weights.input_layernorm, config.rms_norm_eps);

            Tensor attention_delta = qwen3_self_attention(normalized_attention_input, weights, config, rope_cache, kv_cache, layer_index, start_position, cublas_handle);

            // 第一次残差：
            // x = x + Attention(RMSNorm(x))
            ops::add_inplace(hidden_states, attention_delta);
        }

        // 上面的局部作用域结束后，
        // attention input 和 attention delta 会立即释放

        // ====================================================================================
        // 2. MLP 分支
        // ====================================================================================
        {
            Tensor normalized_mlp_input = ops::rms_norm(hidden_states, weights.post_attention_layernorm, config.rms_norm_eps);

            Tensor mlp_delta = qwen3_mlp(normalized_mlp_input, weights, config, cublas_handle);

            // 第二次残差
            // x = x + MLP(RMSNorm(x))
            ops::add_inplace(hidden_states, mlp_delta);
        }

        return hidden_states;
    }
}   // namespace qwen3