#include "qwen3/model/self_attention.h"

#include "qwen3/ops/gqa_attention.h"
#include "qwen3/ops/linear.h"
#include "qwen3/ops/rms_norm.h"
#include "qwen3/ops/rope.h"

#include <cstdint>
#include <stdexcept>

namespace qwen3
{
    namespace 
    {
        void validate_self_attention_inputs(const Tensor &hidden_states, const ModelConfig &config,
                                            const Tensor &rope_cache, const KvCache &kv_cache,
                                            std::int32_t layer_index, std::int64_t start_position)
        {
            if (hidden_states.dtype() !=
                DType::BFloat16)
            {
                throw std::invalid_argument(
                    "self attention input must be bfloat16");
            }

            if (!hidden_states.device().is_cuda())
            {
                throw std::invalid_argument(
                    "self attention input must be on CUDA");
            }

            if (hidden_states.shape().rank() != 3)
            {
                throw std::invalid_argument(
                    "self attention input must have rank 3");
            }

            if (hidden_states.shape()[0] <= 0 ||
                hidden_states.shape()[1] <= 0)
            {
                throw std::invalid_argument(
                    "self attention batch and sequence "
                    "dimensions must be positive");
            }

            if (hidden_states.shape()[2] !=
                config.hidden_size)
            {
                throw std::invalid_argument(
                    "self attention input hidden size "
                    "does not match ModelConfig");
            }

            if (layer_index < 0 ||
                layer_index >= config.num_hidden_layers ||
                layer_index >= kv_cache.num_layers())
            {
                throw std::out_of_range(
                    "self attention layer index is invalid");
            }

            if (start_position < 0)
            {
                throw std::invalid_argument(
                    "self attention start_position "
                    "cannot be negative");
            }

            if (kv_cache.cached_length(layer_index) !=
                start_position)
            {
                throw std::logic_error(
                    "self attention start_position must "
                    "equal the current KV cache length");
            }

            const std::int64_t batch_size =
                hidden_states.shape()[0];

            const std::int64_t sequence_length =
                hidden_states.shape()[1];

            if (batch_size >
                kv_cache.max_batch_size())
            {
                throw std::out_of_range(
                    "self attention batch size exceeds "
                    "KV cache capacity");
            }

            if (start_position >
                    kv_cache.max_sequence_length() ||
                sequence_length >
                    kv_cache.max_sequence_length() -
                        start_position)
            {
                throw std::out_of_range(
                    "self attention sequence exceeds "
                    "KV cache capacity");
            }

            if (kv_cache.num_key_value_heads() !=
                    config.num_key_value_heads ||
                kv_cache.head_dim() !=
                    config.head_dim)
            {
                throw std::invalid_argument(
                    "KV cache structure does not match "
                    "ModelConfig");
            }

            if (rope_cache.device() !=
                hidden_states.device())
            {
                throw std::invalid_argument(
                    "RoPE cache and hidden states must "
                    "be on the same device");
            }
        }
    }   // namespace

    Tensor qwen3_self_attention(const Tensor &normalized_hidden_states, const Qwen3LayerWeights &weights,
                                const ModelConfig &config, const Tensor &rope_cache, KvCache &kv_cache,
                                std::int32_t layer_index, std::int64_t start_position, cuda::CublasHandle &cublas_handle)
    {
        validate_self_attention_inputs(normalized_hidden_states, config, rope_cache, kv_cache, layer_index, start_position);

        // [B, S, hidden_size]
        //          x
        // [q_projection_size, hidden_size]
        //
        // ->[B, S, q_projection_size]
        Tensor query = ops::linear(normalized_hidden_states, weights.q_proj, cublas_handle);

        // -> [B, S, kv_projection_size]
        Tensor key = ops::linear(normalized_hidden_states, weights.k_proj, cublas_handle);

        // -> [B, S, kv_projection_size]
        Tensor value = ops::linear(normalized_hidden_states, weights.v_proj, cublas_handle);

        // q_norm.weight 和 k_norm.weight 都是 [head_dim]
        //
        // RMSNorm 会把 Q/K 的连续内存分别视为：
        //
        // Q: [B * S * num_query_heads, head_dim]
        // K: [B * S * num_kv_hads, head_dim]
        query = ops::rms_norm(query, weights.q_norm, config.rms_norm_eps);

        key = ops::rms_norm(key, weights.k_norm, config.rms_norm_eps);

        // 只旋转 Q/K，不旋转 V
        ops::apply_rope_inplace(query, key, rope_cache, config.num_attention_heads, config.num_key_value_heads, start_position);

        // 必须先把当前 K/V 写入缓存
        // 当前 token 才能再 attention 中关注自己
        kv_cache.append(layer_index, start_position, key, value);

        Tensor attention_context = ops::grouped_query_attention(query, kv_cache.key_view(layer_index), kv_cache.value_view(layer_index), config.num_attention_heads, start_position);

        // attention_context:
        // [B, S, num_query_heads * head_dim]
        //
        // o_proj.weight:
        // [hidden_size, num_query_heads * head_dim]
        // 
        // output:
        // [B, S, hidden_size]
        return ops::linear(attention_context, weights.o_proj, cublas_handle);
    }
}   // namespace qwen3