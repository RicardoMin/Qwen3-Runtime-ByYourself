#include "qwen3/model/model.h"

#include "qwen3/core/dtype.h"
#include "qwen3/model/decoder_layer.h"
#include "qwen3/ops/embedding.h"
#include "qwen3/ops/rms_norm.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace qwen3
{
    namespace
    {
        void validate_model_forward_arguments(const Tensor &token_ids, const Qwen3Weights &weights, const ModelConfig &config,
                                            const Tensor &rope_cache, const KvCache &kv_cache, std::int64_t start_position)
        {
            if (token_ids.dtype() != DType::Int32)
            {
                throw std::invalid_argument(
                    "Qwen3 token IDs must have Int32 dtype");
            }

            if (!token_ids.device().is_cuda())
            {
                throw std::invalid_argument(
                    "Qwen3 token IDs must be on CUDA");
            }

            if (token_ids.shape().rank() != 2)
            {
                throw std::invalid_argument(
                    "Qwen3 token IDs must have shape "
                    "[batch_size, sequence_length]");
            }

            const std::int64_t batch_size =
                token_ids.shape()[0];

            const std::int64_t sequence_length =
                token_ids.shape()[1];

            if (batch_size <= 0)
            {
                throw std::invalid_argument(
                    "Qwen3 batch size must be positive");
            }

            if (sequence_length <= 0)
            {
                throw std::invalid_argument(
                    "Qwen3 sequence length must be positive");
            }

            if (start_position < 0)
            {
                throw std::invalid_argument(
                    "Qwen3 start position cannot be negative");
            }

            if (weights.layer_count() !=
                static_cast<std::size_t>(
                    config.num_hidden_layers))
            {
                throw std::invalid_argument(
                    "Qwen3 weight layer count does not "
                    "match ModelConfig");
            }

            if (rope_cache.shape().rank() != 3)
            {
                throw std::invalid_argument(
                    "Qwen3 RoPE cache must be rank 3");
            }

            const std::int64_t rope_capacity =
                rope_cache.shape()[0];

            if (start_position > rope_capacity ||
                sequence_length >
                    rope_capacity - start_position)
            {
                throw std::out_of_range(
                    "Qwen3 token positions exceed "
                    "the RoPE cache capacity");
            }

            // 每一层都有自己的 KV Cache。
            // 调用前所有层的长度都必须等于 start_position。
            for (std::int32_t layer_index = 0;
                 layer_index < config.num_hidden_layers;
                 ++layer_index)
            {
                if (kv_cache.cached_length(layer_index) !=
                    start_position)
                {
                    throw std::invalid_argument(
                        "Qwen3 KV cache length does not "
                        "match start_position");
                }
            }
        }
    }   // namespace

    Tensor qwen3_model_forward(const Tensor &token_ids, const Qwen3Weights &weights, const ModelConfig &config, const Tensor &rope_cache,
                                KvCache &kv_cache, std::int64_t start_position, cuda::CublasHandle &cublas_handle)
    {
        validate_model_forward_arguments(token_ids, weights, config, rope_cache, kv_cache, start_position);

        const std::int64_t sequence_length = token_ids.shape()[1];

        // ==========================================================================
        // 1. Token IDs -> Embedding
        // ==========================================================================

        Tensor hidden_states = ops::embedding_lookup(token_ids, weights.token_embedding());

        // ==========================================================================
        // 2.顺序执行全部 DecoderLayer
        // ==========================================================================

        for (std::int32_t layer_index = 0; layer_index < config.num_hidden_layers; ++layer_index)
        {
            // 所有层使用相同得 start_position
            // 绝对不能在层循环里递增
            hidden_states = qwen3_decoder_layer(std::move(hidden_states), weights.layer(static_cast<std::size_t>(layer_index)),
                config, rope_cache, kv_cache, layer_index, start_position, cublas_handle);
        }

        // 每层都应当追加相同数量得 K/V
        const std::int64_t expected_cached_length = start_position + sequence_length;

        for (std::int32_t layer_index = 0; layer_index < config.num_hidden_layers; ++layer_index)
        {
            if (kv_cache.cached_length(layer_index) != expected_cached_length) {
                throw std::logic_error("Qwen3 decoder layer did not advance its KV cache correctly");
            }
        }

        // ==========================================================================
        // 3.模型最重 RMSNorm
        // ==========================================================================

        return ops::rms_norm(hidden_states, weights.final_norm(), config.rms_norm_eps);
    }
}