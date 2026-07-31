#include "qwen3/generation/generator.h"

#include "qwen3/core/device.h"
#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/model/lm_head.h"
#include "qwen3/model/model.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace qwen3::generation
{
    namespace
    {
        bool contains_stop_token(const std::vector<std::int32_t> &stop_token_ids, std::int32_t token_id)
        {
            return std::find(stop_token_ids.begin(), stop_token_ids.end(), token_id) != stop_token_ids.end();
        }

        std::int64_t common_cached_length(const KvCache &kv_cache, std::int32_t num_layers)
        {
            if (num_layers <= 0) {
                throw std::logic_error("Qwen3 model must contain at least one layer");
            }

            const std::int64_t length = kv_cache.cached_length(0);

            for (std::int32_t layer_index = 1; layer_index < num_layers; ++layer_index)
            {
                if (kv_cache.cached_length(layer_index) != length) {
                    throw std::logic_error("Qwen3 KV cache lengths differ between layers");
                }
            }

            return length;
        }

        void validate_generation_arguments(const Tensor &prompt_token_ids, const Qwen3Weights &weights, const ModelConfig &config,
                                            const Tensor &rope_cache, const KvCache &kv_cache, const GenerationOptions &options)
        {
            config.validate();

            if (options.max_new_tokens < 0)
            {
                throw std::invalid_argument(
                    "max_new_tokens cannot be negative");
            }

            if (prompt_token_ids.dtype() != DType::Int32)
            {
                throw std::invalid_argument(
                    "prompt token IDs must have Int32 dtype");
            }

            if (!prompt_token_ids.device().is_cuda())
            {
                throw std::invalid_argument(
                    "prompt token IDs must be on CUDA");
            }

            if (prompt_token_ids.shape().rank() != 2)
            {
                throw std::invalid_argument(
                    "prompt token IDs must have shape "
                    "[batch_size, sequence_length]");
            }

            const std::int64_t batch_size =
                prompt_token_ids.shape()[0];

            const std::int64_t prompt_length =
                prompt_token_ids.shape()[1];

            // 当前版本只做单序列生成。
            // batch 内不同序列可能在不同时间遇到 EOS，
            // 需要 finished mask，后面再实现。
            if (batch_size != 1)
            {
                throw std::invalid_argument(
                    "the current Generator only supports batch_size == 1");
            }

            if (prompt_length <= 0)
            {
                throw std::invalid_argument(
                    "prompt cannot be empty");
            }

            if (weights.layer_count() !=
                static_cast<std::size_t>(
                    config.num_hidden_layers))
            {
                throw std::invalid_argument(
                    "weight layer count does not match ModelConfig");
            }

            if (kv_cache.num_layers() !=
                config.num_hidden_layers)
            {
                throw std::invalid_argument(
                    "KV cache layer count does not match ModelConfig");
            }

            if (kv_cache.num_key_value_heads() !=
                config.num_key_value_heads)
            {
                throw std::invalid_argument(
                    "KV cache head count does not match ModelConfig");
            }

            if (kv_cache.head_dim() != config.head_dim)
            {
                throw std::invalid_argument(
                    "KV cache head dimension does not match ModelConfig");
            }

            if (kv_cache.max_batch_size() < 1)
            {
                throw std::invalid_argument(
                    "KV cache batch capacity is too small");
            }

            if (rope_cache.shape().rank() != 3)
            {
                throw std::invalid_argument(
                    "RoPE cache must be rank 3");
            }

            if (rope_cache.dtype() != DType::Float32)
            {
                throw std::invalid_argument(
                    "RoPE cache must have Float32 dtype");
            }

            const Device &device =
                prompt_token_ids.device();

            if (rope_cache.device() != device ||
                weights.token_embedding().device() != device ||
                weights.output_weight().device() != device ||
                kv_cache.key_storage().device() != device ||
                kv_cache.value_storage().device() != device)
            {
                throw std::invalid_argument(
                    "all generation tensors must be on the same CUDA device");
            }

            // generate_token_ids 表示开始一个新序列。
            // 防止不小心把旧会话的 KV Cache 带进来。
            for (std::int32_t layer_index = 0;
                 layer_index < config.num_hidden_layers;
                 ++layer_index)
            {
                if (kv_cache.cached_length(layer_index) != 0)
                {
                    throw std::invalid_argument(
                        "KV cache must be empty before generation");
                }
            }

            const std::int64_t rope_capacity =
                rope_cache.shape()[0];

            const std::int64_t runtime_capacity =
                std::min(
                    rope_capacity,
                    kv_cache.max_sequence_length());

            if (prompt_length > runtime_capacity)
            {
                throw std::length_error(
                    "prompt exceeds the runtime sequence capacity");
            }

            // 使用减法，避免 prompt_length + max_new_tokens 溢出。
            if (options.max_new_tokens >
                runtime_capacity - prompt_length)
            {
                throw std::length_error(
                    "prompt plus generated tokens exceeds "
                    "the runtime sequence capacity");
            }

            for (const std::int32_t stop_token_id :
                 options.stop_token_ids)
            {
                if (stop_token_id < 0 ||
                    stop_token_id >= config.vocab_size)
                {
                    throw std::invalid_argument(
                        "stop token ID is outside the vocabulary");
                }
            }

            // 防止 embedding_lookup 越界访问。
            Tensor prompt_cpu =
                prompt_token_ids.copy_to(Device::cpu());

            const auto *token_ids =
                static_cast<const std::int32_t *>(
                    prompt_cpu.data());

            for (std::int64_t index = 0;
                 index < prompt_cpu.numel();
                 ++index)
            {
                if (token_ids[index] < 0 ||
                    token_ids[index] >= config.vocab_size)
                {
                    throw std::invalid_argument(
                        "prompt contains a token ID outside "
                        "the vocabulary");
                }
            }
        }

        void validate_sampled_token(const Tensor &token_ids, const Device &expected_device)
        {
             if (token_ids.dtype() != DType::Int32)
            {
                throw std::logic_error(
                    "Sampler must return Int32 token IDs");
            }

            if (token_ids.shape() != Shape{1, 1})
            {
                throw std::logic_error(
                    "Sampler must return token IDs with shape [1, 1]");
            }

            if (token_ids.device() != expected_device)
            {
                throw std::logic_error(
                    "Sampler returned token IDs on the wrong device");
            }
        }

        std::int32_t read_single_token_id(const Tensor &token_ids)
        {
            Tensor cpu = token_ids.copy_to(Device::cpu());

            return static_cast<const std::int32_t *>(cpu.data())[0];
        }

    }   // namespace

    GenerationResult generate_token_ids(const Tensor &prompt_token_ids, const Qwen3Weights &weights, const ModelConfig &config, const Tensor &rope_cache,
                                        KvCache &kv_cache, cuda::CublasHandle &cublas_handle, Sampler &sampler, const GenerationOptions &options)
    {
        validate_generation_arguments(prompt_token_ids, weights, config, rope_cache, kv_cache, options);

        GenerationResult result;

        result.token_ids.reserve(static_cast<std::size_t>(options.max_new_tokens));

        if (options.max_new_tokens == 0) {
            return result;
        }

        // =========================================================================================
        // 1. Prefill: 一次性处理完整prompt
        // =========================================================================================

        Tensor hidden_states = qwen3_model_forward(prompt_token_ids, weights, config, rope_cache, kv_cache, 0, cublas_handle);

        // =========================================================================================
        // 2. 自回归生成
        // =========================================================================================

        for (std::int64_t generated_index = 0; generated_index < options.max_new_tokens; ++generated_index)
        {
            // hidden_states:
            // prefill 时是 [1, prompt_length, hidden_size]
            // decode 时是 [1, 1, hidden_size]
            //
            // LM Head 内部只取最后一个位置，返回 [1, vocab_size]
            Tensor logits = qwen3_lm_head(hidden_states, weights.output_weight(), cublas_handle);

            // 返回 CUDA Int32 [1, 1]
            Tensor next_token_ids = sampler.sample(logits);

            validate_sampled_token(next_token_ids, prompt_token_ids.device());

            const std::int32_t next_token_id = read_single_token_id(next_token_ids);

            if (next_token_id < 0 || next_token_id >= config.vocab_size) {
                throw std::logic_error("Sampler returned a token ID outside the vocabulary");
            }

            result.token_ids.push_back(next_token_id);

            // 先将 stop token 放进结果，再停止
            // 不要再把 stop token 喂回模型
            if (contains_stop_token(options.stop_token_ids, next_token_id))
            {
                result.stop_reason = StopReason::StopToken;

                return result;
            }

            // 已经得到所需数量的新 token
            // 最后一个 token 没必要再做一次无用 forward
            if (generated_index + 1 == options.max_new_tokens) {
                break;
            }

            // 刚采样出的 token 目前还没有进入 KV Cache
            //
            // prefill 后这里等于 prompt_Length
            // 每次 decode 后自动加 1
            const std::int64_t start_position = common_cached_length(kv_cache, config.num_hidden_layers);

            // 下一轮只计算刚生成的一个 token
            // 历史 token 的 K/V 已经保存在 KV Cache 中
            hidden_states = qwen3_model_forward(next_token_ids, weights, config, rope_cache, kv_cache, start_position, cublas_handle);
        }

        result.stop_reason = StopReason::MaxNewTokens;

        return result;
    }
}   // namespace qwen3::generation