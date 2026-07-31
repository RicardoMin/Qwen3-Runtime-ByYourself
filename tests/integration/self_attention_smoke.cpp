#include "qwen3/core/device.h"
#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/core/tensor.h"

#include "qwen3/cuda/cublas_handle.h"

#include "qwen3/io/safetensors_checkpoint.h"

#include "qwen3/model/config.h"
#include "qwen3/model/kv_cache.h"
#include "qwen3/model/self_attention.h"
#include "qwen3/model/weights.h"

#include "qwen3/ops/embedding.h"
#include "qwen3/ops/rms_norm.h"
#include "qwen3/ops/rope.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace
{
    void require(bool condition, const char *message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    qwen3::Tensor make_token_ids(
        std::initializer_list<std::int32_t> token_ids,
        const qwen3::Shape &shape,
        const qwen3::Device &device)
    {
        require(
            static_cast<std::int64_t>(token_ids.size()) ==
                shape.numel(),
            "token count does not match shape");

        qwen3::Tensor cpu_tokens =
            qwen3::Tensor::empty(
                shape,
                qwen3::DType::Int32,
                qwen3::Device::cpu());

        auto *destination =
            static_cast<std::int32_t *>(
                cpu_tokens.data());

        std::int64_t index = 0;

        for (const std::int32_t token_id : token_ids)
        {
            destination[index++] = token_id;
        }

        return cpu_tokens.copy_to(device);
    }

    void require_all_finite(
        const qwen3::Tensor &cuda_tensor)
    {
        qwen3::Tensor cpu_tensor =
            cuda_tensor.copy_to(
                qwen3::Device::cpu());

        require(
            cpu_tensor.dtype() ==
                qwen3::DType::BFloat16,
            "expected BF16 output");

        const auto *values =
            static_cast<const __nv_bfloat16 *>(
                cpu_tensor.data());

        for (std::int64_t index = 0;
             index < cpu_tensor.numel();
             ++index)
        {
            const float value =
                __bfloat162float(values[index]);

            if (!std::isfinite(value))
            {
                throw std::runtime_error(
                    "attention output contains NaN or Inf");
            }
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 3)
        {
            std::cerr
                << "Usage: self_attention_real_smoke "
                << "<Qwen3-0.6B model directory>\n";

            return 1;
        }

        // CUDA_VISIBLE_DEVICES=6 后，
        // 物理 GPU 6 会映射为进程内的 cuda:0。
        const qwen3::Device device =
            qwen3::Device::cuda(0);

        // 这一轮明确测试 Qwen3-0.6B。
        const qwen3::ModelConfig config =
            qwen3::load_model_config(argv[1]);

        // 同时支持单文件和分片 checkpoint。
        qwen3::SafetensorsCheckpoint checkpoint{
            argv[2]
        };

        std::cout
            << "Loading real Qwen3-0.6B weights...\n";

        qwen3::Qwen3Weights weights =
            qwen3::Qwen3Weights::load(
                checkpoint,
                config,
                device);

        require(
            weights.layer_count() ==
                static_cast<std::size_t>(
                    config.num_hidden_layers),
            "model layer count is incorrect");

        // 本次只计算第 0 层。
        const qwen3::Qwen3LayerWeights &layer0 =
            weights.layer(0);

        // 测试只需要 3 个位置：
        // 先 prefill 两个 token，再 decode 一个 token。
        constexpr std::int64_t cache_sequence_length = 8;

        qwen3::Tensor rope_cache =
            qwen3::ops::build_rope_cache(
                cache_sequence_length,
                config.head_dim,
                config.rope_theta,
                device);

        qwen3::KvCache kv_cache(
            config.num_hidden_layers,
            1,                          // max batch size
            cache_sequence_length,
            config.num_key_value_heads,
            config.head_dim,
            qwen3::DType::BFloat16,
            device);

        qwen3::cuda::CublasHandle cublas_handle{
            device
        };

        // ==================================================
        // 1. Prefill：输入两个真实 token embedding
        // ==================================================

        qwen3::Tensor prefill_tokens =
            make_token_ids(
                {1, 2},
                qwen3::Shape{1, 2},
                device);

        qwen3::Tensor prefill_embeddings =
            qwen3::ops::embedding_lookup(
                prefill_tokens,
                weights.token_embedding());

        require(
            prefill_embeddings.shape() ==
                qwen3::Shape{
                    1,
                    2,
                    config.hidden_size},
            "prefill embedding shape is incorrect");

        // Self-Attention 接收的是已经经过
        // input_layernorm 的 hidden states。
        qwen3::Tensor normalized_prefill =
            qwen3::ops::rms_norm(
                prefill_embeddings,
                layer0.input_layernorm,
                config.rms_norm_eps);

        qwen3::Tensor prefill_output =
            qwen3::qwen3_self_attention(
                normalized_prefill,
                layer0,
                config,
                rope_cache,
                kv_cache,
                0,                  // layer_index
                0,                  // start_position
                cublas_handle);

        require(
            prefill_output.shape() ==
                qwen3::Shape{
                    1,
                    2,
                    config.hidden_size},
            "prefill attention output shape is incorrect");

        require(
            kv_cache.cached_length(0) == 2,
            "KV cache length after prefill must be 2");

        require_all_finite(prefill_output);

        std::cout
            << "Real layer 0 prefill passed.\n"
            << "Output shape: "
            << prefill_output.shape().to_string()
            << '\n';

        // ==================================================
        // 2. Decode：加入第三个真实 token embedding
        // ==================================================

        qwen3::Tensor decode_tokens =
            make_token_ids(
                {3},
                qwen3::Shape{1, 1},
                device);

        qwen3::Tensor decode_embeddings =
            qwen3::ops::embedding_lookup(
                decode_tokens,
                weights.token_embedding());

        qwen3::Tensor normalized_decode =
            qwen3::ops::rms_norm(
                decode_embeddings,
                layer0.input_layernorm,
                config.rms_norm_eps);

        qwen3::Tensor decode_output =
            qwen3::qwen3_self_attention(
                normalized_decode,
                layer0,
                config,
                rope_cache,
                kv_cache,
                0,                  // layer_index
                2,                  // 已缓存两个 token
                cublas_handle);

        require(
            decode_output.shape() ==
                qwen3::Shape{
                    1,
                    1,
                    config.hidden_size},
            "decode attention output shape is incorrect");

        require(
            kv_cache.cached_length(0) == 3,
            "KV cache length after decode must be 3");

        require_all_finite(decode_output);

        const cudaError_t synchronize_status =
            cudaDeviceSynchronize();

        require(
            synchronize_status == cudaSuccess,
            cudaGetErrorString(synchronize_status));

        std::cout
            << "Real layer 0 decode passed.\n"
            << "Output shape: "
            << decode_output.shape().to_string()
            << '\n'
            << "KV cache length: "
            << kv_cache.cached_length(0)
            << '\n'
            << "Real Qwen3 layer-0 Self-Attention test passed.\n";

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Real Self-Attention test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}