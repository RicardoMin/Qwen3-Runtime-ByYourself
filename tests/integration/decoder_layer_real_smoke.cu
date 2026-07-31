#include "qwen3/core/device.h"
#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/core/tensor.h"

#include "qwen3/cuda/cublas_handle.h"
#include "qwen3/io/safetensors_checkpoint.h"

#include "qwen3/model/config.h"
#include "qwen3/model/decoder_layer.h"
#include "qwen3/model/kv_cache.h"
#include "qwen3/model/mlp.h"
#include "qwen3/model/self_attention.h"
#include "qwen3/model/weights.h"

#include "qwen3/ops/elementwise.h"
#include "qwen3/ops/embedding.h"
#include "qwen3/ops/rms_norm.h"
#include "qwen3/ops/rope.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

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
        std::initializer_list<std::int32_t> values,
        const qwen3::Shape &shape,
        const qwen3::Device &device)
    {
        qwen3::Tensor cpu =
            qwen3::Tensor::empty(
                shape,
                qwen3::DType::Int32,
                qwen3::Device::cpu());

        require(
            static_cast<std::int64_t>(values.size()) ==
                cpu.numel(),
            "token count does not match shape");

        auto *destination =
            static_cast<std::int32_t *>(cpu.data());

        std::int64_t index = 0;

        for (const std::int32_t value : values)
        {
            destination[index++] = value;
        }

        return cpu.copy_to(device);
    }

    void require_close(
        const qwen3::Tensor &actual,
        const qwen3::Tensor &expected)
    {
        require(
            actual.shape() == expected.shape(),
            "decoder outputs have different shapes");

        qwen3::Tensor actual_cpu =
            actual.copy_to(qwen3::Device::cpu());

        qwen3::Tensor expected_cpu =
            expected.copy_to(qwen3::Device::cpu());

        const auto *actual_data =
            static_cast<const __nv_bfloat16 *>(
                actual_cpu.data());

        const auto *expected_data =
            static_cast<const __nv_bfloat16 *>(
                expected_cpu.data());

        for (std::int64_t index = 0;
             index < actual_cpu.numel();
             ++index)
        {
            const float actual_value =
                __bfloat162float(actual_data[index]);

            const float expected_value =
                __bfloat162float(expected_data[index]);

            require(
                std::isfinite(actual_value),
                "actual decoder output contains NaN/Inf");

            require(
                std::isfinite(expected_value),
                "reference decoder output contains NaN/Inf");

            const float difference =
                std::abs(actual_value - expected_value);

            const float tolerance =
                0.02F +
                0.02F * std::abs(expected_value);

            if (difference > tolerance)
            {
                throw std::runtime_error(
                    "decoder output differs from "
                    "the explicit reference chain");
            }
        }
    }

    qwen3::Tensor reference_decoder_layer(
        qwen3::Tensor hidden_states,
        const qwen3::Qwen3LayerWeights &weights,
        const qwen3::ModelConfig &config,
        const qwen3::Tensor &rope_cache,
        qwen3::KvCache &kv_cache,
        std::int32_t layer_index,
        std::int64_t start_position,
        qwen3::cuda::CublasHandle &cublas_handle)
    {
        qwen3::Tensor attention_input =
            qwen3::ops::rms_norm(
                hidden_states,
                weights.input_layernorm,
                config.rms_norm_eps);

        qwen3::Tensor attention_delta =
            qwen3::qwen3_self_attention(
                attention_input,
                weights,
                config,
                rope_cache,
                kv_cache,
                layer_index,
                start_position,
                cublas_handle);

        qwen3::ops::add_inplace(
            hidden_states,
            attention_delta);

        qwen3::Tensor mlp_input =
            qwen3::ops::rms_norm(
                hidden_states,
                weights.post_attention_layernorm,
                config.rms_norm_eps);

        qwen3::Tensor mlp_delta =
            qwen3::qwen3_mlp(
                mlp_input,
                weights,
                config,
                cublas_handle);

        qwen3::ops::add_inplace(
            hidden_states,
            mlp_delta);

        return hidden_states;
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 2)
        {
            std::cerr
                << "Usage: decoder_layer_real_smoke "
                << "<Qwen3-0.6B model directory>\n";

            return 1;
        }

        const qwen3::Device device =
            qwen3::Device::cuda(0);

        const qwen3::ModelConfig config =
            qwen3::make_qwen3_0_6b_config();

        qwen3::SafetensorsCheckpoint checkpoint{
            argv[1]
        };

        std::cout
            << "Loading real Qwen3-0.6B weights...\n";

        qwen3::Qwen3Weights weights =
            qwen3::Qwen3Weights::load(
                checkpoint,
                config,
                device);

        const qwen3::Qwen3LayerWeights &layer0 =
            weights.layer(0);

        constexpr std::int64_t max_sequence_length = 8;

        qwen3::Tensor rope_cache =
            qwen3::ops::build_rope_cache(
                max_sequence_length,
                config.head_dim,
                config.rope_theta,
                device);

        qwen3::KvCache actual_cache(
            config.num_hidden_layers,
            1,
            max_sequence_length,
            config.num_key_value_heads,
            config.head_dim,
            qwen3::DType::BFloat16,
            device);

        qwen3::KvCache reference_cache(
            config.num_hidden_layers,
            1,
            max_sequence_length,
            config.num_key_value_heads,
            config.head_dim,
            qwen3::DType::BFloat16,
            device);

        qwen3::cuda::CublasHandle cublas_handle{
            device
        };

        // ==============================================
        // Prefill：两个 token
        // ==============================================

        qwen3::Tensor prefill_tokens =
            make_token_ids(
                {1, 2},
                qwen3::Shape{1, 2},
                device);

        qwen3::Tensor prefill_embeddings =
            qwen3::ops::embedding_lookup(
                prefill_tokens,
                weights.token_embedding());

        qwen3::Tensor actual_prefill =
            qwen3::qwen3_decoder_layer(
                prefill_embeddings.clone(),
                layer0,
                config,
                rope_cache,
                actual_cache,
                0,
                0,
                cublas_handle);

        qwen3::Tensor expected_prefill =
            reference_decoder_layer(
                prefill_embeddings.clone(),
                layer0,
                config,
                rope_cache,
                reference_cache,
                0,
                0,
                cublas_handle);

        require_close(
            actual_prefill,
            expected_prefill);

        require(
            actual_cache.cached_length(0) == 2,
            "actual prefill cache length must be 2");

        require(
            reference_cache.cached_length(0) == 2,
            "reference prefill cache length must be 2");

        // ==============================================
        // Decode：第三个 token
        // ==============================================

        qwen3::Tensor decode_tokens =
            make_token_ids(
                {3},
                qwen3::Shape{1, 1},
                device);

        qwen3::Tensor decode_embeddings =
            qwen3::ops::embedding_lookup(
                decode_tokens,
                weights.token_embedding());

        qwen3::Tensor actual_decode =
            qwen3::qwen3_decoder_layer(
                decode_embeddings.clone(),
                layer0,
                config,
                rope_cache,
                actual_cache,
                0,
                2,
                cublas_handle);

        qwen3::Tensor expected_decode =
            reference_decoder_layer(
                decode_embeddings.clone(),
                layer0,
                config,
                rope_cache,
                reference_cache,
                0,
                2,
                cublas_handle);

        require_close(
            actual_decode,
            expected_decode);

        require(
            actual_cache.cached_length(0) == 3,
            "actual decode cache length must be 3");

        const cudaError_t status =
            cudaDeviceSynchronize();

        require(
            status == cudaSuccess,
            cudaGetErrorString(status));

        std::cout
            << "Real Qwen3 decoder layer test passed.\n"
            << "Prefill output: "
            << actual_prefill.shape().to_string()
            << '\n'
            << "Decode output: "
            << actual_decode.shape().to_string()
            << '\n'
            << "KV cache length: "
            << actual_cache.cached_length(0)
            << '\n';

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Decoder layer test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}