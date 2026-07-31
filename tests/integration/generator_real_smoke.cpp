#include "qwen3/core/device.h"
#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/core/tensor.h"

#include "qwen3/cuda/cublas_handle.h"
#include "qwen3/generation/generator.h"
#include "qwen3/generation/sampler.h"
#include "qwen3/io/safetensors_checkpoint.h"
#include "qwen3/model/config.h"
#include "qwen3/model/kv_cache.h"
#include "qwen3/model/lm_head.h"
#include "qwen3/model/model.h"
#include "qwen3/model/weights.h"
#include "qwen3/ops/rope.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
    void require(
        bool condition,
        const char *message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    qwen3::Tensor make_token_ids(
        const std::vector<std::int32_t> &values,
        const qwen3::Shape &shape,
        const qwen3::Device &device)
    {
        qwen3::Tensor cpu =
            qwen3::Tensor::empty(
                shape,
                qwen3::DType::Int32,
                qwen3::Device::cpu());

        require(
            cpu.numel() ==
                static_cast<std::int64_t>(values.size()),
            "token count does not match shape");

        auto *destination =
            static_cast<std::int32_t *>(cpu.data());

        for (std::size_t index = 0;
             index < values.size();
             ++index)
        {
            destination[index] = values[index];
        }

        return cpu.copy_to(device);
    }

    std::int32_t read_token_id(
        const qwen3::Tensor &token_ids)
    {
        qwen3::Tensor cpu =
            token_ids.copy_to(qwen3::Device::cpu());

        return static_cast<const std::int32_t *>(
            cpu.data())[0];
    }

    void require_cache_length(
        const qwen3::KvCache &cache,
        std::int32_t layer_count,
        std::int64_t expected_length)
    {
        for (std::int32_t layer_index = 0;
             layer_index < layer_count;
             ++layer_index)
        {
            require(
                cache.cached_length(layer_index) ==
                    expected_length,
                "KV cache length is incorrect");
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 2)
        {
            std::cerr
                << "Usage: generator_real_smoke "
                << "<Qwen3 model directory>\n";

            return 1;
        }

        const std::filesystem::path model_directory{
            argv[1]
        };

        const qwen3::Device device =
            qwen3::Device::cuda(0);

        const qwen3::ModelConfig config =
            qwen3::load_model_config(
                model_directory / "config.json");

        qwen3::SafetensorsCheckpoint checkpoint{
            model_directory
        };

        std::cout
            << "Loading real Qwen3 weights...\n";

        qwen3::Qwen3Weights weights =
            qwen3::Qwen3Weights::load(
                checkpoint,
                config,
                device);

        constexpr std::int64_t max_sequence_length = 8;
        constexpr std::int64_t prompt_length = 3;

        qwen3::Tensor rope_cache =
            qwen3::ops::build_rope_cache(
                max_sequence_length,
                config.head_dim,
                config.rope_theta,
                device);

        qwen3::cuda::CublasHandle cublas_handle{
            device
        };

        qwen3::generation::SamplingConfig sampling_config;
        sampling_config.mode =
            qwen3::generation::SamplingMode::Greedy;

        qwen3::generation::Sampler sampler{
            sampling_config
        };

        qwen3::Tensor prompt =
            make_token_ids(
                {1, 2, 3},
                qwen3::Shape{1, prompt_length},
                device);

        // ============================================================
        // 1. 手工执行两步，得到参考 token
        // ============================================================

        qwen3::KvCache reference_cache(
            config.num_hidden_layers,
            1,
            max_sequence_length,
            config.num_key_value_heads,
            config.head_dim,
            qwen3::DType::BFloat16,
            device);

        qwen3::Tensor reference_hidden_0 =
            qwen3::qwen3_model_forward(
                prompt,
                weights,
                config,
                rope_cache,
                reference_cache,
                0,
                cublas_handle);

        qwen3::Tensor reference_logits_0 =
            qwen3::qwen3_lm_head(
                reference_hidden_0,
                weights.output_weight(),
                cublas_handle);

        qwen3::Tensor reference_token_0 =
            sampler.sample(reference_logits_0);

        const std::int32_t expected_token_0 =
            read_token_id(reference_token_0);

        qwen3::Tensor reference_hidden_1 =
            qwen3::qwen3_model_forward(
                reference_token_0,
                weights,
                config,
                rope_cache,
                reference_cache,
                prompt_length,
                cublas_handle);

        qwen3::Tensor reference_logits_1 =
            qwen3::qwen3_lm_head(
                reference_hidden_1,
                weights.output_weight(),
                cublas_handle);

        qwen3::Tensor reference_token_1 =
            sampler.sample(reference_logits_1);

        const std::int32_t expected_token_1 =
            read_token_id(reference_token_1);

        // ============================================================
        // 2. 使用 Generator 执行相同过程
        // ============================================================

        qwen3::KvCache generator_cache(
            config.num_hidden_layers,
            1,
            max_sequence_length,
            config.num_key_value_heads,
            config.head_dim,
            qwen3::DType::BFloat16,
            device);

        qwen3::generation::GenerationOptions options;
        options.max_new_tokens = 2;

        // 此处不设置 stop token，确保一定生成两个 token。
        qwen3::generation::GenerationResult result =
            qwen3::generation::generate_token_ids(
                prompt,
                weights,
                config,
                rope_cache,
                generator_cache,
                cublas_handle,
                sampler,
                options);

        require(
            result.token_ids.size() == 2,
            "Generator returned the wrong token count");

        require(
            result.token_ids[0] == expected_token_0,
            "Generator first token differs from manual reference");

        require(
            result.token_ids[1] == expected_token_1,
            "Generator second token differs from manual reference");

        require(
            result.stop_reason ==
                qwen3::generation::StopReason::MaxNewTokens,
            "Generator stop reason is incorrect");

        // prompt 长度 3，生成 2 个 token。
        // 最后一个 token 没有喂回模型，所以 cache 长度是 4。
        require_cache_length(
            generator_cache,
            config.num_hidden_layers,
            prompt_length + 1);

        // ============================================================
        // 3. 验证 stop token 会立即停止
        // ============================================================

        qwen3::KvCache stop_cache(
            config.num_hidden_layers,
            1,
            max_sequence_length,
            config.num_key_value_heads,
            config.head_dim,
            qwen3::DType::BFloat16,
            device);

        qwen3::generation::GenerationOptions stop_options;
        stop_options.max_new_tokens = 3;

        // 动态把真实预测出的第一个 token 当成 stop token。
        stop_options.stop_token_ids = {
            expected_token_0
        };

        qwen3::generation::GenerationResult stop_result =
            qwen3::generation::generate_token_ids(
                prompt,
                weights,
                config,
                rope_cache,
                stop_cache,
                cublas_handle,
                sampler,
                stop_options);

        require(
            stop_result.token_ids.size() == 1,
            "Generator did not stop after the stop token");

        require(
            stop_result.token_ids[0] == expected_token_0,
            "Generator returned an unexpected stop token");

        require(
            stop_result.stop_reason ==
                qwen3::generation::StopReason::StopToken,
            "Generator stop reason should be StopToken");

        // stop token 刚采样出来，还没有被喂回模型。
        require_cache_length(
            stop_cache,
            config.num_hidden_layers,
            prompt_length);

        std::cout
            << "Generator real smoke passed.\n"
            << "Generated token IDs: "
            << result.token_ids[0]
            << ", "
            << result.token_ids[1]
            << '\n'
            << "Final cache length: "
            << prompt_length + 1
            << '\n';

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Generator real smoke failed: "
            << error.what()
            << '\n';

        return 1;
    }
}