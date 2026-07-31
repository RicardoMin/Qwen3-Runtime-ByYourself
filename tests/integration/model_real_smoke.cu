#include "qwen3/core/device.h"
#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/core/tensor.h"

#include "qwen3/cuda/cublas_handle.h"
#include "qwen3/io/safetensors_checkpoint.h"

#include "qwen3/model/config.h"
#include "qwen3/model/kv_cache.h"
#include "qwen3/model/model.h"
#include "qwen3/model/weights.h"
#include "qwen3/model/lm_head.h"

#include "qwen3/ops/rope.h"
#include "qwen3/ops/argmax.h"

#include "qwen3/generation/sampler.h"

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

    void require_all_finite(
        const qwen3::Tensor &tensor)
    {
        qwen3::Tensor cpu =
            tensor.copy_to(qwen3::Device::cpu());

        const auto *values =
            static_cast<const __nv_bfloat16 *>(
                cpu.data());

        bool found_nonzero = false;

        for (std::int64_t index = 0;
             index < cpu.numel();
             ++index)
        {
            const float value =
                __bfloat162float(values[index]);

            require(
                std::isfinite(value),
                "model output contains NaN or Inf");

            if (value != 0.0F)
            {
                found_nonzero = true;
            }
        }

        require(
            found_nonzero,
            "model output is unexpectedly all zero");
    }

    void compare_last_token(
        const qwen3::Tensor &one_shot,
        const qwen3::Tensor &split_decode,
        std::int64_t hidden_size)
    {
        require(
            one_shot.shape().rank() == 3,
            "one-shot output must be rank 3");

        require(
            split_decode.shape() ==
                qwen3::Shape{1, 1, hidden_size},
            "split decode output shape is incorrect");

        qwen3::Tensor one_shot_cpu =
            one_shot.copy_to(qwen3::Device::cpu());

        qwen3::Tensor split_cpu =
            split_decode.copy_to(qwen3::Device::cpu());

        const auto *one_shot_values =
            static_cast<const __nv_bfloat16 *>(
                one_shot_cpu.data());

        const auto *split_values =
            static_cast<const __nv_bfloat16 *>(
                split_cpu.data());

        const std::int64_t last_token_index =
            one_shot.shape()[1] - 1;

        const std::int64_t one_shot_offset =
            last_token_index * hidden_size;

        for (std::int64_t hidden_index = 0;
             hidden_index < hidden_size;
             ++hidden_index)
        {
            const float one_shot_value =
                __bfloat162float(
                    one_shot_values[one_shot_offset +
                                    hidden_index]);

            const float split_value =
                __bfloat162float(
                    split_values[hidden_index]);

            require(
                std::isfinite(one_shot_value) &&
                    std::isfinite(split_value),
                "comparison encountered NaN or Inf");

            const float difference =
                std::abs(
                    one_shot_value -
                    split_value);

            const float tolerance =
                0.04F +
                0.03F * std::abs(
                            one_shot_value);

            if (difference > tolerance)
            {
                printf("difference: %.4f  tolerance: %.4f\n", difference, tolerance);
                throw std::runtime_error(
                    "one-shot and cached decode outputs differ");
            }
        }
    }

    void require_all_cache_lengths(
        const qwen3::KvCache &cache,
        std::int32_t layer_count,
        std::int64_t expected_length)
    {
        for (std::int32_t layer_index = 0;
             layer_index < layer_count;
             ++layer_index)
        {
            if (cache.cached_length(layer_index) !=
                expected_length)
            {
                throw std::runtime_error(
                    "a model layer has an incorrect "
                    "KV cache length");
            }
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
                << "Usage: model_real_smoke "
                << "<Qwen3-0.6B model directory>\n";

            return 1;
        }

        const qwen3::Device device =
            qwen3::Device::cuda(0);

        const qwen3::ModelConfig config =
            qwen3::make_qwen3_0_6b_config();

        qwen3::SafetensorsCheckpoint checkpoint{
            argv[1]};

        std::cout
            << "Loading real Qwen3 model weights...\n";

        qwen3::Qwen3Weights weights =
            qwen3::Qwen3Weights::load(
                checkpoint,
                config,
                device);

        constexpr std::int64_t max_sequence_length = 8;

        qwen3::Tensor rope_cache =
            qwen3::ops::build_rope_cache(
                max_sequence_length,
                config.head_dim,
                config.rope_theta,
                device);

        // 一次性输入使用自己的 cache。
        qwen3::KvCache one_shot_cache(
            config.num_hidden_layers,
            1,
            max_sequence_length,
            config.num_key_value_heads,
            config.head_dim,
            qwen3::DType::BFloat16,
            device);

        // prefill + decode 使用另一套 cache。
        qwen3::KvCache split_cache(
            config.num_hidden_layers,
            1,
            max_sequence_length,
            config.num_key_value_heads,
            config.head_dim,
            qwen3::DType::BFloat16,
            device);

        qwen3::cuda::CublasHandle cublas_handle{
            device};

        // ==============================================
        // 路径一：一次性输入三个 token
        // ==============================================

        qwen3::Tensor one_shot_tokens =
            make_token_ids(
                {1, 2, 3},
                qwen3::Shape{1, 3},
                device);

        qwen3::Tensor one_shot_output =
            qwen3::qwen3_model_forward(
                one_shot_tokens,
                weights,
                config,
                rope_cache,
                one_shot_cache,
                0,
                cublas_handle);

        require(
            one_shot_output.shape() ==
                qwen3::Shape{
                    1,
                    3,
                    config.hidden_size},
            "one-shot output shape is incorrect");

        require_all_finite(one_shot_output);

        require_all_cache_lengths(
            one_shot_cache,
            config.num_hidden_layers,
            3);

        // ==============================================
        // 路径二：先 prefill 两个 token
        // ==============================================

        qwen3::Tensor prefill_tokens =
            make_token_ids(
                {1, 2},
                qwen3::Shape{1, 2},
                device);

        qwen3::Tensor prefill_output =
            qwen3::qwen3_model_forward(
                prefill_tokens,
                weights,
                config,
                rope_cache,
                split_cache,
                0,
                cublas_handle);

        require(
            prefill_output.shape() ==
                qwen3::Shape{
                    1,
                    2,
                    config.hidden_size},
            "prefill output shape is incorrect");

        require_all_finite(prefill_output);

        require_all_cache_lengths(
            split_cache,
            config.num_hidden_layers,
            2);

        // ==============================================
        // 路径二：再 decode 第三个 token
        // ==============================================

        qwen3::Tensor decode_tokens =
            make_token_ids(
                {3},
                qwen3::Shape{1, 1},
                device);

        qwen3::Tensor decode_output =
            qwen3::qwen3_model_forward(
                decode_tokens,
                weights,
                config,
                rope_cache,
                split_cache,
                2,
                cublas_handle);

        require(
            decode_output.shape() ==
                qwen3::Shape{
                    1,
                    1,
                    config.hidden_size},
            "decode output shape is incorrect");

        require_all_finite(decode_output);

        require_all_cache_lengths(
            split_cache,
            config.num_hidden_layers,
            3);

        // ============================================================================================
        // 下面这个函数执行失败
        // ============================================================================================

        // 对比两种执行方式的最后一个 token。
        compare_last_token(
            one_shot_output,
            decode_output,
            config.hidden_size);

        const cudaError_t status =
            cudaDeviceSynchronize();

        require(
            status == cudaSuccess,
            cudaGetErrorString(status));

        qwen3::Tensor decode_logits =
            qwen3::qwen3_lm_head(
                decode_output,
                weights.output_weight(),
                cublas_handle);

        require(
            decode_logits.shape() ==
                qwen3::Shape{
                    1,
                    config.vocab_size},
            "decode logits shape is incorrect");

        require_all_finite(decode_logits);

        qwen3::Tensor next_token_ids =
            qwen3::ops::argmax_last_dimension(
                decode_logits,
                true);

        require(
            next_token_ids.shape() ==
                qwen3::Shape{1, 1},
            "next token IDs shape is incorrect");

        require(
            next_token_ids.dtype() ==
                qwen3::DType::Int32,
            "next token IDs dtype is incorrect");

        qwen3::Tensor next_token_ids_cpu =
            next_token_ids.copy_to(
                qwen3::Device::cpu());

        std::int32_t next_token_id =
            static_cast<std::int32_t *>(
                next_token_ids_cpu.data())[0];

        require(
            next_token_id >= 0 &&
                next_token_id <
                    config.vocab_size,
            "predicted token ID is out of range");

        std::cout
            << "Predicted next token ID: "
            << next_token_id
            << '\n';

        // ============================================================================================================================
        // ==============================================================================================================================

        // topk-topp 获得后一个token
        qwen3::generation::SamplingConfig
            sampling_config;

        sampling_config.mode =
            qwen3::generation::
                SamplingMode::Sample;

        sampling_config.temperature = 0.6F;
        sampling_config.top_k = 20;
        sampling_config.top_p = 0.95F;
        sampling_config.seed = 42;

        // 真正生成时，这个对象应创建在生成循环外。
        qwen3::generation::Sampler sampler{
            sampling_config};

        next_token_ids =
            sampler.sample(
                decode_logits);

        require(
            next_token_ids.shape() ==
                qwen3::Shape{1, 1},
            "sampled token shape is incorrect");

        next_token_ids_cpu =
            next_token_ids.copy_to(
                qwen3::Device::cpu());

        next_token_id =
            static_cast<const std::int32_t *>(
                next_token_ids_cpu.data())[0];

        require(
            next_token_id >= 0 &&
                next_token_id <
                    config.vocab_size,
            "sampled token ID is out of range");

        std::cout
            << "Sampled next token ID: "
            << next_token_id
            << '\n';

        std::cout
            << "Real Qwen3 model backbone test passed.\n"
            << "One-shot output: "
            << one_shot_output.shape().to_string()
            << '\n'
            << "Prefill output: "
            << prefill_output.shape().to_string()
            << '\n'
            << "Decode output: "
            << decode_output.shape().to_string()
            << '\n'
            << "All layer cache lengths: 3\n"
            << "Decode logits: "
            << decode_logits.shape().to_string()
            << '\n';

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Model backbone test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}