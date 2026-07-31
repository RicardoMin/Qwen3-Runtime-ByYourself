#include "qwen3/core/device.h"
#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/core/tensor.h"

#include "qwen3/cuda/cublas_handle.h"

#include "qwen3/io/safetensors_checkpoint.h"

#include "qwen3/model/config.h"
#include "qwen3/model/mlp.h"
#include "qwen3/model/weights.h"

#include "qwen3/ops/embedding.h"
#include "qwen3/ops/linear.h"
#include "qwen3/ops/rms_norm.h"
#include "qwen3/ops/elementwise.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
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
        require(
            static_cast<std::int64_t>(values.size()) ==
                shape.numel(),
            "token count does not match shape");

        qwen3::Tensor cpu =
            qwen3::Tensor::empty(
                shape,
                qwen3::DType::Int32,
                qwen3::Device::cpu());

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
            "actual and expected shapes differ");

        qwen3::Tensor actual_cpu =
            actual.copy_to(qwen3::Device::cpu());

        qwen3::Tensor expected_cpu =
            expected.copy_to(qwen3::Device::cpu());

        const auto *actual_values =
            static_cast<const __nv_bfloat16 *>(
                actual_cpu.data());

        const auto *expected_values =
            static_cast<const __nv_bfloat16 *>(
                expected_cpu.data());

        bool found_nonzero = false;

        for (std::int64_t index = 0;
             index < actual_cpu.numel();
             ++index)
        {
            const float actual_value =
                __bfloat162float(actual_values[index]);

            const float expected_value =
                __bfloat162float(expected_values[index]);

            require(
                std::isfinite(actual_value),
                "MLP output contains NaN or Inf");

            require(
                std::isfinite(expected_value),
                "reference output contains NaN or Inf");

            const float difference =
                std::abs(actual_value - expected_value);

            const float tolerance =
                0.01F +
                0.01F * std::abs(expected_value);

            if (difference > tolerance)
            {
                throw std::runtime_error(
                    "MLP output differs from explicit "
                    "SwiGLU reference chain");
            }

            if (actual_value != 0.0F)
            {
                found_nonzero = true;
            }
        }

        require(
            found_nonzero,
            "MLP output is unexpectedly all zero");
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 2)
        {
            std::cerr
                << "Usage: mlp_real_smoke "
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

        require(
            layer0.gate_proj.shape() ==
                qwen3::Shape{
                    config.intermediate_size,
                    config.hidden_size},
            "real gate_proj shape is incorrect");

        require(
            layer0.up_proj.shape() ==
                qwen3::Shape{
                    config.intermediate_size,
                    config.hidden_size},
            "real up_proj shape is incorrect");

        require(
            layer0.down_proj.shape() ==
                qwen3::Shape{
                    config.hidden_size,
                    config.intermediate_size},
            "real down_proj shape is incorrect");

        qwen3::Tensor token_ids =
            make_token_ids(
                {1, 2},
                qwen3::Shape{1, 2},
                device);

        // 使用真实 embedding 权重构造输入。
        qwen3::Tensor embeddings =
            qwen3::ops::embedding_lookup(
                token_ids,
                weights.token_embedding());

        // MLP 接收的输入应该已经经过
        // post_attention_layernorm。
        //
        // 这里没有先跑 attention，是为了让失败范围只落在 MLP。
        qwen3::Tensor mlp_input =
            qwen3::ops::rms_norm(
                embeddings,
                layer0.post_attention_layernorm,
                config.rms_norm_eps);

        qwen3::cuda::CublasHandle cublas_handle{
            device
        };

        // 调用刚刚组合好的 MLP 模块。
        qwen3::Tensor actual =
            qwen3::qwen3_mlp(
                mlp_input,
                layer0,
                config,
                cublas_handle);

        require(
            actual.shape() ==
                qwen3::Shape{
                    1,
                    2,
                    config.hidden_size},
            "MLP output shape is incorrect");

        // 显式执行同一条 SwiGLU 线路作为组合参考。
        qwen3::Tensor gate =
            qwen3::ops::linear(
                mlp_input,
                layer0.gate_proj,
                cublas_handle);

        qwen3::Tensor up =
            qwen3::ops::linear(
                mlp_input,
                layer0.up_proj,
                cublas_handle);

        qwen3::Tensor activated =
            qwen3::ops::silu_multiply(
                gate,
                up);

        qwen3::Tensor expected =
            qwen3::ops::linear(
                activated,
                layer0.down_proj,
                cublas_handle);

        require_close(actual, expected);

        const cudaError_t synchronize_status =
            cudaDeviceSynchronize();

        require(
            synchronize_status == cudaSuccess,
            cudaGetErrorString(synchronize_status));

        std::cout
            << "Real Qwen3 layer-0 MLP test passed.\n"
            << "Input shape: "
            << mlp_input.shape().to_string()
            << '\n'
            << "Intermediate shape: "
            << gate.shape().to_string()
            << '\n'
            << "Output shape: "
            << actual.shape().to_string()
            << '\n';

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Real Qwen3 MLP test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}