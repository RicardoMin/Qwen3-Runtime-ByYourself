#include "qwen3/core/device.h"
#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cublas_handle.h"
#include "qwen3/cuda/cuda_check.h"
#include "qwen3/model/lm_head.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
    void require(
        bool condition,
        const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    std::uint16_t float_to_bfloat16(float value)
    {
        std::uint32_t bits = 0;

        std::memcpy(
            &bits,
            &value,
            sizeof(bits));

        const std::uint32_t rounding_bias =
            0x7fffU +
            ((bits >> 16U) & 1U);

        bits += rounding_bias;

        return static_cast<std::uint16_t>(
            bits >> 16U);
    }

    float bfloat16_to_float(
        std::uint16_t value)
    {
        const std::uint32_t bits =
            static_cast<std::uint32_t>(value)
            << 16U;

        float result = 0.0F;

        std::memcpy(
            &result,
            &bits,
            sizeof(result));

        return result;
    }

    void fill_bfloat16(
        qwen3::Tensor &tensor,
        std::initializer_list<float> values)
    {
        require(
            static_cast<std::int64_t>(values.size()) ==
                tensor.numel(),
            "test value count is incorrect");

        auto *destination =
            static_cast<std::uint16_t *>(
                tensor.data());

        std::size_t index = 0;

        for (const float value : values)
        {
            destination[index++] =
                float_to_bfloat16(value);
        }
    }

    void expect_values(
        const qwen3::Tensor &cuda_tensor,
        std::initializer_list<float> expected)
    {
        qwen3::Tensor cpu =
            cuda_tensor.copy_to(
                qwen3::Device::cpu());

        require(
            static_cast<std::int64_t>(expected.size()) ==
                cpu.numel(),
            "expected value count is incorrect");

        const auto *actual =
            static_cast<const std::uint16_t *>(
                cpu.data());

        std::size_t index = 0;

        for (const float expected_value : expected)
        {
            const float actual_value =
                bfloat16_to_float(actual[index++]);

            require(
                std::fabs(
                    actual_value -
                    expected_value) <= 0.001F,
                "LM Head numerical result is incorrect");
        }
    }
}

int main()
{
    try
    {
        int device_count = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDeviceCount(&device_count));

        require(
            device_count > 0,
            "no visible CUDA device");

        const qwen3::Device device =
            qwen3::Device::cuda(0);

        constexpr std::int64_t batch_size = 2;
        constexpr std::int64_t sequence_length = 3;
        constexpr std::int64_t hidden_size = 4;
        constexpr std::int64_t vocab_size = 5;

        qwen3::Tensor cpu_hidden =
            qwen3::Tensor::empty(
                qwen3::Shape{
                    batch_size,
                    sequence_length,
                    hidden_size},
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        // 每个 batch 前两个 token 都是哨兵值。
        // LM Head 必须只选择各 batch 最后一个 token。
        fill_bfloat16(
            cpu_hidden,
            {
                100, 100, 100, 100,
                200, 200, 200, 200,
                1, 2, 3, 4,

                -100, -100, -100, -100,
                -200, -200, -200, -200,
                -1, 0.5F, 2, -3
            });

        qwen3::Tensor cpu_weight =
            qwen3::Tensor::empty(
                qwen3::Shape{
                    vocab_size,
                    hidden_size},
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        fill_bfloat16(
            cpu_weight,
            {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
                1, 1, 1, 1
            });

        qwen3::Tensor cuda_hidden =
            cpu_hidden.copy_to(device);

        qwen3::Tensor cuda_weight =
            cpu_weight.copy_to(device);

        qwen3::cuda::CublasHandle handle{
            device};

        qwen3::Tensor selected =
            qwen3::select_last_token_hidden(
                cuda_hidden);

        require(
            selected.shape() ==
                qwen3::Shape{
                    batch_size,
                    hidden_size},
            "selected hidden shape is incorrect");

        expect_values(
            selected,
            {
                1, 2, 3, 4,
                -1, 0.5F, 2, -3
            });

        qwen3::Tensor logits =
            qwen3::qwen3_lm_head(
                cuda_hidden,
                cuda_weight,
                handle);

        require(
            logits.shape() ==
                qwen3::Shape{
                    batch_size,
                    vocab_size},
            "LM Head logits shape is incorrect");

        require(
            logits.dtype() ==
                qwen3::DType::BFloat16,
            "LM Head logits dtype is incorrect");

        require(
            logits.device() == device,
            "LM Head logits device is incorrect");

        expect_values(
            logits,
            {
                1, 2, 3, 4, 10,
                -1, 0.5F, 2, -3, -1.5F
            });

        QWEN3_CUDA_CHECK(
            cudaDeviceSynchronize());

        std::cout
            << "LM Head CUDA test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "LM Head CUDA test failed: "
            << error.what()
            << '\n';

        return EXIT_FAILURE;
    }
}