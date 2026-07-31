#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cuda_check.h"
#include "qwen3/ops/embedding.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

    // 本测试只使用可以被 BF16 精确表达的整数，
    // 因此直接保留 float 高 16 位即可。
    return static_cast<std::uint16_t>(
        bits >> 16U);
}

float bfloat16_to_float(std::uint16_t value)
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

} // namespace

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

        int device_before = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDevice(&device_before));

        constexpr std::int64_t vocab_size = 5;
        constexpr std::int64_t hidden_size = 4;
        constexpr std::int64_t token_count = 6;

        // 创建一个 [5, 4] 的 embedding 表。
        qwen3::Tensor cpu_weight =
            qwen3::Tensor::empty(
                qwen3::Shape{
                    vocab_size,
                    hidden_size},
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        auto *weight_values =
            static_cast<std::uint16_t *>(
                cpu_weight.data());

        // 第 token 行保存：
        // token * 10 + hidden_index
        //
        // row 0 = [0, 1, 2, 3]
        // row 1 = [10, 11, 12, 13]
        // row 2 = [20, 21, 22, 23]
        for (std::int64_t token = 0;
             token < vocab_size;
             ++token)
        {
            for (std::int64_t hidden = 0;
                 hidden < hidden_size;
                 ++hidden)
            {
                const float value =
                    static_cast<float>(
                        token * 10 + hidden);

                weight_values[
                    token * hidden_size +
                    hidden] =
                    float_to_bfloat16(value);
            }
        }

        // shape = [2, 3]
        qwen3::Tensor cpu_token_ids =
            qwen3::Tensor::empty(
                qwen3::Shape{2, 3},
                qwen3::DType::Int32,
                qwen3::Device::cpu());

        auto *token_values =
            static_cast<std::int32_t *>(
                cpu_token_ids.data());

        const std::int32_t expected_tokens[
            token_count] = {
                2, 0, 4,
                1, 2, 3};

        for (std::int64_t index = 0;
             index < token_count;
             ++index)
        {
            token_values[index] =
                expected_tokens[index];
        }

        qwen3::Tensor cuda_weight =
            cpu_weight.copy_to(
                qwen3::Device::cuda(0));

        qwen3::Tensor cuda_token_ids =
            cpu_token_ids.copy_to(
                qwen3::Device::cuda(0));

        qwen3::Tensor cuda_output =
            qwen3::ops::embedding_lookup(
                cuda_token_ids,
                cuda_weight);

        require(
            cuda_output.shape() ==
                qwen3::Shape{2, 3, hidden_size},
            "embedding output shape is incorrect");

        require(
            cuda_output.dtype() ==
                qwen3::DType::BFloat16,
            "embedding output dtype is incorrect");

        require(
            cuda_output.device() ==
                qwen3::Device::cuda(0),
            "embedding output device is incorrect");

        qwen3::Tensor cpu_output =
            cuda_output.copy_to(
                qwen3::Device::cpu());

        const auto *output_values =
            static_cast<const std::uint16_t *>(
                cpu_output.data());

        for (std::int64_t token_index = 0;
             token_index < token_count;
             ++token_index)
        {
            const std::int32_t token_id =
                expected_tokens[token_index];

            for (std::int64_t hidden = 0;
                 hidden < hidden_size;
                 ++hidden)
            {
                const float actual =
                    bfloat16_to_float(
                        output_values[
                            token_index *
                                hidden_size +
                            hidden]);

                const float expected =
                    static_cast<float>(
                        token_id * 10 +
                        hidden);

                require(
                    std::fabs(actual - expected) <
                        0.001F,
                    "embedding output value "
                    "is incorrect");
            }
        }

        int device_after = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDevice(&device_after));

        require(
            device_after == device_before,
            "embedding operation changed "
            "the current CUDA device");

        std::cout
            << "CUDA embedding test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "CUDA embedding test failed: "
            << error.what() << '\n';

        return EXIT_FAILURE;
    }
}