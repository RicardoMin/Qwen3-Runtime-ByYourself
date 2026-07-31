#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cuda_check.h"
#include "qwen3/ops/rms_norm.h"

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

    // Round to nearest, ties to even。
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

float quantize_bfloat16(float value)
{
    return bfloat16_to_float(
        float_to_bfloat16(value));
}

void run_rms_norm_case(
    const qwen3::Shape &shape,
    float epsilon)
{
    const std::size_t rank =
        shape.rank();

    require(
        rank > 0,
        "test shape cannot be scalar");

    const std::int64_t hidden_size =
        shape[rank - 1];

    const std::int64_t element_count =
        shape.numel();

    const std::int64_t row_count =
        element_count / hidden_size;

    qwen3::Tensor cpu_input =
        qwen3::Tensor::empty(
            shape,
            qwen3::DType::BFloat16,
            qwen3::Device::cpu());

    qwen3::Tensor cpu_weight =
        qwen3::Tensor::empty(
            qwen3::Shape{hidden_size},
            qwen3::DType::BFloat16,
            qwen3::Device::cpu());

    auto *input_values =
        static_cast<std::uint16_t *>(
            cpu_input.data());

    auto *weight_values =
        static_cast<std::uint16_t *>(
            cpu_weight.data());

    for (std::int64_t index = 0;
         index < element_count;
         ++index)
    {
        const int signed_value =
            static_cast<int>(
                index % 31) -
            15;

        const float value =
            static_cast<float>(
                signed_value) *
            0.125F;

        input_values[index] =
            float_to_bfloat16(value);
    }

    for (std::int64_t index = 0;
         index < hidden_size;
         ++index)
    {
        const float value =
            0.5F +
            static_cast<float>(
                index % 17) *
            0.03125F;

        weight_values[index] =
            float_to_bfloat16(value);
    }

    qwen3::Tensor cuda_input =
        cpu_input.copy_to(
            qwen3::Device::cuda(0));

    qwen3::Tensor cuda_weight =
        cpu_weight.copy_to(
            qwen3::Device::cuda(0));

    qwen3::Tensor cuda_output =
        qwen3::ops::rms_norm(
            cuda_input,
            cuda_weight,
            epsilon);

    require(
        cuda_output.shape() == shape,
        "RMSNorm changed input shape");

    require(
        cuda_output.dtype() ==
            qwen3::DType::BFloat16,
        "RMSNorm output dtype is incorrect");

    qwen3::Tensor cpu_output =
        cuda_output.copy_to(
            qwen3::Device::cpu());

    const auto *output_values =
        static_cast<const std::uint16_t *>(
            cpu_output.data());

    for (std::int64_t row = 0;
         row < row_count;
         ++row)
    {
        float square_sum = 0.0F;

        for (std::int64_t column = 0;
             column < hidden_size;
             ++column)
        {
            const float value =
                bfloat16_to_float(
                    input_values[
                        row * hidden_size +
                        column]);

            square_sum += value * value;
        }

        const float inverse_rms =
            1.0F /
            std::sqrt(
                square_sum /
                    static_cast<float>(
                        hidden_size) +
                epsilon);

        for (std::int64_t column = 0;
             column < hidden_size;
             ++column)
        {
            const float input_value =
                bfloat16_to_float(
                    input_values[
                        row * hidden_size +
                        column]);

            const float weight_value =
                bfloat16_to_float(
                    weight_values[column]);

            // 模拟 kernel 中间的 BF16 舍入。
            const float normalized =
                quantize_bfloat16(
                    input_value *
                    inverse_rms);

            const float expected =
                quantize_bfloat16(
                    normalized *
                    weight_value);

            const float actual =
                bfloat16_to_float(
                    output_values[
                        row * hidden_size +
                        column]);

            const float tolerance =
                0.02F +
                0.01F *
                    std::fabs(expected);

            require(
                std::fabs(
                    actual -
                    expected) <= tolerance,
                "RMSNorm numerical result "
                "is incorrect");
        }
    }
}

} // namespace

int main()
{
    try
    {
        int device_count = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDeviceCount(
                &device_count));

        require(
            device_count > 0,
            "no visible CUDA device");

        int device_before = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDevice(
                &device_before));

        constexpr float epsilon =
            1.0e-6F;

        // 模拟 Transformer 主干 RMSNorm。
        run_rms_norm_case(
            qwen3::Shape{2, 3, 1024},
            epsilon);

        // 模拟 Q/K 按 head_dim 做 RMSNorm。
        run_rms_norm_case(
            qwen3::Shape{2, 3, 4, 128},
            epsilon);

        int device_after = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDevice(
                &device_after));

        require(
            device_after == device_before,
            "RMSNorm changed current "
            "CUDA device");

        std::cout
            << "CUDA RMSNorm test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "CUDA RMSNorm test failed: "
            << error.what() << '\n';

        return EXIT_FAILURE;
    }
}