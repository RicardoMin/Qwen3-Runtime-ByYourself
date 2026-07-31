#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cublas_handle.h"
#include "qwen3/cuda/cuda_check.h"
#include "qwen3/ops/linear.h"

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

        constexpr std::int64_t batch_size = 2;
        constexpr std::int64_t sequence_length = 4;
        constexpr std::int64_t input_features = 16;
        constexpr std::int64_t output_features = 24;

        constexpr std::int64_t row_count =
            batch_size * sequence_length;

        qwen3::Tensor cpu_input =
            qwen3::Tensor::empty(
                qwen3::Shape{
                    batch_size,
                    sequence_length,
                    input_features},
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        qwen3::Tensor cpu_weight =
            qwen3::Tensor::empty(
                qwen3::Shape{
                    output_features,
                    input_features},
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        auto *input_values =
            static_cast<std::uint16_t *>(
                cpu_input.data());

        auto *weight_values =
            static_cast<std::uint16_t *>(
                cpu_weight.data());

        for (std::int64_t index = 0;
             index < cpu_input.numel();
             ++index)
        {
            const float value =
                static_cast<float>(
                    static_cast<int>(
                        index % 19) -
                    9) *
                0.0625F;

            input_values[index] =
                float_to_bfloat16(value);
        }

        for (std::int64_t index = 0;
             index < cpu_weight.numel();
             ++index)
        {
            const float value =
                static_cast<float>(
                    static_cast<int>(
                        index % 13) -
                    6) *
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

        qwen3::cuda::CublasHandle handle{
            qwen3::Device::cuda(0)};

        qwen3::Tensor cuda_output =
            qwen3::ops::linear(
                cuda_input,
                cuda_weight,
                handle);

        require(
            cuda_output.shape() ==
                qwen3::Shape{
                    batch_size,
                    sequence_length,
                    output_features},
            "linear output shape is incorrect");

        require(
            cuda_output.dtype() ==
                qwen3::DType::BFloat16,
            "linear output dtype is incorrect");

        require(
            cuda_output.device() ==
                qwen3::Device::cuda(0),
            "linear output device is incorrect");

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
            for (std::int64_t output_column = 0;
                 output_column < output_features;
                 ++output_column)
            {
                float expected = 0.0F;

                for (std::int64_t input_column = 0;
                     input_column < input_features;
                     ++input_column)
                {
                    const float input_value =
                        bfloat16_to_float(
                            input_values[
                                row * input_features +
                                input_column]);

                    const float weight_value =
                        bfloat16_to_float(
                            weight_values[
                                output_column *
                                    input_features +
                                input_column]);

                    expected +=
                        input_value *
                        weight_value;
                }

                expected =
                    quantize_bfloat16(
                        expected);

                const float actual =
                    bfloat16_to_float(
                        output_values[
                            row * output_features +
                            output_column]);

                const float tolerance =
                    0.02F +
                    0.01F *
                        std::fabs(expected);

                require(
                    std::fabs(
                        actual -
                        expected) <= tolerance,
                    "linear numerical result "
                    "is incorrect");
            }
        }

        int device_after = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDevice(
                &device_after));

        require(
            device_after == device_before,
            "linear operation changed current "
            "CUDA device");

        std::cout
            << "CUDA Linear test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "CUDA Linear test failed: "
            << error.what() << '\n';

        return EXIT_FAILURE;
    }
}