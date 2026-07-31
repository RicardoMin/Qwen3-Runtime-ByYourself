#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cuda_check.h"
#include "qwen3/ops/elementwise.h"

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

        const qwen3::Shape shape{
            2, 3, 257};

        const std::int64_t element_count =
            shape.numel();

        qwen3::Tensor cpu_residual =
            qwen3::Tensor::empty(
                shape,
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        qwen3::Tensor cpu_delta =
            qwen3::Tensor::empty(
                shape,
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        qwen3::Tensor cpu_gate =
            qwen3::Tensor::empty(
                shape,
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        qwen3::Tensor cpu_up =
            qwen3::Tensor::empty(
                shape,
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        auto *residual_values =
            static_cast<std::uint16_t *>(
                cpu_residual.data());

        auto *delta_values =
            static_cast<std::uint16_t *>(
                cpu_delta.data());

        auto *gate_values =
            static_cast<std::uint16_t *>(
                cpu_gate.data());

        auto *up_values =
            static_cast<std::uint16_t *>(
                cpu_up.data());

        for (std::int64_t index = 0;
             index < element_count;
             ++index)
        {
            const float residual =
                static_cast<float>(
                    static_cast<int>(
                        index % 41) -
                    20) *
                0.0625F;

            const float delta =
                static_cast<float>(
                    static_cast<int>(
                        index % 17) -
                    8) *
                0.03125F;

            const float gate =
                static_cast<float>(
                    static_cast<int>(
                        index % 37) -
                    18) *
                0.25F;

            const float up =
                static_cast<float>(
                    static_cast<int>(
                        index % 23) -
                    11) *
                0.125F;

            residual_values[index] =
                float_to_bfloat16(residual);

            delta_values[index] =
                float_to_bfloat16(delta);

            gate_values[index] =
                float_to_bfloat16(gate);

            up_values[index] =
                float_to_bfloat16(up);
        }

        qwen3::Tensor cuda_residual =
            cpu_residual.copy_to(
                qwen3::Device::cuda(0));

        qwen3::Tensor cuda_delta =
            cpu_delta.copy_to(
                qwen3::Device::cuda(0));

        void *residual_pointer_before =
            cuda_residual.data();

        qwen3::ops::add_inplace(
            cuda_residual,
            cuda_delta);

        require(
            cuda_residual.data() ==
                residual_pointer_before,
            "add_inplace allocated new storage");

        qwen3::Tensor added_cpu =
            cuda_residual.copy_to(
                qwen3::Device::cpu());

        const auto *added_values =
            static_cast<const std::uint16_t *>(
                added_cpu.data());

        for (std::int64_t index = 0;
             index < element_count;
             ++index)
        {
            const float lhs =
                bfloat16_to_float(
                    residual_values[index]);

            const float rhs =
                bfloat16_to_float(
                    delta_values[index]);

            const float expected =
                quantize_bfloat16(
                    lhs + rhs);

            const float actual =
                bfloat16_to_float(
                    added_values[index]);

            require(
                std::fabs(actual - expected) <
                    0.001F,
                "add_inplace result is incorrect");
        }

        qwen3::Tensor cuda_gate =
            cpu_gate.copy_to(
                qwen3::Device::cuda(0));

        qwen3::Tensor cuda_up =
            cpu_up.copy_to(
                qwen3::Device::cuda(0));

        qwen3::Tensor cuda_swiglu =
            qwen3::ops::silu_multiply(
                cuda_gate,
                cuda_up);

        require(
            cuda_swiglu.shape() == shape,
            "silu_multiply changed shape");

        require(
            cuda_swiglu.dtype() ==
                qwen3::DType::BFloat16,
            "silu_multiply returned wrong dtype");

        qwen3::Tensor swiglu_cpu =
            cuda_swiglu.copy_to(
                qwen3::Device::cpu());

        const auto *swiglu_values =
            static_cast<const std::uint16_t *>(
                swiglu_cpu.data());

        for (std::int64_t index = 0;
             index < element_count;
             ++index)
        {
            const float gate =
                bfloat16_to_float(
                    gate_values[index]);

            const float up =
                bfloat16_to_float(
                    up_values[index]);

            const float silu =
                gate /
                (1.0F + std::exp(-gate));

            const float rounded_silu =
                quantize_bfloat16(silu);

            const float expected =
                quantize_bfloat16(
                    rounded_silu * up);

            const float actual =
                bfloat16_to_float(
                    swiglu_values[index]);

            const float tolerance =
                0.02F +
                0.01F *
                    std::fabs(expected);

            require(
                std::fabs(
                    actual -
                    expected) <= tolerance,
                "silu_multiply result is incorrect");
        }

        int device_after = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDevice(
                &device_after));

        require(
            device_after == device_before,
            "elementwise operation changed "
            "current CUDA device");

        std::cout
            << "CUDA elementwise test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "CUDA elementwise test failed: "
            << error.what() << '\n';

        return EXIT_FAILURE;
    }
}