#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cuda_check.h"
#include "qwen3/ops/argmax.h"

#include <cuda_runtime_api.h>

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
            throw std::runtime_error(
                message);
        }
    }

    std::uint16_t float_to_bfloat16(
        float value)
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

        // 大于 256，并且不是 256 的整数倍，
        // 用于验证线程的跨步遍历。
        constexpr std::int64_t vocab_size = 513;

        qwen3::Tensor cpu_logits =
            qwen3::Tensor::empty(
                qwen3::Shape{
                    batch_size,
                    vocab_size},
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        auto *values =
            static_cast<std::uint16_t *>(
                cpu_logits.data());

        const std::uint16_t default_value =
            float_to_bfloat16(-8.0F);

        for (std::int64_t index = 0;
             index < cpu_logits.numel();
             ++index)
        {
            values[index] =
                default_value;
        }

        // 第 0 行最大值放在最后一个元素。
        values[512] =
            float_to_bfloat16(12.0F);

        // 第 1 行制造两个相同最大值。
        // 应返回较小索引 17。
        values[
            vocab_size + 17] =
            float_to_bfloat16(9.0F);

        values[
            vocab_size + 500] =
            float_to_bfloat16(9.0F);

        qwen3::Tensor cuda_logits =
            cpu_logits.copy_to(
                qwen3::Device::cuda(0));

        qwen3::Tensor compact_output =
            qwen3::ops::
                argmax_last_dimension(
                    cuda_logits,
                    false);

        qwen3::Tensor kept_output =
            qwen3::ops::
                argmax_last_dimension(
                    cuda_logits,
                    true);

        require(
            compact_output.shape() ==
                qwen3::Shape{2},
            "argmax compact output shape "
            "is incorrect");

        require(
            kept_output.shape() ==
                qwen3::Shape{2, 1},
            "argmax kept output shape "
            "is incorrect");

        require(
            kept_output.dtype() ==
                qwen3::DType::Int32,
            "argmax output dtype "
            "is incorrect");

        require(
            kept_output.device() ==
                qwen3::Device::cuda(0),
            "argmax output device "
            "is incorrect");

        qwen3::Tensor compact_cpu =
            compact_output.copy_to(
                qwen3::Device::cpu());

        qwen3::Tensor kept_cpu =
            kept_output.copy_to(
                qwen3::Device::cpu());

        const auto *compact_values =
            static_cast<
                const std::int32_t *>(
                compact_cpu.data());

        const auto *kept_values =
            static_cast<
                const std::int32_t *>(
                kept_cpu.data());

        constexpr std::int32_t expected[2] = {
            512,
            17
        };

        for (std::int64_t batch = 0;
             batch < batch_size;
             ++batch)
        {
            require(
                compact_values[batch] ==
                    expected[batch],
                "compact argmax result "
                "is incorrect");

            require(
                kept_values[batch] ==
                    expected[batch],
                "kept argmax result "
                "is incorrect");
        }

        int device_after = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDevice(
                &device_after));

        require(
            device_after == device_before,
            "argmax changed the current "
            "CUDA device");

        std::cout
            << "CUDA argmax test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "CUDA argmax test failed: "
            << error.what()
            << '\n';

        return EXIT_FAILURE;
    }
}