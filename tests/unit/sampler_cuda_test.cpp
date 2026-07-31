#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cuda_check.h"
#include "qwen3/generation/sampler.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

    qwen3::Tensor make_cuda_logits(
        const std::vector<float> &values,
        std::int64_t batch_size,
        std::int64_t vocab_size)
    {
        require(
            static_cast<std::int64_t>(
                values.size()) ==
                batch_size * vocab_size,
            "logit value count is incorrect");

        qwen3::Tensor cpu =
            qwen3::Tensor::empty(
                qwen3::Shape{
                    batch_size,
                    vocab_size},
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        auto *destination =
            static_cast<std::uint16_t *>(
                cpu.data());

        for (std::size_t index = 0;
             index < values.size();
             ++index)
        {
            destination[index] =
                float_to_bfloat16(
                    values[index]);
        }

        return cpu.copy_to(
            qwen3::Device::cuda(0));
    }

    std::vector<std::int32_t>
    read_token_ids(
        const qwen3::Tensor &token_ids)
    {
        qwen3::Tensor cpu =
            token_ids.copy_to(
                qwen3::Device::cpu());

        const auto *values =
            static_cast<
                const std::int32_t *>(
                cpu.data());

        return std::vector<std::int32_t>(
            values,
            values + cpu.numel());
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

        // ==========================================
        // 1. Greedy 路径
        // ==========================================

        qwen3::Tensor greedy_logits =
            make_cuda_logits(
                {
                    -1.0F, 0.0F, 4.0F,
                    2.0F, -2.0F,

                    3.0F, 7.0F, 7.0F,
                    1.0F, 0.0F,
                },
                2,
                5);

        qwen3::generation::SamplingConfig
            greedy_config;

        greedy_config.mode =
            qwen3::generation::
                SamplingMode::Greedy;

        qwen3::generation::Sampler
            greedy_sampler{
                greedy_config};

        qwen3::Tensor greedy_output =
            greedy_sampler.sample(
                greedy_logits);

        require(
            greedy_output.shape() ==
                qwen3::Shape{2, 1},
            "greedy output shape "
            "is incorrect");

        require(
            greedy_output.dtype() ==
                qwen3::DType::Int32,
            "greedy output dtype "
            "is incorrect");

        require(
            greedy_output.device() ==
                qwen3::Device::cuda(0),
            "greedy output device "
            "is incorrect");

        const auto greedy_ids =
            read_token_ids(
                greedy_output);

        require(
            greedy_ids[0] == 2,
            "greedy batch 0 result "
            "is incorrect");

        // token 1 和 2 分数相同，
        // Argmax 应选择较小 token ID 1。
        require(
            greedy_ids[1] == 1,
            "greedy tie-breaking "
            "is incorrect");

        // ==========================================
        // 2. Sampling 路径，top_k=1
        //
        // 虽然进入随机采样分支，但只保留一个
        // 候选，因此结果必然等于 Argmax。
        // ==========================================

        qwen3::generation::SamplingConfig
            top_one_config;

        top_one_config.mode =
            qwen3::generation::
                SamplingMode::Sample;

        top_one_config.temperature = 0.6F;
        top_one_config.top_k = 1;
        top_one_config.top_p = 1.0F;
        top_one_config.seed = 122;

        qwen3::generation::Sampler
            top_one_sampler{
                top_one_config};

        const auto top_one_ids =
            read_token_ids(
                top_one_sampler.sample(
                    greedy_logits));
        
        printf("%d  %d\n", top_one_ids[0], top_one_ids[1]);
        require(
            top_one_ids[0] == 2 &&
                top_one_ids[1] == 1,
            "top_k=1 sampling result "
            "is incorrect");

        // ==========================================
        // 3. Top-k + Top-p + 随机数状态
        // ==========================================

        // Softmax 前三个候选分数：
        // token 0 = 2
        // token 1 = 2
        // token 2 = 1
        //
        // top_k=3 后仍保留 0、1、2。
        // top_p=0.8 会保留 token 0、1，
        // 但排除 token 2。
        qwen3::Tensor sampling_logits =
            make_cuda_logits(
                {
                    2.0F,
                    2.0F,
                    1.0F,
                    -10.0F,
                },
                1,
                4);

        qwen3::generation::SamplingConfig
            sampling_config;

        sampling_config.mode =
            qwen3::generation::
                SamplingMode::Sample;

        sampling_config.temperature = 1.0F;
        sampling_config.top_k = 3;
        sampling_config.top_p = 0.8F;
        sampling_config.seed = 42;

        qwen3::generation::Sampler sampler{
            sampling_config};

        std::vector<std::int32_t>
            first_sequence;

        constexpr int sample_count = 16;

        for (int index = 0;
             index < sample_count;
             ++index)
        {
            const auto sampled_ids =
                read_token_ids(
                    sampler.sample(
                        sampling_logits));

            require(
                sampled_ids[0] == 0 ||
                    sampled_ids[0] == 1,
                "top-p selected a filtered "
                "token");

            first_sequence.push_back(
                sampled_ids[0]);
        }

        // 使用相同 seed，应该重放相同序列。
        sampler.reseed(
            sampling_config.seed);

        for (int index = 0;
             index < sample_count;
             ++index)
        {
            const auto sampled_ids =
                read_token_ids(
                    sampler.sample(
                        sampling_logits));

            require(
                sampled_ids[0] ==
                    first_sequence[
                        static_cast<
                            std::size_t>(
                            index)],
                "reseed did not reproduce "
                "the sampling sequence");
        }

        int device_after = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDevice(
                &device_after));

        require(
            device_after == device_before,
            "sampler changed the current "
            "CUDA device");

        std::cout
            << "CUDA sampler test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "CUDA sampler test failed: "
            << error.what()
            << '\n';

        return EXIT_FAILURE;
    }
}