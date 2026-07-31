#include "qwen3/ops/rope.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
    std::uint16_t float_to_bfloat16(float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));

        // round-to-nearest-even
        const std::uint32_t least_significant_bit =
            (bits >> 16U) & 1U;

        bits += 0x7FFFU + least_significant_bit;

        return static_cast<std::uint16_t>(
            bits >> 16U);
    }

    float bfloat16_to_float(std::uint16_t value)
    {
        const std::uint32_t bits =
            static_cast<std::uint32_t>(value) << 16U;

        float result = 0.0F;
        std::memcpy(&result, &bits, sizeof(result));

        return result;
    }

    std::vector<std::uint16_t> make_values(
        std::size_t element_count,
        int offset)
    {
        std::vector<std::uint16_t> values(element_count);

        for (std::size_t index = 0;
             index < element_count;
             ++index)
        {
            const int integer_value =
                static_cast<int>((index + offset) % 17) - 8;

            const float value =
                static_cast<float>(integer_value) * 0.25F;

            values[index] = float_to_bfloat16(value);
        }

        return values;
    }

    std::vector<std::uint16_t> rotate_reference(
        std::vector<std::uint16_t> values,
        std::int64_t batch_size,
        std::int64_t sequence_length,
        std::int32_t num_heads,
        std::int64_t head_dim,
        std::int64_t start_position,
        float rope_theta)
    {
        const std::int64_t half_dim = head_dim / 2;

        for (std::int64_t batch = 0;
             batch < batch_size;
             ++batch)
        {
            for (std::int64_t token = 0;
                 token < sequence_length;
                 ++token)
            {
                const std::int64_t position =
                    start_position + token;

                for (std::int32_t head = 0;
                     head < num_heads;
                     ++head)
                {
                    const std::int64_t head_offset =
                        ((batch * sequence_length + token) *
                             num_heads +
                         head) *
                        head_dim;

                    for (std::int64_t pair = 0;
                         pair < half_dim;
                         ++pair)
                    {
                        const double exponent =
                            (2.0 * static_cast<double>(pair)) /
                            static_cast<double>(head_dim);

                        const double inverse_frequency =
                            std::pow(
                                static_cast<double>(rope_theta),
                                -exponent);

                        const double angle =
                            static_cast<double>(position) *
                            inverse_frequency;

                        const float cosine =
                            static_cast<float>(std::cos(angle));

                        const float sine =
                            static_cast<float>(std::sin(angle));

                        const std::size_t first_offset =
                            static_cast<std::size_t>(
                                head_offset + pair);

                        const std::size_t second_offset =
                            static_cast<std::size_t>(
                                head_offset +
                                half_dim +
                                pair);

                        const float first =
                            bfloat16_to_float(
                                values[first_offset]);

                        const float second =
                            bfloat16_to_float(
                                values[second_offset]);

                        values[first_offset] =
                            float_to_bfloat16(
                                first * cosine -
                                second * sine);

                        values[second_offset] =
                            float_to_bfloat16(
                                second * cosine +
                                first * sine);
                    }
                }
            }
        }

        return values;
    }

    void copy_values(
        qwen3::Tensor &tensor,
        const std::vector<std::uint16_t> &values)
    {
        if (tensor.nbytes() !=
            values.size() * sizeof(std::uint16_t))
        {
            throw std::runtime_error(
                "test input size mismatch");
        }

        std::memcpy(
            tensor.data(),
            values.data(),
            tensor.nbytes());
    }

    void expect_close(
        const qwen3::Tensor &actual_tensor,
        const std::vector<std::uint16_t> &expected,
        float tolerance,
        const char *tensor_name)
    {
        const auto *actual =
            static_cast<const std::uint16_t *>(
                actual_tensor.data());

        for (std::size_t index = 0;
             index < expected.size();
             ++index)
        {
            const float actual_value =
                bfloat16_to_float(actual[index]);

            const float expected_value =
                bfloat16_to_float(expected[index]);

            const float difference =
                std::fabs(
                    actual_value - expected_value);

            if (difference > tolerance)
            {
                throw std::runtime_error(
                    std::string(tensor_name) +
                    " mismatch at index " +
                    std::to_string(index) +
                    ": actual=" +
                    std::to_string(actual_value) +
                    ", expected=" +
                    std::to_string(expected_value));
            }
        }
    }
}

int main()
{
    try
    {
        constexpr std::int64_t batch_size = 1;
        constexpr std::int64_t sequence_length = 3;

        constexpr std::int32_t query_heads = 2;
        constexpr std::int32_t key_value_heads = 1;

        constexpr std::int64_t head_dim = 8;
        constexpr std::int64_t max_positions = 16;
        constexpr std::int64_t start_position = 5;

        constexpr float rope_theta = 1'000'000.0F;

        const qwen3::Shape query_shape{
            batch_size,
            sequence_length,
            query_heads * head_dim};

        const qwen3::Shape key_shape{
            batch_size,
            sequence_length,
            key_value_heads * head_dim};

        auto query_values = make_values(
            static_cast<std::size_t>(
                query_shape.numel()),
            0);

        auto key_values = make_values(
            static_cast<std::size_t>(
                key_shape.numel()),
            5);

        const auto expected_query =
            rotate_reference(
                query_values,
                batch_size,
                sequence_length,
                query_heads,
                head_dim,
                start_position,
                rope_theta);

        const auto expected_key =
            rotate_reference(
                key_values,
                batch_size,
                sequence_length,
                key_value_heads,
                head_dim,
                start_position,
                rope_theta);

        auto host_query = qwen3::Tensor::empty(
            query_shape,
            qwen3::DType::BFloat16,
            qwen3::Device::cpu());

        auto host_key = qwen3::Tensor::empty(
            key_shape,
            qwen3::DType::BFloat16,
            qwen3::Device::cpu());

        copy_values(host_query, query_values);
        copy_values(host_key, key_values);

        const auto cuda_device =
            qwen3::Device::cuda(0);

        auto query =
            host_query.copy_to(cuda_device);

        auto key =
            host_key.copy_to(cuda_device);

        auto rope_cache =
            qwen3::ops::build_rope_cache(
                max_positions,
                static_cast<std::int32_t>(head_dim),
                rope_theta,
                cuda_device);

        if (rope_cache.shape() !=
            qwen3::Shape{
                max_positions,
                head_dim / 2,
                2})
        {
            throw std::runtime_error(
                "RoPE cache shape is incorrect");
        }

        void *query_pointer_before = query.data();
        void *key_pointer_before = key.data();

        qwen3::ops::apply_rope_inplace(
            query,
            key,
            rope_cache,
            query_heads,
            key_value_heads,
            start_position);

        if (query.data() != query_pointer_before ||
            key.data() != key_pointer_before)
        {
            throw std::runtime_error(
                "RoPE operation was not in-place");
        }

        auto actual_query =
            query.copy_to(qwen3::Device::cpu());

        auto actual_key =
            key.copy_to(qwen3::Device::cpu());

        expect_close(
            actual_query,
            expected_query,
            0.03F,
            "query");

        expect_close(
            actual_key,
            expected_key,
            0.03F,
            "key");

        bool range_error_thrown = false;

        try
        {
            qwen3::ops::apply_rope_inplace(
                query,
                key,
                rope_cache,
                query_heads,
                key_value_heads,
                max_positions - 1);
        }
        catch (const std::out_of_range &)
        {
            range_error_thrown = true;
        }

        if (!range_error_thrown)
        {
            throw std::runtime_error(
                "RoPE did not reject an invalid position range");
        }

        std::cout << "RoPE CUDA test passed.\n";
        std::cout << "cache shape: "
                  << rope_cache.shape().to_string()
                  << '\n';
        std::cout << "start position: "
                  << start_position
                  << '\n';

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "RoPE CUDA test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}