#include "qwen3/model/kv_cache.h"
#include "qwen3/ops/gqa_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    std::uint16_t float_to_bfloat16(float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));

        const std::uint32_t lsb =
            (bits >> 16U) & 1U;

        bits += 0x7FFFU + lsb;

        return static_cast<std::uint16_t>(
            bits >> 16U);
    }

    float bfloat16_to_float(std::uint16_t value)
    {
        const std::uint32_t bits =
            static_cast<std::uint32_t>(value)
            << 16U;

        float result = 0.0F;
        std::memcpy(&result, &bits, sizeof(result));

        return result;
    }

    std::vector<std::uint16_t> make_values(
        std::size_t count,
        float scale,
        int offset)
    {
        std::vector<std::uint16_t> values(count);

        for (std::size_t index = 0;
             index < count;
             ++index)
        {
            const int integer =
                static_cast<int>(
                    (index + offset) % 11) -
                5;

            values[index] =
                float_to_bfloat16(
                    static_cast<float>(integer) *
                    scale);
        }

        return values;
    }

    qwen3::Tensor to_cuda_tensor(
        const qwen3::Shape &shape,
        const std::vector<std::uint16_t> &values)
    {
        auto host = qwen3::Tensor::empty(
            shape,
            qwen3::DType::BFloat16,
            qwen3::Device::cpu());

        std::memcpy(
            host.data(),
            values.data(),
            host.nbytes());

        return host.copy_to(
            qwen3::Device::cuda(0));
    }

    std::vector<std::uint16_t> combine_cache(
        const std::vector<std::uint16_t> &history,
        const std::vector<std::uint16_t> &current,
        std::int64_t batch_size,
        std::int64_t current_length,
        std::int64_t projection)
    {
        const std::int64_t total_length =
            1 + current_length;

        std::vector<std::uint16_t> result(
            static_cast<std::size_t>(
                batch_size *
                total_length *
                projection));

        for (std::int64_t batch = 0;
             batch < batch_size;
             ++batch)
        {
            for (std::int64_t column = 0;
                 column < projection;
                 ++column)
            {
                result[
                    static_cast<std::size_t>(
                        (batch * total_length) *
                            projection +
                        column)] =
                    history[
                        static_cast<std::size_t>(
                            batch * projection +
                            column)];
            }

            for (std::int64_t token = 0;
                 token < current_length;
                 ++token)
            {
                for (std::int64_t column = 0;
                     column < projection;
                     ++column)
                {
                    result[
                        static_cast<std::size_t>(
                            (batch * total_length +
                             token + 1) *
                                projection +
                            column)] =
                        current[
                            static_cast<std::size_t>(
                                (batch *
                                     current_length +
                                 token) *
                                    projection +
                                column)];
                }
            }
        }

        return result;
    }

    std::vector<std::uint16_t> attention_reference(
        const std::vector<std::uint16_t> &query,
        const std::vector<std::uint16_t> &keys,
        const std::vector<std::uint16_t> &values,
        std::int64_t batch_size,
        std::int64_t sequence_length,
        std::int64_t cached_length,
        std::int32_t query_heads,
        std::int32_t kv_heads,
        std::int64_t head_dim,
        std::int64_t start_position)
    {
        std::vector<std::uint16_t> output(
            query.size());

        const std::int32_t heads_per_kv =
            query_heads / kv_heads;

        const float scale =
            1.0F /
            std::sqrt(
                static_cast<float>(head_dim));

        for (std::int64_t batch = 0;
             batch < batch_size;
             ++batch)
        {
            for (std::int64_t query_position = 0;
                 query_position < sequence_length;
                 ++query_position)
            {
                const std::int64_t visible =
                    start_position +
                    query_position +
                    1;

                for (std::int32_t query_head = 0;
                     query_head < query_heads;
                     ++query_head)
                {
                    const std::int32_t kv_head =
                        query_head / heads_per_kv;

                    std::vector<float> scores(
                        static_cast<std::size_t>(
                            visible));

                    float maximum =
                        -std::numeric_limits<float>::
                            infinity();

                    for (std::int64_t key_position = 0;
                         key_position < visible;
                         ++key_position)
                    {
                        float score = 0.0F;

                        for (std::int64_t dimension = 0;
                             dimension < head_dim;
                             ++dimension)
                        {
                            const std::size_t query_index =
                                static_cast<std::size_t>(
                                    ((batch *
                                          sequence_length +
                                      query_position) *
                                         query_heads +
                                     query_head) *
                                        head_dim +
                                    dimension);

                            const std::size_t key_index =
                                static_cast<std::size_t>(
                                    ((batch *
                                          cached_length +
                                      key_position) *
                                         kv_heads +
                                     kv_head) *
                                        head_dim +
                                    dimension);

                            score +=
                                bfloat16_to_float(
                                    query[query_index]) *
                                bfloat16_to_float(
                                    keys[key_index]);
                        }

                        score *= scale;

                        scores[
                            static_cast<std::size_t>(
                                key_position)] =
                            score;

                        maximum =
                            std::max(maximum, score);
                    }

                    float denominator = 0.0F;

                    for (float &score : scores)
                    {
                        score =
                            std::exp(
                                score - maximum);

                        denominator += score;
                    }

                    for (std::int64_t dimension = 0;
                         dimension < head_dim;
                         ++dimension)
                    {
                        float result = 0.0F;

                        for (std::int64_t key_position = 0;
                             key_position < visible;
                             ++key_position)
                        {
                            const std::size_t value_index =
                                static_cast<std::size_t>(
                                    ((batch *
                                          cached_length +
                                      key_position) *
                                         kv_heads +
                                     kv_head) *
                                        head_dim +
                                    dimension);

                            result +=
                                scores[
                                    static_cast<std::size_t>(
                                        key_position)] /
                                denominator *
                                bfloat16_to_float(
                                    values[value_index]);
                        }

                        const std::size_t output_index =
                            static_cast<std::size_t>(
                                ((batch *
                                      sequence_length +
                                  query_position) *
                                     query_heads +
                                 query_head) *
                                    head_dim +
                                dimension);

                        output[output_index] =
                            float_to_bfloat16(result);
                    }
                }
            }
        }

        return output;
    }

    void expect_close(
        const qwen3::Tensor &actual,
        const std::vector<std::uint16_t> &expected)
    {
        const auto *actual_values =
            static_cast<const std::uint16_t *>(
                actual.data());

        for (std::size_t index = 0;
             index < expected.size();
             ++index)
        {
            const float actual_float =
                bfloat16_to_float(
                    actual_values[index]);

            const float expected_float =
                bfloat16_to_float(
                    expected[index]);

            if (std::fabs(
                    actual_float -
                    expected_float) >
                0.04F)
            {
                throw std::runtime_error(
                    "GQA output mismatch at index " +
                    std::to_string(index));
            }
        }
    }
}

int main()
{
    try
    {
        constexpr std::int64_t batch_size = 2;
        constexpr std::int64_t sequence_length = 2;
        constexpr std::int64_t start_position = 1;
        constexpr std::int64_t cached_length = 3;

        constexpr std::int32_t query_heads = 4;
        constexpr std::int32_t kv_heads = 2;
        constexpr std::int32_t head_dim = 4;

        constexpr std::int64_t query_projection =
            query_heads * head_dim;

        constexpr std::int64_t kv_projection =
            kv_heads * head_dim;

        qwen3::KvCache cache{
            1,
            batch_size,
            4,
            kv_heads,
            head_dim,
            qwen3::DType::BFloat16,
            qwen3::Device::cuda(0)};

        const qwen3::Shape history_shape{
            batch_size,
            1,
            kv_projection};

        const qwen3::Shape current_shape{
            batch_size,
            sequence_length,
            kv_projection};

        const qwen3::Shape query_shape{
            batch_size,
            sequence_length,
            query_projection};

        auto history_keys =
            make_values(
                static_cast<std::size_t>(
                    history_shape.numel()),
                0.125F,
                1);

        auto history_values =
            make_values(
                static_cast<std::size_t>(
                    history_shape.numel()),
                0.25F,
                2);

        auto current_keys =
            make_values(
                static_cast<std::size_t>(
                    current_shape.numel()),
                0.125F,
                3);

        auto current_values =
            make_values(
                static_cast<std::size_t>(
                    current_shape.numel()),
                0.25F,
                4);

        auto query_values =
            make_values(
                static_cast<std::size_t>(
                    query_shape.numel()),
                0.125F,
                5);

        auto history_key_tensor =
            to_cuda_tensor(
                history_shape,
                history_keys);

        auto history_value_tensor =
            to_cuda_tensor(
                history_shape,
                history_values);

        auto current_key_tensor =
            to_cuda_tensor(
                current_shape,
                current_keys);

        auto current_value_tensor =
            to_cuda_tensor(
                current_shape,
                current_values);

        auto query_tensor =
            to_cuda_tensor(
                query_shape,
                query_values);

        cache.append(
            0,
            0,
            history_key_tensor,
            history_value_tensor);

        cache.append(
            0,
            start_position,
            current_key_tensor,
            current_value_tensor);

        const auto combined_keys =
            combine_cache(
                history_keys,
                current_keys,
                batch_size,
                sequence_length,
                kv_projection);

        const auto combined_values =
            combine_cache(
                history_values,
                current_values,
                batch_size,
                sequence_length,
                kv_projection);

        const auto expected =
            attention_reference(
                query_values,
                combined_keys,
                combined_values,
                batch_size,
                sequence_length,
                cached_length,
                query_heads,
                kv_heads,
                head_dim,
                start_position);

        auto key_view = cache.key_view(0);
        auto value_view = cache.value_view(0);

        auto output =
            qwen3::ops::grouped_query_attention(
                query_tensor,
                key_view,
                value_view,
                query_heads,
                start_position);

        if (output.shape() != query_shape)
        {
            throw std::runtime_error(
                "GQA output shape is incorrect");
        }

        auto host_output =
            output.copy_to(
                qwen3::Device::cpu());

        expect_close(
            host_output,
            expected);

        std::cout
            << "GQA attention CUDA test passed.\n";
        std::cout
            << "output shape: "
            << output.shape().to_string()
            << '\n';

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "GQA attention CUDA test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}