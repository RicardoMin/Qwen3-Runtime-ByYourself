#include "qwen3/model/kv_cache.h"

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

        const std::uint32_t lsb =
            (bits >> 16U) & 1U;

        bits += 0x7FFFU + lsb;

        return static_cast<std::uint16_t>(
            bits >> 16U);
    }

    std::vector<std::uint16_t> make_values(
        std::size_t count,
        float start)
    {
        std::vector<std::uint16_t> values(count);

        for (std::size_t index = 0;
             index < count;
             ++index)
        {
            values[index] =
                float_to_bfloat16(
                    start +
                    static_cast<float>(index) * 0.25F);
        }

        return values;
    }

    qwen3::Tensor make_cuda_tensor(
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

    void verify_slice(
        const qwen3::Tensor &host_storage,
        std::int64_t max_batch,
        std::int64_t max_sequence,
        std::int32_t kv_heads,
        std::int32_t head_dim,
        std::int32_t layer,
        std::int64_t start_position,
        std::int64_t sequence_length,
        const std::vector<std::uint16_t> &expected)
    {
        const auto *actual =
            static_cast<const std::uint16_t *>(
                host_storage.data());

        const std::int64_t projection =
            static_cast<std::int64_t>(kv_heads) *
            head_dim;

        for (std::int64_t batch = 0;
             batch < max_batch;
             ++batch)
        {
            for (std::int64_t token = 0;
                 token < sequence_length;
                 ++token)
            {
                for (std::int64_t column = 0;
                     column < projection;
                     ++column)
                {
                    const std::int64_t destination_index =
                        (((static_cast<std::int64_t>(layer) *
                               max_batch +
                           batch) *
                              max_sequence +
                          start_position +
                          token) *
                             projection +
                         column);

                    const std::int64_t source_index =
                        (batch * sequence_length + token) *
                            projection +
                        column;

                    if (actual[destination_index] !=
                        expected[source_index])
                    {
                        throw std::runtime_error(
                            "KV cache data mismatch");
                    }
                }
            }
        }
    }
}

int main()
{
    try
    {
        constexpr std::int32_t layers = 2;
        constexpr std::int64_t max_batch = 2;
        constexpr std::int64_t max_sequence = 5;
        constexpr std::int32_t kv_heads = 2;
        constexpr std::int32_t head_dim = 4;

        constexpr std::int64_t projection =
            kv_heads * head_dim;

        qwen3::KvCache cache{
            layers,
            max_batch,
            max_sequence,
            kv_heads,
            head_dim,
            qwen3::DType::BFloat16,
            qwen3::Device::cuda(0)};

        const qwen3::Shape prefill_shape{
            max_batch,
            2,
            projection};

        const auto prefill_keys =
            make_values(
                static_cast<std::size_t>(
                    prefill_shape.numel()),
                1.0F);

        const auto prefill_values =
            make_values(
                static_cast<std::size_t>(
                    prefill_shape.numel()),
                21.0F);

        auto prefill_key_tensor =
            make_cuda_tensor(
                prefill_shape,
                prefill_keys);

        auto prefill_value_tensor =
            make_cuda_tensor(
                prefill_shape,
                prefill_values);

        // 两层都写入相同的 position 0..1。
        cache.append(
            0,
            0,
            prefill_key_tensor,
            prefill_value_tensor);

        cache.append(
            1,
            0,
            prefill_key_tensor,
            prefill_value_tensor);

        if (cache.cached_length(0) != 2 ||
            cache.cached_length(1) != 2)
        {
            throw std::runtime_error(
                "KV cache prefill length is incorrect");
        }

        const qwen3::Shape decode_shape{
            max_batch,
            1,
            projection};

        const auto decode_keys =
            make_values(
                static_cast<std::size_t>(
                    decode_shape.numel()),
                41.0F);

        const auto decode_values =
            make_values(
                static_cast<std::size_t>(
                    decode_shape.numel()),
                61.0F);

        auto decode_key_tensor =
            make_cuda_tensor(
                decode_shape,
                decode_keys);

        auto decode_value_tensor =
            make_cuda_tensor(
                decode_shape,
                decode_values);

        // 模拟生成一个新 token。
        cache.append(
            0,
            2,
            decode_key_tensor,
            decode_value_tensor);

        cache.append(
            1,
            2,
            decode_key_tensor,
            decode_value_tensor);

        if (cache.cached_length(0) != 3 ||
            cache.cached_length(1) != 3)
        {
            throw std::runtime_error(
                "KV cache decode length is incorrect");
        }

        auto key_view = cache.key_view(1);

        if (key_view.shape() !=
            qwen3::Shape{
                max_batch,
                3,
                kv_heads,
                head_dim})
        {
            throw std::runtime_error(
                "KV cache view shape is incorrect");
        }

        if (key_view.strides() !=
            std::vector<std::int64_t>{
                max_sequence * projection,
                projection,
                head_dim,
                1})
        {
            throw std::runtime_error(
                "KV cache view strides are incorrect");
        }

        auto host_keys =
            cache.key_storage().copy_to(
                qwen3::Device::cpu());

        auto host_values =
            cache.value_storage().copy_to(
                qwen3::Device::cpu());

        verify_slice(
            host_keys,
            max_batch,
            max_sequence,
            kv_heads,
            head_dim,
            1,
            0,
            2,
            prefill_keys);

        verify_slice(
            host_values,
            max_batch,
            max_sequence,
            kv_heads,
            head_dim,
            1,
            0,
            2,
            prefill_values);

        verify_slice(
            host_keys,
            max_batch,
            max_sequence,
            kv_heads,
            head_dim,
            1,
            2,
            1,
            decode_keys);

        verify_slice(
            host_values,
            max_batch,
            max_sequence,
            kv_heads,
            head_dim,
            1,
            2,
            1,
            decode_values);

        bool rejected_non_append = false;

        try
        {
            cache.append(
                0,
                1,
                decode_key_tensor,
                decode_value_tensor);
        }
        catch (const std::logic_error &)
        {
            rejected_non_append = true;
        }

        if (!rejected_non_append)
        {
            throw std::runtime_error(
                "KV cache accepted a non-append write");
        }

        cache.reset();

        if (cache.cached_length(0) != 0 ||
            cache.cached_length(1) != 0 ||
            cache.active_batch_size() != 0)
        {
            throw std::runtime_error(
                "KV cache reset failed");
        }

        std::cout << "KV cache CUDA test passed.\n";
        std::cout << "storage shape: "
                  << cache.key_storage()
                         .shape()
                         .to_string()
                  << '\n';

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "KV cache CUDA test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}