#pragma once

#include "qwen3/core/tensor.h"
#include "qwen3/core/tensor_view.h"

#include <cstdint>
#include <vector>

namespace qwen3
{
    class KvCache final
    {
    public:
        KvCache(std::int32_t num_layers, std::int64_t max_batch_size,
            std::int64_t max_sequence_length, std::int32_t num_key_value_heads,
            std::int32_t head_dim, DType dtype, Device device);
        
        ~KvCache() = default;

        KvCache(const KvCache &) = delete;
        KvCache &operator=(const KvCache &) = delete;

        KvCache(KvCache &&) noexcept = default;
        KvCache &operator=(KvCache &&) noexcept = default;

        // key / value 形状：
        // [batch_size, sequence_length, num_key_value_heads * head_dim]
        //
        // start_position 必须等于该层当前的 cached_length
        void append(std::int32_t layer_index, std::int64_t start_position,
            const Tensor &key, const Tensor &value);
        
        // 只重置逻辑长度，不清零显存
        void reset() noexcept;

        [[nodiscard]]
        std::int64_t cached_length(std::int32_t layer_index) const;

        [[nodiscard]]
        std::int64_t active_batch_size() const noexcept;

        // 返回：
        // [active_batch_size, cached_length, num_key_value_heads, head_dim]
        //
        // 当 batch > 1 且 cached_length < capacity 时
        // 该视图不是连续张量，需要使用 strides
        [[nodiscard]]
        TensorView key_view(std::int32_t layer_index);

        [[nodiscard]]
        TensorView value_view(std::int32_t layer_index);

        [[nodiscard]]
        const Tensor &key_storage() const noexcept;

        [[nodiscard]]
        const Tensor &value_storage() const noexcept;

        [[nodiscard]]
        std::int32_t num_layers() const noexcept;

        [[nodiscard]]
        std::int64_t max_batch_size() const noexcept;

        [[nodiscard]]
        std::int64_t max_sequence_length() const noexcept;

        [[nodiscard]]
        std::int32_t num_key_value_heads() const noexcept;

        [[nodiscard]]
        std::int32_t head_dim() const noexcept;

    private:
        void validate_layer_index(std::int32_t layer_index) const;

        [[nodiscard]]
        TensorView make_layer_view(Tensor &storage, std::int32_t layer_index);

    private:
        std::int32_t num_layers_ = 0;
        std::int64_t max_batch_size_ = 0;
        std::int64_t max_sequence_length_ = 0;
        std::int32_t num_key_value_heads_ = 0;
        std::int32_t head_dim_ = 0;

        DType dtype_ = DType::BFloat16;
        Device device_ = Device::cpu();

        Tensor key_storage_;
        Tensor value_storage_;

        // 每一层独立记录长度
        //
        // 因为一次 forward 会依次处理各层
        // layer 0 写 position 10 后，layer 1 依然需要写 position 10
        std::vector<std::int64_t> cached_lengths_;

        std::int64_t active_batch_size_ = 0;
    };
}   // namespace qwen3