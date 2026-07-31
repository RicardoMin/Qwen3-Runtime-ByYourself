#include "qwen3/model/kv_cache.h"

#include "qwen3/cuda/cuda_check.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen3
{
    namespace
    {
        std::int32_t require_positive(std::int32_t value, const char *name)
        {
            if (value <= 0) {
                throw std::invalid_argument(std::string(name) + " must be positive");
            }

            return value;
        }

        std::int64_t require_positive(std::int64_t value, const char *name)
        {
            if (value <= 0) {
                throw std::invalid_argument(std::string(name) + " must be positive");
            }

            return value;
        }

        DType require_cache_dtype(DType dtype)
        {
            if (dtype != DType::BFloat16 && dtype != DType::Float32) {
                throw std::invalid_argument("KV cache supports only bfloat16 or float32");
            }

            return dtype;
        }

        Device require_cuda_device(Device device)
        {
            if (!device.is_cuda()) {
                throw std::invalid_argument("KV cache must be stored on CUDA");
            }

            return device;
        }
    }   // namespace

    KvCache::KvCache(std::int32_t num_layers, std::int64_t max_batch_size,
        std::int64_t max_sequence_length, std::int32_t num_key_value_heads,
        std::int32_t head_dim, DType dtype, Device device
    )
        : num_layers_(require_positive(num_layers, "num_layers"))
        , max_batch_size_(require_positive(max_batch_size, "max_batch_size"))
        , max_sequence_length_(require_positive(max_sequence_length, "max_sequence_length"))
        , num_key_value_heads_(require_positive(num_key_value_heads, "num_key_value_heads"))
        , head_dim_(require_positive(head_dim, "head_dim"))
        , dtype_(require_cache_dtype(dtype))
        , device_(require_cuda_device(device))
        , key_storage_(Tensor::empty(Shape{
            num_layers_, max_batch_size_, max_sequence_length_, num_key_value_heads_, head_dim
            }, dtype_, device_))
        , value_storage_(Tensor::empty(Shape{
            num_layers_, max_batch_size_, max_sequence_length_, num_key_value_heads_, head_dim
            }, dtype_, device_))
        ,cached_lengths_(static_cast<std::size_t>(num_layers_), 0)
    {
    }

    void KvCache::append(std::int32_t layer_index, std::int64_t start_position, const Tensor &key, const Tensor &value)
    {
        validate_layer_index(layer_index);

        if (key.dtype() != dtype_ || value.dtype() != dtype_) {
            throw std::invalid_argument("KV cache input dtype does not match cache dtype");
        }

        if (key.device() != device_ || value.device() != device_) {
            throw std::invalid_argument("KV cache inputs must be on the cache device");
        }

        if (key.shape().rank() != 3 || value.shape().rank() != 3) {
            throw std::invalid_argument("KV cache inputs must have rank 3");
        }

        if (key.shape() != value.shape()) {
            throw std::invalid_argument("KV cache key and value shape must match");
        }

        const std::int64_t batch_size = key.shape()[0];
        const std::int64_t sequence_length = key.shape()[1];

        const std::int64_t projection_size = static_cast<std::int64_t>(num_key_value_heads_) * head_dim_;

        if (batch_size <= 0 || sequence_length <= 0) {
            throw std::invalid_argument("KV cache input batch and sequence dimenisons must be positive");
        }

        if (batch_size > max_batch_size_) {
            throw std::out_of_range("KV cache batch size exceeds capacity");
        }

        if (key.shape()[2] != projection_size) {
            throw std::invalid_argument("KV cache projection size does not match num_key_value_heads * head_dim");
        }

        const auto layer_offset = static_cast<std::size_t>(layer_index);

        if (start_position != cached_lengths_[layer_offset]) {
            throw std::logic_error("KV cache append must start at the current cached length");
        }

        if (start_position < 0 || start_position > max_sequence_length_ ||
            sequence_length > max_sequence_length_ - start_position)
        {
            throw std::out_of_range("KV cache append exceeds sequence capacity");
        }

        if (active_batch_size_ != 0 && batch_size != active_batch_size_)
        {
            throw std::invalid_argument("KV cache batch size cannot change before reset"); 
        }

        const std::size_t element_size = dtype_size(dtype_);

        const std::size_t projection_bytes = static_cast<std::size_t>(projection_size) * element_size;

        // 输入的是连续的 [B, S, P]
        const std::size_t source_pitch = static_cast<std::size_t>(sequence_length) * projection_bytes;

        // 缓存每一个 batch row 的实际跨度是 max_sequence_length
        const std::size_t destination_pitch = static_cast<std::size_t>(max_sequence_length_) * projection_bytes;

        const std::size_t layer_bytes = static_cast<std::size_t>(max_batch_size_) * destination_pitch;

        const std::size_t destination_offset = layer_offset * layer_bytes + static_cast<std::size_t>(start_position) * projection_bytes;

        auto *key_destination = static_cast<std::byte *>(key_storage_.data()) + destination_offset;

        auto *value_destination = static_cast<std::byte *>(value_storage_.data()) + destination_offset;

        int previous_device = 0;

        QWEN3_CUDA_CHECK(cudaGetDevice(&previous_device));

        const bool changed_device = previous_device != device_.index();

        if (changed_device) {
            QWEN3_CUDA_CHECK(cudaSetDevice(device_.index()));
        }

        // 用二维异步拷贝处理 batch 之间的缓存空隙
        //
        // 因为当写入数据长度短于cache中的长度，写入数据是连续的，没办法实现处于不同cache之间
        // 所以使用二维拷贝，可以保证连续的输入内存被不连续的拷贝到不同cache之间，使中间的间歇正常
        //
        // width = 当前 S 个 token 的字节数
        // height = batch_size
        cudaError_t operation_status = cudaMemcpy2DAsync(
            key_destination,
            destination_pitch,
            key.data(),
            source_pitch,
            source_pitch,
            static_cast<std::size_t>(batch_size),
            cudaMemcpyDeviceToDevice,
            nullptr
        );

        if (operation_status == cudaSuccess) 
        {
            operation_status =
                cudaMemcpy2DAsync(
                    value_destination,
                    destination_pitch,
                    value.data(),
                    source_pitch,
                    source_pitch,
                    static_cast<std::size_t>(batch_size),
                    cudaMemcpyDeviceToDevice,
                    nullptr);
        }

        cudaError_t restore_status = cudaSuccess;

        if (changed_device) {
            restore_status = cudaSetDevice(previous_device);
        }

        QWEN3_CUDA_CHECK(operation_status);
        QWEN3_CUDA_CHECK(restore_status);

        if (active_batch_size_ == 0) {
            active_batch_size_ = batch_size;
        }

        cached_lengths_[layer_offset] = start_position + sequence_length;
    }

    void KvCache::reset() noexcept
    {
        std::fill(cached_lengths_.begin(), cached_lengths_.end(), 0);

        active_batch_size_ = 0;

        // 不需要 cudaMemset
        // reset 后逻辑长度为 0，旧数据不会再被读取
    }

    std::int64_t KvCache::cached_length(std::int32_t layer_index) const
    {
        validate_layer_index(layer_index);

        return cached_lengths_[static_cast<std::size_t>(layer_index)];
    }

    std::int64_t KvCache::active_batch_size() const noexcept
    {
        return active_batch_size_;
    }

    TensorView KvCache::key_view(std::int32_t layer_index)
    {
        return make_layer_view(key_storage_, layer_index);
    }

    TensorView KvCache::value_view(std::int32_t layer_index)
    {
        return make_layer_view(value_storage_, layer_index);
    }

    const Tensor &KvCache::key_storage() const noexcept
    {
        return key_storage_;
    }

    const Tensor &KvCache::value_storage() const noexcept
    {
        return value_storage_;
    }

    std::int32_t KvCache::num_layers() const noexcept
    {
        return num_layers_;
    }

    std::int64_t KvCache::max_batch_size() const noexcept
    {
        return max_batch_size_;
    }

    std::int64_t KvCache::max_sequence_length() const noexcept
    {
        return max_sequence_length_;
    }

    std::int32_t KvCache::num_key_value_heads() const noexcept
    {
        return num_key_value_heads_;
    }

    std::int32_t KvCache::head_dim() const noexcept
    {
        return head_dim_;
    }

    void KvCache::validate_layer_index(std::int32_t layer_index) const
    {
        if (layer_index < 0 || layer_index >= num_layers_) {
            throw std::out_of_range("KV cache layer index is out of range");
        }
    }

    TensorView KvCache::make_layer_view(Tensor &storage, std::int32_t layer_index)
    {
        validate_layer_index(layer_index);

        const std::int64_t projection_size = static_cast<std::int64_t>(num_key_value_heads_) * head_dim_;
        
        const std::int64_t batch_stride = max_sequence_length_ * projection_size;

        const std::int64_t sequence_stride = projection_size;

        const std::int64_t layer_elements = max_batch_size_ * max_sequence_length_ * projection_size;

        const std::size_t layer_byte_offset = static_cast<std::size_t>(layer_index * layer_elements) * dtype_size(dtype_);

        auto *layer_data = static_cast<std::byte *>(storage.data()) + layer_byte_offset;

        return TensorView{
            layer_data,
            dtype_,
            device_,
            Shape{
                active_batch_size_,
                cached_lengths_[static_cast<std::size_t>(layer_index)],
                num_key_value_heads_,
                head_dim_,
            },
            std::vector<std::int64_t>{batch_stride, sequence_stride, head_dim_, 1}
        };
    }
}   // namespace qwen3