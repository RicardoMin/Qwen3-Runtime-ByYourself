#include "qwen3/io/safetensors_loader.h"

#include "qwen3/cuda/cuda_check.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen3
{
    SafetensorsLoader::SafetensorsLoader(const std::filesystem::path &model_path)
        : file_(model_path)
        , metadata_(file_)
    {
    }

    SafetensorsLoader::~SafetensorsLoader()
    {
        release_transfer_resources();
    }

    const SafetensorsMetadata &SafetensorsLoader::metadata() const noexcept
    {
        return metadata_;
    }

    const std::byte *SafetensorsLoader::source_data(const SafetensorsTensorInfo &info) const noexcept
    {
        return file_.data_bytes().data() + info.begin_offset;
    }

    Tensor SafetensorsLoader::load(std::string_view tensor_name, Device target_device) const
    {
        const SafetensorsTensorInfo &info = metadata_.tensor(tensor_name);

        Tensor result = Tensor::empty(info.shape, info.dtype, target_device);

        if (result.nbytes() != info.size_bytes()) {
            throw std::logic_error("Tensor size does not match metadata for '" + std::string(tensor_name) + "'");
        }

        if (info.size_bytes() == 0) {
            return result;
        }

        const std::byte *source = source_data(info);

        if (target_device.is_cpu()) 
        {
            std::memcpy(result.data(), source, info.size_bytes());
            return result;
        }
        else if(target_device.is_cuda()) 
        {
            QWEN3_CUDA_CHECK(cudaSetDevice(target_device.index()));

            // 单权重接口保持简单同步
            QWEN3_CUDA_CHECK(cudaMemcpy(result.data(), source, info.size_bytes(), cudaMemcpyHostToDevice));

            return result;
        }

        throw std::invalid_argument("unsupport target device");
    }

    void SafetensorsLoader::ensure_transfer_resources(std::int32_t cuda_device_index) 
    {
        // 已经为相同设备创建，直接复用
        if (transfer_stream_ != nullptr && transfer_device_index_ == cuda_device_index) {
            return;
        }

        // 如果此前绑定的是其他 GPU，先释放
        if (transfer_stream_ != nullptr) {
            release_transfer_resources();
        }

        transfer_device_index_ = cuda_device_index;

        try 
        {
            QWEN3_CUDA_CHECK(cudaSetDevice(cuda_device_index));

            QWEN3_CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_stream_, cudaStreamNonBlocking));

            for (std::size_t index = 0; index < kStagingSlotCount; ++index)
            {
                QWEN3_CUDA_CHECK(cudaHostAlloc(&staging_buffers_[index], kStagingBufferBytes, cudaHostAllocDefault));

                QWEN3_CUDA_CHECK(cudaEventCreateWithFlags(&staging_events_[index], cudaEventDisableTiming));
            }
        }
        catch (...)
        {
            release_transfer_resources();
            throw;
        }
    }

    void SafetensorsLoader::release_transfer_resources() noexcept
    {
        if (transfer_device_index_ >= 0) {
            // 析构函数中不能抛异常，因此忽略返回值
            cudaSetDevice(transfer_device_index_);
        }

        if (transfer_stream_ != nullptr) {
            // 确保没有 DMA 仍在读取 staging buffer
            cudaStreamSynchronize(transfer_stream_);
        }

        for (cudaEvent_t &event : staging_events_) 
        {
            if (event != nullptr) 
            {
                cudaEventDestroy(event);
                event = nullptr;
            }
        }

        for (void *&buffer : staging_buffers_) 
        {
            if (buffer != nullptr) {
                cudaFreeHost(buffer);
                buffer = nullptr;
            }
        }

        if (transfer_stream_ != nullptr) {
            cudaStreamDestroy(transfer_stream_);
            transfer_stream_ = nullptr;
        }

        transfer_device_index_ = -1;
    }

    std::vector<Tensor> SafetensorsLoader::load_many(const std::vector<std::string> &tensor_names, Device target_device)
    {
        std::vector<Tensor> results;
        results.reserve(tensor_names.size());

        if (tensor_names.empty()) {
            return results;
        }

        // CPU 路径不需要 pinned memory 和 stream
        if (target_device.is_cpu()) 
        {
            for (const std::string &name : tensor_names) {
                results.push_back(load(name, target_device));
            }

            return results;
        }
        if (!target_device.is_cuda()) {
            throw std::invalid_argument("unsupported target device");
        }

        QWEN3_CUDA_CHECK(cudaSetDevice(target_device.index()));

        ensure_transfer_resources(target_device.index());

        // 保存 metadata 指针，避免后面再次按名称查询
        std::vector<const SafetensorsTensorInfo *> infos;
        infos.reserve(tensor_names.size());

        // 先分配全部目标 Tensor
        // 这样不会在异步传输过程中穿插 cudaMalloc
        for (const std::string &name : tensor_names)
        {
            const SafetensorsTensorInfo &info = metadata_.tensor(name);

            infos.push_back(&info);

            results.emplace_back(Tensor::empty(info.shape, info.dtype, target_device));

            if (results.back().nbytes() != info.size_bytes()) {
                throw std::logic_error("Tensor size does not match metadata for '" + name + "'");
            }
        }

        std::array<bool, kStagingSlotCount> slot_in_flight{{false, false}};

        std::size_t next_slot = 0;

        try
        {
            for (std::size_t tensor_index = 0; tensor_index < results.size(); ++tensor_index)
            {
                Tensor &destination = results[tensor_index];

                const SafetensorsTensorInfo &info = *infos[tensor_index];

                const std::byte *source = source_data(info);

                auto *destination_bytes = static_cast<std::byte *>(destination.data());

                std::size_t copied_bytes = 0;

                while (copied_bytes < info.size_bytes()) 
                {
                    const std::size_t slot = next_slot % kStagingSlotCount;

                    ++next_slot;

                    // 这个槽位上一次的异步复制还没完成的时候
                    // 不能覆盖它
                    if (slot_in_flight[slot]) {
                        QWEN3_CUDA_CHECK(cudaEventSynchronize(staging_events_[slot]));
                    }

                    const std::size_t remaining = info.size_bytes() - copied_bytes;

                    const std::size_t chunk_bytes = std::min(remaining, kStagingBufferBytes);

                    // mmap 普通内存 -> pinned staging buffer
                    std::memcpy(staging_buffers_[slot], source + copied_bytes, chunk_bytes);

                    // pinned staging buffer -> GPU
                    QWEN3_CUDA_CHECK(cudaMemcpyAsync(destination_bytes + copied_bytes, staging_buffers_[slot]
                                    , chunk_bytes, cudaMemcpyHostToDevice, transfer_stream_));

                    QWEN3_CUDA_CHECK(cudaEventRecord(staging_events_[slot], transfer_stream_));

                    slot_in_flight[slot] = true;
                    copied_bytes += chunk_bytes;
                }
            }

            // 整批任务只在最后同步一次
            QWEN3_CUDA_CHECK(cudaStreamSynchronize(transfer_stream_));
        }
        catch (...)
        {
            // results 析构释放显存之前
            // 必须保证异步复制已经停止
            cudaStreamSynchronize(transfer_stream_);

            throw;
        }

        return results;
        
    }
}   // namespace qwen3