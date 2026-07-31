#pragma once

#include "qwen3/core/device.h"
#include "qwen3/core/tensor.h"
#include "qwen3/io/safetensors_file.h"
#include "qwen3/io/safetensors_metadata.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace qwen3
{
    class SafetensorsLoader final
    {
    public:
        explicit SafetensorsLoader(const std::filesystem::path &model_path);

        ~SafetensorsLoader();

        SafetensorsLoader(const SafetensorsLoader &) = delete;
        SafetensorsLoader &operator=(const SafetensorsLoader &) = delete;

        SafetensorsLoader(SafetensorsLoader &&) = delete;

        SafetensorsLoader &operator=(SafetensorsLoader &&) = delete;

        [[nodiscard]]
        const SafetensorsMetadata &metadata() const noexcept;

        // 简单同步接口，适合调试和加载单个权重
        [[nodiscard]]
        Tensor load(std::string_view tensor_name, Device target_device) const;

        // 批量加载接口
        // CUDA 路径使用 pinned 双缓冲和异步复制
        // 返回时所有 Tensor 都已经传输完成
        [[nodiscard]]
        std::vector<Tensor> load_many(const std::vector<std::string> &tensor_names, Device target_device);

    private:
        void ensure_transfer_resources(std::int32_t cuda_device_index);

        void release_transfer_resources() noexcept;

        [[nodiscard]]
        const std::byte *source_data(const SafetensorsTensorInfo &info) const noexcept;

    private:
        static constexpr std::size_t kStagingSlotCount = 2;

        static constexpr std::size_t kStagingBufferBytes = 32ULL * 1024ULL * 1024ULL;

        SafetensorsFile file_;
        SafetensorsMetadata metadata_;

        // 仅第一次调用 CUDA load_many（）时创建
        cudaStream_t transfer_stream_ = nullptr;

        std::array<void *, kStagingSlotCount> staging_buffers_{{nullptr, nullptr}};

        std::array<cudaEvent_t, kStagingSlotCount> staging_events_{{nullptr, nullptr}};

        std::int32_t transfer_device_index_ = -1;
    };
}   // namespace qwen3