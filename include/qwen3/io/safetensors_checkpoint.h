#pragma once

#include "qwen3/core/device.h"
#include "qwen3/core/tensor.h"
#include "qwen3/io/safetensors_metadata.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace qwen3
{
    class SafetensorsCheckpoint final
    {
    public:
        explicit SafetensorsCheckpoint(const std::filesystem::path &model_directory);

        SafetensorsCheckpoint(const SafetensorsCheckpoint &) = delete;

        SafetensorsCheckpoint& operator=(const SafetensorsCheckpoint&) = delete;

        SafetensorsCheckpoint(SafetensorsCheckpoint&&) = default;

        SafetensorsCheckpoint& operator=(SafetensorsCheckpoint&&) = default;

        [[nodiscard]]
        const std::filesystem::path &model_directory() const noexcept;

        [[nodiscard]]
        bool is_sharded() const noexcept;

        [[nodiscard]]
        std::size_t shard_count() const noexcept;

        [[nodiscard]]
        std::size_t tensor_count() const noexcept;

        [[nodiscard]]
        bool contains(std::string_view tensor_name) const;

        [[nodiscard]]
        const SafetensorsTensorInfo &tensor_info(std::string_view tensor_name) const;

        [[nodiscard]]
        Tensor load(std::string_view tensor_name, Device target_device) const;

        [[nodiscard]]
        std::vector<Tensor> load_many(const std::vector<std::string> &tensor_names, Device target_device) const;

    private:
        struct TensorLocation final
        {
            std::size_t shard_index;
            SafetensorsTensorInfo info;
        };

        void open_single_file(const std::filesystem::path &model_path);

        void open_sharded_index(const std::filesystem::path &index_path);

    private:
        std::filesystem::path model_directory_;

        bool is_sharded_ = false;

        std::vector<std::filesystem::path> shard_paths_;

        std::unordered_map<std::string, TensorLocation> tensors_;
    };
    
}   // namespace qwen3