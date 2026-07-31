#include "qwen3/io/safetensors_checkpoint.h"

#include "qwen3/io/safetensors_loader.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qwen3
{
    namespace
    {
        using Json = nlohmann::json;

        Json read_index(const std::filesystem::path &index_path)
        {
            std::ifstream input{index_path};

            if (!input)
            {
                throw std::runtime_error("cannot open checkpoint index: " + index_path.string());
            }

            Json root;

            try
            {
                input >> root;
            }
            catch (const Json::exception &error)
            {
                throw std::runtime_error("invalid checkpoint index '" + index_path.string() + "': " + error.what());
            }

            if (!root.is_object())
            {
                throw std::runtime_error("checkpoint index root must be an object");
            }

            return root;
        }

        std::size_t read_total_size(const Json &root)
        {
            const auto metadata_iterator =
                root.find("metadata");

            if (metadata_iterator == root.end() ||
                !metadata_iterator->is_object())
            {
                throw std::runtime_error(
                    "checkpoint index must contain "
                    "a metadata object");
            }

            const auto total_size_iterator =
                metadata_iterator->find("total_size");

            if (total_size_iterator ==
                    metadata_iterator->end() ||
                !total_size_iterator->is_number_unsigned())
            {
                throw std::runtime_error(
                    "checkpoint metadata.total_size "
                    "must be an unsigned integer");
            }

            const std::uint64_t raw_total_size =
                total_size_iterator->get<std::uint64_t>();

            if (raw_total_size >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()))
            {
                throw std::runtime_error(
                    "checkpoint total_size is too large");
            }

            return static_cast<std::size_t>(
                raw_total_size);
        }

        std::filesystem::path checked_shard_path(const std::filesystem::path &model_directory, const std::string& shard_name)
        {
            if (shard_name.empty() || shard_name.find('/') != std::string::npos || shard_name.find('\\') != std::string::npos) {
                throw std::runtime_error("invalid checkpoint shard name: " + shard_name);
            }

            const std::filesystem::path relative{shard_name};

            if (relative.is_absolute() || relative.has_parent_path() || relative.extension() != ".safetensors") {
                throw std::runtime_error("invalid checkpoint shard name: " + shard_name);
            }

            const std::filesystem::path full_path = model_directory / relative;

            if (!std::filesystem::is_regular_file(full_path)) {
                throw std::runtime_error("checkpoint shard does not exist: " + full_path.string());
            }

            return full_path;
        }

        std::size_t checked_add(std::size_t lhs, std::size_t rhs)
        {
            if (rhs >
                std::numeric_limits<std::size_t>::max() -
                    lhs)
            {
                throw std::overflow_error(
                    "checkpoint tensor byte count overflow");
            }

            return lhs + rhs;
        }
    } // namespace

    SafetensorsCheckpoint::SafetensorsCheckpoint(const std::filesystem::path &model_directory)
        : model_directory_{std::filesystem::absolute(model_directory)}
    {
        if (!std::filesystem::is_directory(model_directory_)) {
            throw std::invalid_argument("model directory does not exist: " + model_directory_.string());
        }

        const std::filesystem::path single_path = model_directory_ / "model.safetensors";

        const std::filesystem::path index_path = model_directory_ / "model.safetensors.index.json";

        const bool has_single = std::filesystem::is_regular_file(single_path);

        const bool has_index = std::filesystem::is_regular_file(index_path);

        // 两者同时存在会产生歧义，因此主动拒绝
        if (has_single && has_index) {
            throw std::runtime_error("model directory contains both model.safetensors and model.safetensors.index.json");
        }

        if (!has_single && !has_index) {
            throw std::runtime_error("model directory contains neither model.safetensors ans model.safetensor.index.json");
        }

        if (has_single) {
            open_single_file(single_path);
        }else {
            open_sharded_index(index_path);
        }
    }

    void SafetensorsCheckpoint::open_single_file (const std::filesystem::path &model_path)
    {
        shard_paths_.push_back(model_path);

        // 只在当前作用域读取 metadata
        // 此处没有调用 CUDA Load_many
        // 所以不会分配 pinned staging buffer
        SafetensorsLoader loader{model_path};

        for (const auto& entry : loader.metadata().tensors()) {
            const std::string &name = entry.first;
            const SafetensorsTensorInfo &info = entry.second;

            const auto result = tensors_.emplace(name, TensorLocation{0, info});

            if (!result.second) {
                throw std::runtime_error("duplicate tensor in checkpoint: " + name);
            }
        }

        if (tensors_.empty()) {
            throw std::runtime_error("single-file checkpoint contains no tensors");
        }
    }

    void SafetensorsCheckpoint::open_sharded_index(const std::filesystem::path &index_path)
    {
        const Json root = read_index(index_path);
        
        const std::size_t expected_total_size = read_total_size(root);

        const auto weight_map_iterator = root.find("weight_map");

        if (weight_map_iterator == root.end() || !weight_map_iterator->is_object() || weight_map_iterator->empty())
        {
            throw std::runtime_error("checkpoint index weight_map must be a non-empty object");
        }

        const Json &weight_map = *weight_map_iterator;

        // shard 文件名 -> shard 下标
        std::unordered_map<std::string, std::size_t> shard_by_name;

        // tensor 名 -> index 指定的 shard 下标
        std::unordered_map<std::string, std::size_t> expected_shard;

        for (auto iterator = weight_map.begin(); iterator != weight_map.end(); ++iterator)
        {
            const std::string tensor_name = iterator.key();

            if (tensor_name.empty() || !iterator.value().is_string()) 
            {
                throw std::runtime_error("checkpoint weight_map contains an valid entry");
            }

            const std::string shard_name = iterator.value().get<std::string>();

            auto shard_iterator = shard_by_name.find(shard_name);

            std::size_t shard_index = 0;
            if (shard_iterator == shard_by_name.end())
            {
                shard_index = shard_paths_.size();

                shard_paths_.push_back(checked_shard_path(model_directory_, shard_name));
                
                shard_by_name.emplace(shard_name, shard_index);
            }else {
                shard_index = shard_iterator->second;
            }

            const auto inserted = expected_shard.emplace(tensor_name, shard_index);

            if (!inserted.second) {
                throw std::runtime_error("duplicate tensor in checkpoint index: " + tensor_name);
            }
        }

        std::size_t actual_total_size = 0;

        // 逐个打开 shard，检查真实 metadata
        // 与 index.json 是否双向一致
        for (std::size_t shard_index = 0; shard_index < shard_paths_.size(); ++shard_index)
        {
            SafetensorsLoader loader{shard_paths_[shard_index]};

            for (const auto &entry : loader.metadata().tensors())
            {
                const std::string &name = entry.first;

                const SafetensorsTensorInfo &info = entry.second;

                actual_total_size = checked_add(actual_total_size, info.size_bytes());

                const auto expected_iterator = expected_shard.find(name);

                if (expected_iterator == expected_shard.end()) {
                    throw std::runtime_error("tensor '" + name + "' exists in a shard but not in the checkpoint index");
                }

                if (expected_iterator->second != shard_index) {
                    throw std::runtime_error("tensor '" + name + "' is stored in the wrong shard");
                }

                const auto inserted = tensors_.emplace(name, TensorLocation{shard_index, info});

                if (!inserted.second) {
                    throw std::runtime_error("tensor appears in multiple shards: " + name);
                }
            }
        }

        // 检查 index 中声明的每个 tensor
        // 都确实在物理 shard 中出现
        for (const auto &entry : expected_shard) 
        {
            const std::string &name = entry.first;

            if (tensors_.find(name) == tensors_.end()) {
                throw std::runtime_error("indexed tensor is missing from its shard: " + name);
            }
        }

        if (actual_total_size != expected_total_size) 
        {
            throw std::runtime_error("checkpoint total_size mismatch: expected" + std::to_string(expected_total_size) + ", got" + std::to_string(actual_total_size));
        }

        is_sharded_ = true;
    }

    const std::filesystem::path &SafetensorsCheckpoint::model_directory() const noexcept
    {
        return model_directory_;
    }

    bool SafetensorsCheckpoint::is_sharded() const noexcept
    {
        return is_sharded_;
    }

    std::size_t SafetensorsCheckpoint::shard_count() const noexcept
    {
        return shard_paths_.size();
    }

    std::size_t SafetensorsCheckpoint::tensor_count() const noexcept
    {
        return tensors_.size();
    }

    bool SafetensorsCheckpoint::contains(std::string_view tensor_name) const
    {
        return tensors_.find(std::string{tensor_name}) != tensors_.end();
    }

    const SafetensorsTensorInfo &SafetensorsCheckpoint::tensor_info(std::string_view tensor_name) const
    {
        const auto iterator = tensors_.find(std::string{tensor_name});

        if(iterator == tensors_.end()) {
            throw std::out_of_range("checkpoint tensor not found: " + std::string{tensor_name});
        }

        return iterator->second.info;
    }

    Tensor SafetensorsCheckpoint::load(std::string_view tensor_name, Device target_device) const
    {
        const auto iterator = tensors_.find(std::string{tensor_name});

        if (iterator == tensors_.end()) {
            throw std::out_of_range("checkpoint tensor not found: " + std::string{tensor_name});
        }

        const TensorLocation &location = iterator->second;

        SafetensorsLoader loader{shard_paths_.at(location.shard_index)};

        Tensor result = loader.load(tensor_name, target_device);

        if (result.dtype() != location.info.dtype || result.shape() != location.info.shape) {
            throw std::runtime_error("checkpoint changed while loading tensor: " + std::string(tensor_name));
        }

        return result;
    }

    std::vector<Tensor> SafetensorsCheckpoint::load_many(const std::vector<std::string> &tensor_names, Device target_device) const
    {
        struct Group final
        {
            std::size_t shard_index;
            std::vector<std::string> names;
            // 这个成员是为了记录原本用户输入列表的位置，加载的时候顺序可能改变，但是返回给用户的顺序不能改变
            std::vector<std::size_t> output_indices;
        };

        std::vector<Group> groups;

        // 输入分片编号，找出这个分片在groups 中的位置
        std::unordered_map<std::size_t, std::size_t> group_positions;

        for (std::size_t index = 0; index < tensor_names.size(); ++index)
        {
            const auto location_iterator = tensors_.find(tensor_names[index]);

            if (location_iterator == tensors_.end()) {
                throw std::out_of_range("checkpoint tensor not found: " + tensor_names[index]);
            }

            const std::size_t shard_index = location_iterator->second.shard_index;

            const auto position_result = group_positions.emplace(shard_index, groups.size());

            if (position_result.second) {
                groups.push_back(Group{shard_index, {}, {}});
            }

            Group &group = groups.at(position_result.first->second);

            group.names.push_back(tensor_names[index]);

            group.output_indices.push_back(index);
        }

        // Tensor 没有默认构造函数，所以不能vector<Tensor> results(names.size())
        std::vector<std::optional<Tensor>> output_slots(tensor_names.size());

        for (const Group &group : groups)
        {
            // 每次只让一个 loader 存活
            SafetensorsLoader loader{shard_paths_.at(group.shard_index)};

            std::vector<Tensor> loaded = loader.load_many(group.names, target_device);

            if (loaded.size() != group.names.size()) {
                throw std::logic_error("SafetensorsLoader returned an unexpected tensor count");
            }

            for (std::size_t local_index = 0; local_index < loaded.size(); ++local_index)
            {
                const SafetensorsTensorInfo &expected = tensor_info(group.names[local_index]);

                if (loaded[local_index].dtype() != expected.dtype || loaded[local_index].shape() != expected.shape) {
                    throw std::runtime_error("checkpoint changed while loading tensor: " + group.names[local_index]);
                }

                output_slots[group.output_indices[local_index]].emplace(std::move(loaded[local_index]));
            }
        }

        std::vector<Tensor> results;
        results.reserve(output_slots.size());

        for (std::optional<Tensor> &slot : output_slots)
        {
            if (!slot.has_value()) {
                throw std::logic_error("checkpoint did not fill every output slot");
            } 

            results.push_back(std::move(*slot));
        }

        return results;
    }
}   // namespace qwen3

