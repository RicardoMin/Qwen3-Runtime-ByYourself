#include "qwen3/io/safetensors_metadata.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen3
{
    namespace
    {
        using Json = nlohmann::json;

        [[noreturn]]
        void throw_tensor_error(const std::string &tensor_name, const std::string &message)
        {
            throw SafetensorsFormatError("tensor '" + tensor_name + "': " + message);
        }

        DType parse_dtype(const Json &value, const std::string &tensor_name)
        {
            if (!value.is_string()) 
            {
                throw_tensor_error(tensor_name, "'dtype' must be a string");
            }

            const std::string dtype_name = value.get<std::string>();

            if (dtype_name == "BF16") {
                return DType::BFloat16;
            }
            else if(dtype_name == "F32") {
                return DType::Float32;
            }
            else if(dtype_name == "I32") {
                return DType::Int32;
            }

            throw_tensor_error(tensor_name, "unsupported dtype '" + dtype_name + "'");
        }

        std::size_t parse_size_value(const Json &value, const std::string &tensor_name, const std::string &field_name)
        {
            // 拒绝负数，小数和字符串
            if (!value.is_number_unsigned()) {
                throw_tensor_error(tensor_name, "'" + field_name + "' must contain unsigned integers");
            }

            const std::uint64_t number = value.get<std::uint64_t>();

            // 因为 64 位平台和 32 位平台 size_t 的字节数不一样，所以这里判断依旧有效
            if (number > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                throw_tensor_error(tensor_name, "'" + field_name + "' contains a value too large for size_t");
            }

            return static_cast<std::size_t>(number);
        }
    }   // namespace

    SafetensorsMetadata::SafetensorsMetadata(const SafetensorsFile &file)
    {
        const ConstByteView header = file.header_bytes();
        const std::size_t data_size = file.data_bytes().size_bytes();

        // reinterpret 是不改变底层bit，而是将底层bit换一种解释形式，而static_cast 主要是将数值转变
        const std::string header_json(reinterpret_cast<const char *>(header.data()), header.size_bytes());

        Json root;

        try {
            root = Json::parse(header_json);
        }
        catch (const Json::parse_error &error)
        {
            throw SafetensorsFormatError(std::string("invalid Safetensor JSON header: ") + error.what());
        }

        if (!root.is_object()) {
            throw SafetensorsFormatError("Safetensors header must be a JSON object");
        }

        for (auto iterator = root.begin(); iterator != root.end(); iterator++)
        {
            const std::string tensor_name = iterator.key();

            // __metadata__ 保存创建工具等附加信息
            // 它不是模型权重
            if (tensor_name == "__metadata__") {
                continue;
            }

            const Json &record = iterator.value();

            if (!record.is_object()) {
                throw_tensor_error(tensor_name, "metadata must be a JSON object");
            }

            if (!record.contains("dtype") ||
                !record.contains("shape") ||
                !record.contains("data_offsets"))
            {
                throw_tensor_error(tensor_name, "missing dtype, shape or data_offsets");
            }

            const DType dtype = parse_dtype(record.at("dtype"), tensor_name);

            const Json &shape_json = record.at("shape");

            if (!shape_json.is_array()) {
                throw_tensor_error(tensor_name, "'shape' must be an array");
            }

            std::vector<std::int64_t> dimensions;
            dimensions.reserve(shape_json.size());

            for (const Json &dimension : shape_json) {
                dimensions.push_back(parse_size_value(dimension, tensor_name, "shape"));
            }

            Shape shape{std::move(dimensions)};

            const Json &offsets = record.at("data_offsets");

            if (!offsets.is_array() || offsets.size() != 2) {
                throw_tensor_error(tensor_name, "'data_offsets' must contain exactly two integers");
            }

            const std::size_t begin = parse_size_value(offsets.at(0), tensor_name, "data_offsets");
            const std::size_t end = parse_size_value(offsets.at(1), tensor_name, "data_offsets");

            if (end < begin) {
                throw_tensor_error(tensor_name, "end offset is smaller than begin offset");
            }

            if (end > data_size) {
                throw_tensor_error(tensor_name, "data offset exceeds the weight data buffer");
            }

            std::size_t numel = 0;

            try {
                numel = shape.numel();
            }
            catch (const std::overflow_error &) {
                throw_tensor_error(tensor_name, "shape element count overflow");
            }

            const std::size_t element_size = dtype_size(dtype);

            if (numel != 0 && element_size > std::numeric_limits<std::size_t>::max() / numel) {
                throw_tensor_error(tensor_name, "tensor byte size overflow");
            }

            const std::size_t expected_bytes = numel * element_size;

            const std::size_t actual_bytes = end - begin;

            if (expected_bytes != actual_bytes) {
                throw_tensor_error(tensor_name, "shape and data_offsets describe different byte sizes");
            }

            tensors_.emplace(tensor_name, SafetensorsTensorInfo{dtype, std::move(shape), begin, end});
        }
    }

    std::size_t SafetensorsMetadata::tensor_count() const noexcept
    {
        return tensors_.size();
    }

    bool SafetensorsMetadata::contains(std::string_view name) const
    {
        return tensors_.find(std::string(name)) != tensors_.end();
    }

    const SafetensorsTensorInfo &SafetensorsMetadata::tensor(std::string_view name) const
    {
        const auto iterator = tensors_.find(std::string(name));

        if (iterator == tensors_.end()) {
            throw std::out_of_range("Safetensors tensor not found: " + std::string(name));
        }

        return iterator->second;
    }

    const std::unordered_map<std::string, SafetensorsTensorInfo> &SafetensorsMetadata::tensors() const noexcept
    {
        return tensors_;
    }
}   // namespace qwen3