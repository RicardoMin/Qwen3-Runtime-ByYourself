#pragma once

#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/io/safetensors_file.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

namespace qwen3
{
    struct SafetensorsTensorInfo final
    {
        DType dtype;
        Shape shape;

        // 这里是相对于 Safetensors 的数据区的开头，而不是相对于文件的开头
        std::size_t begin_offset;
        std::size_t end_offset;

        [[nodiscard]]
        std::size_t size_bytes() const noexcept
        {
            return end_offset - begin_offset;
        }
    };

    class SafetensorsMetadata final
    {
    public:
        explicit SafetensorsMetadata(const SafetensorsFile &file);

        [[nodiscard]]
        std::size_t tensor_count() const noexcept;

        [[nodiscard]]
        bool contains(std::string_view name) const;

        [[nodiscard]]
        const SafetensorsTensorInfo &tensor(std::string_view name) const;

        [[nodiscard]]
        const std::unordered_map<std::string, SafetensorsTensorInfo> &tensors() const noexcept;

    private:
        std::unordered_map<std::string, SafetensorsTensorInfo> tensors_;
    };

}   // namespace qwen3