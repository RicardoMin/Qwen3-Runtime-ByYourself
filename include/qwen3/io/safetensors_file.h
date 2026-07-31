#pragma once

#include "qwen3/io/mapped_file.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace qwen3
{
    class SafetensorsFormatError final
        : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    class SafetensorsFile final
    {
    public:
        static constexpr std::size_t kLengthPrefixSizeBytes = 8;

        // 官方限制是十进制 100,000,000 字节
        // 不是 100 * 1024 * 1024
        static constexpr std::uint64_t kMaxHeaderSizeBytes = 100'000'000ULL;

        explicit SafetensorsFile(const std::filesystem::path &path);

        SafetensorsFile(const SafetensorsFile &) = delete;

        SafetensorsFile &operator=(
            const SafetensorsFile &) = delete;

        SafetensorsFile(SafetensorsFile &&other) noexcept;

        SafetensorsFile &operator=(SafetensorsFile &&other) noexcept;

        [[nodiscard]]
        std::size_t file_size_bytes() const noexcept;

        [[nodiscard]]
        std::size_t header_size_bytes() const noexcept;

        [[nodiscard]]
        std::size_t data_offset_bytes() const noexcept;

        [[nodiscard]]
        std::size_t data_size_bytes() const noexcept;

        [[nodiscard]]
        ConstByteView header_bytes() const;

        [[nodiscard]]
        ConstByteView data_bytes() const;

    private:
        MappedFile mapped_file_;

        std::size_t header_size_bytes_ = 0;
        std::size_t data_offset_bytes_ = 0;
    };
}   // namespace qwen3