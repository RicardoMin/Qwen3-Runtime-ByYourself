#include "qwen3/io/safetensors_file.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace qwen3
{
    namespace
    {
        std::uint64_t decode_little_endian_u64(ConstByteView bytes)
        {
            if (bytes.size_bytes() != 8) {
                throw std::logic_error("u64 decoder requires exactly 8 bytes");
            }

            std::uint64_t value = 0;

            for (std::size_t index = 0; index < 8; ++index)
            {
                const unsigned char byte_value = std::to_integer<unsigned char>(bytes.data()[index]);

                value |= static_cast<std::uint64_t>(byte_value) << (index * 8);
            }

            return value;
        }
    }   // namespace

    SafetensorsFile::SafetensorsFile(const std::filesystem::path &path)
        : mapped_file_(path)
    {
        const std::string path_text = path.string();

        const ConstByteView file_bytes = mapped_file_.bytes();

        // 至少需要前面的 8 字节长度字段
        if (file_bytes.size_bytes() < kLengthPrefixSizeBytes) {
            throw SafetensorsFormatError("safetensors file is shorter than "
                                        "the 8-byte header length prefix: " + path_text);
        }

        const ConstByteView length_prefix = file_bytes.subview(0, kLengthPrefixSizeBytes);

        const std::uint64_t raw_header_size = decode_little_endian_u64(length_prefix);

        if (raw_header_size == 0) {
            throw SafetensorsFormatError("safetensors header is empty: " + path_text);
        }

        if (raw_header_size > kMaxHeaderSizeBytes) {
            throw SafetensorsFormatError("safetensors header exceeds 100,000,000 bytes: " + path_text);
        }

        const std::size_t remaining_bytes = file_bytes.size_bytes() - kLengthPrefixSizeBytes;

        if (raw_header_size > static_cast<std::uint64_t>(remaining_bytes)) {
            throw SafetensorsFormatError("safetensors header extends beyond the end of the file: " + path_text);
        }

        header_size_bytes_ = static_cast<std::size_t>(raw_header_size);
        data_offset_bytes_ = kLengthPrefixSizeBytes + header_size_bytes_;

        const ConstByteView header = header_bytes();

        const unsigned char first_character = std::to_integer<unsigned char>(header.data()[0]);

        // Header 必须直接以 { 开始
        // 前面不能有空格或者UTF-8 BOM
        if (first_character != static_cast<unsigned char>('{')) {
            throw SafetensorsFormatError("safetensors header must begin with '{': " + path_text);
        }
    }

    SafetensorsFile::SafetensorsFile(SafetensorsFile &&other) noexcept
        : mapped_file_(std::move(other.mapped_file_))
        , header_size_bytes_(std::exchange(other.header_size_bytes_, 0))
        , data_offset_bytes_(std::exchange(other.data_offset_bytes_, 0))
    {}

    SafetensorsFile &
    SafetensorsFile::operator=(
        SafetensorsFile &&other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        mapped_file_ =
            std::move(other.mapped_file_);

        header_size_bytes_ =
            std::exchange(
                other.header_size_bytes_,
                0);

        data_offset_bytes_ =
            std::exchange(
                other.data_offset_bytes_,
                0);

        return *this;
    }

    std::size_t SafetensorsFile::file_size_bytes() const noexcept
    {
        return mapped_file_.size_bytes();
    }

    std::size_t SafetensorsFile::header_size_bytes() const noexcept
    {
        return header_size_bytes_;
    }

    std::size_t SafetensorsFile::data_offset_bytes() const noexcept
    {
        return data_offset_bytes_;
    }

    std::size_t SafetensorsFile::data_size_bytes() const noexcept
    {
        if (data_offset_bytes_ == 0)
        {
            return 0;
        }

        return mapped_file_.size_bytes() -
               data_offset_bytes_;
    }

    ConstByteView SafetensorsFile::header_bytes() const
    {
        if (data_offset_bytes_ == 0) {
            return {};
        }

        return mapped_file_.bytes().subview(kLengthPrefixSizeBytes, header_size_bytes_);
    }

    ConstByteView SafetensorsFile::data_bytes() const 
    {
        if (data_offset_bytes_ == 0) {
            return {};
        }

        return mapped_file_.bytes().subview(data_offset_bytes_, data_size_bytes());
    }

}   // namespace qwen3