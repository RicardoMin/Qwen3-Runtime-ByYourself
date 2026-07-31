#include "qwen3/io/safetensors_file.h"

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace
{

    void require(
        bool condition,
        const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    template <typename Exception, typename Function>
    void require_throws(
        Function &&function,
        const std::string &message)
    {
        bool caught = false;

        try
        {
            function();
        }
        catch (const Exception &)
        {
            caught = true;
        }

        require(caught, message);
    }

    std::uint64_t next_temp_id() noexcept
    {
        static std::uint64_t value = 0;
        return value++;
    }

    class TempFile final
    {
    public:
        explicit TempFile(
            const std::string &contents)
            : path_(
                  std::filesystem::temp_directory_path() /
                  ("qwen3-safetensors-" +
                   std::to_string(::getpid()) +
                   "-" +
                   std::to_string(next_temp_id()) +
                   ".bin"))
        {
            try
            {
                std::ofstream output(
                    path_,
                    std::ios::binary |
                        std::ios::trunc);

                if (!output)
                {
                    throw std::runtime_error(
                        "failed to create temporary file");
                }

                output.write(
                    contents.data(),
                    static_cast<std::streamsize>(
                        contents.size()));

                output.close();

                if (!output)
                {
                    throw std::runtime_error(
                        "failed to write temporary file");
                }
            }
            catch (...)
            {
                std::error_code ignored;
                std::filesystem::remove(
                    path_,
                    ignored);

                throw;
            }
        }

        ~TempFile() noexcept
        {
            std::error_code ignored;

            std::filesystem::remove(
                path_,
                ignored);
        }

        TempFile(const TempFile &) = delete;
        TempFile &operator=(const TempFile &) = delete;

        [[nodiscard]]
        const std::filesystem::path &
        path() const noexcept
        {
            return path_;
        }

    private:
        std::filesystem::path path_;
    };

    std::string encode_u64_le(
        std::uint64_t value)
    {
        std::string bytes(8, '\0');

        for (std::size_t index = 0;
             index < 8;
             ++index)
        {
            bytes[index] =
                static_cast<char>(
                    (value >> (index * 8)) &
                    0xFFULL);
        }

        return bytes;
    }

    std::string make_safetensors_file(
        const std::string &header,
        const std::string &data)
    {
        std::string contents =
            encode_u64_le(
                static_cast<std::uint64_t>(
                    header.size()));

        contents.append(header);
        contents.append(data);

        return contents;
    }

} // namespace

static_assert(
    !std::is_copy_constructible_v<
        qwen3::SafetensorsFile>);

static_assert(
    !std::is_copy_assignable_v<
        qwen3::SafetensorsFile>);

static_assert(
    std::is_nothrow_move_constructible_v<
        qwen3::SafetensorsFile>);

static_assert(
    std::is_nothrow_move_assignable_v<
        qwen3::SafetensorsFile>);

int main()
{
    try
    {
        std::string header =
            R"({"tensor":{"dtype":"F32"}})";

        // Safetensors Header 允许在结尾使用空格填充。
        header.append(5, ' ');

        const std::string binary_data{
            "\x00\x80\xFF",
            3};

        TempFile valid_file{
            make_safetensors_file(
                header,
                binary_data)};

        qwen3::SafetensorsFile file{
            valid_file.path()};

        require(
            file.file_size_bytes() ==
                8 +
                    header.size() +
                    binary_data.size(),
            "file size is incorrect");

        require(
            file.header_size_bytes() ==
                header.size(),
            "header size is incorrect");

        require(
            file.data_offset_bytes() ==
                8 + header.size(),
            "data offset is incorrect");

        require(
            file.data_size_bytes() ==
                binary_data.size(),
            "data size is incorrect");

        const qwen3::ConstByteView header_view =
            file.header_bytes();

        require(
            std::memcmp(
                header_view.data(),
                header.data(),
                header.size()) == 0,
            "header content is incorrect");

        const qwen3::ConstByteView data_view =
            file.data_bytes();

        require(
            std::memcmp(
                data_view.data(),
                binary_data.data(),
                binary_data.size()) == 0,
            "binary data content is incorrect");

        require(
            header_view.data() +
                    header_view.size_bytes() ==
                data_view.data(),
            "header and data regions are not adjacent");

        // Header 长度使用 258，
        // 前八字节应该为 02 01 00 00 00 00 00 00。
        std::string large_header = "{}";
        large_header.resize(258, ' ');

        TempFile little_endian_file{
            make_safetensors_file(
                large_header,
                "")};

        qwen3::SafetensorsFile little_endian{
            little_endian_file.path()};

        require(
            little_endian.header_size_bytes() ==
                258,
            "little-endian length was decoded incorrectly");

        require(
            little_endian.data_size_bytes() == 0,
            "empty data region is incorrect");

        // 文件长度从 0 到 7，都没有完整长度字段。
        for (std::size_t length = 0;
             length < 8;
             ++length)
        {
            TempFile too_short{
                std::string(length, 'x')};

            require_throws<
                qwen3::SafetensorsFormatError>(
                [&]
                {
                    qwen3::SafetensorsFile invalid{
                        too_short.path()};
                },
                "short safetensors file was accepted");
        }

        TempFile empty_header{
            encode_u64_le(0)};

        require_throws<
            qwen3::SafetensorsFormatError>(
            [&]
            {
                qwen3::SafetensorsFile invalid{
                    empty_header.path()};
            },
            "empty safetensors header was accepted");

        TempFile oversized_header{
            encode_u64_le(
                qwen3::SafetensorsFile::
                    kMaxHeaderSizeBytes +
                1)};

        require_throws<
            qwen3::SafetensorsFormatError>(
            [&]
            {
                qwen3::SafetensorsFile invalid{
                    oversized_header.path()};
            },
            "oversized safetensors header was accepted");

        std::string truncated_contents =
            encode_u64_le(10);

        truncated_contents += "{}";

        TempFile truncated_header{
            truncated_contents};

        require_throws<
            qwen3::SafetensorsFormatError>(
            [&]
            {
                qwen3::SafetensorsFile invalid{
                    truncated_header.path()};
            },
            "truncated safetensors header was accepted");

        TempFile invalid_first_character{
            make_safetensors_file(
                "[]",
                "")};

        require_throws<
            qwen3::SafetensorsFormatError>(
            [&]
            {
                qwen3::SafetensorsFile invalid{
                    invalid_first_character.path()};
            },
            "non-object header was accepted");

        TempFile leading_space{
            make_safetensors_file(
                " {}",
                "")};

        require_throws<
            qwen3::SafetensorsFormatError>(
            [&]
            {
                qwen3::SafetensorsFile invalid{
                    leading_space.path()};
            },
            "header with leading space was accepted");

        // 验证移动只转移映射所有权，不复制文件内容。
        const std::byte *original_header_address =
            file.header_bytes().data();

        qwen3::SafetensorsFile moved{
            std::move(file)};

        require(
            moved.header_bytes().data() ==
                original_header_address,
            "moving changed the mapped address");

        require(
            file.file_size_bytes() == 0,
            "moved-from file still owns the mapping");

        TempFile target_file{
            make_safetensors_file(
                "{}",
                "")};

        qwen3::SafetensorsFile assigned{
            target_file.path()};

        assigned = std::move(moved);

        require(
            assigned.header_bytes().data() ==
                original_header_address,
            "move assignment changed mapped address");

        require(
            moved.file_size_bytes() == 0,
            "move-assigned source still owns mapping");

        std::cout
            << "SafetensorsFile test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "SafetensorsFile test failed: "
            << error.what()
            << '\n';

        return EXIT_FAILURE;
    }
}