#include "qwen3/io/mapped_file.h"

#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
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

    [[noreturn]]
    void throw_errno(
        const std::string &operation)
    {
        throw std::system_error(
            errno,
            std::generic_category(),
            operation);
    }

    class TempFile final
    {
    public:
        explicit TempFile(
            const std::string &contents)
        {
            char pattern[] =
                "/tmp/qwen3-mapped-file-XXXXXX";

            int descriptor =
                ::mkstemp(pattern);

            if (descriptor < 0)
            {
                throw_errno("mkstemp");
            }

            path_ = pattern;

            try
            {
                std::size_t written = 0;

                while (written < contents.size())
                {
                    const ssize_t result = ::write(
                        descriptor,
                        contents.data() + written,
                        contents.size() - written);

                    if (result < 0)
                    {
                        if (errno == EINTR)
                        {
                            continue;
                        }

                        throw_errno(
                            "write temporary file");
                    }

                    if (result == 0)
                    {
                        throw std::runtime_error(
                            "write returned zero");
                    }

                    written +=
                        static_cast<std::size_t>(
                            result);
                }

                const int descriptor_to_close =
                    std::exchange(
                        descriptor,
                        -1);

                if (::close(descriptor_to_close) != 0)
                {
                    throw_errno(
                        "close temporary file");
                }
            }
            catch (...)
            {
                if (descriptor >= 0)
                {
                    static_cast<void>(
                        ::close(descriptor));
                }

                static_cast<void>(
                    ::unlink(path_.c_str()));

                throw;
            }
        }

        ~TempFile() noexcept
        {
            if (!path_.empty())
            {
                static_cast<void>(
                    ::unlink(path_.c_str()));
            }
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

} // namespace

static_assert(
    !std::is_copy_constructible_v<
        qwen3::MappedFile>);

static_assert(
    !std::is_copy_assignable_v<
        qwen3::MappedFile>);

static_assert(
    std::is_nothrow_move_constructible_v<
        qwen3::MappedFile>);

static_assert(
    std::is_nothrow_move_assignable_v<
        qwen3::MappedFile>);

int main()
{
    try
    {
        // 中间包含 \0，验证它是真正的二进制读取。
        const std::string payload{
            "QW\0EN3",
            6};

        TempFile file{payload};

        qwen3::MappedFile mapped{
            file.path()};

        require(
            !mapped.empty(),
            "mapped file should not be empty");

        require(
            mapped.size_bytes() ==
                payload.size(),
            "mapped file size is incorrect");

        require(
            std::memcmp(
                mapped.data(),
                payload.data(),
                payload.size()) == 0,
            "mapped file content is incorrect");

        const qwen3::ConstByteView bytes =
            mapped.bytes();

        require(
            bytes.data() == mapped.data(),
            "byte view points to wrong memory");

        require(
            bytes.size_bytes() ==
                mapped.size_bytes(),
            "byte view size is incorrect");

        const qwen3::ConstByteView middle =
            bytes.subview(2, 3);

        require(
            middle.size_bytes() == 3,
            "subview size is incorrect");

        require(
            std::memcmp(
                middle.data(),
                payload.data() + 2,
                3) == 0,
            "subview content is incorrect");

        require_throws<std::out_of_range>(
            [&]
            {
                static_cast<void>(
                    bytes.subview(
                        bytes.size_bytes() + 1,
                        0));
            },
            "subview accepted invalid offset");

        require_throws<std::out_of_range>(
            [&]
            {
                static_cast<void>(
                    bytes.subview(
                        1,
                        bytes.size_bytes()));
            },
            "subview accepted invalid byte count");

        const std::byte *original_address =
            mapped.data();

        qwen3::MappedFile moved{
            std::move(mapped)};

        require(
            moved.data() == original_address,
            "move changed mapped address");

        require(
            mapped.empty(),
            "moved-from mapping is not empty");

        TempFile old_file{"old"};

        qwen3::MappedFile assigned{
            old_file.path()};

        assigned = std::move(moved);

        require(
            assigned.data() == original_address,
            "move assignment changed mapped address");

        require(
            moved.empty(),
            "move-assigned source is not empty");

        assigned = std::move(assigned);

        require(
            assigned.data() == original_address,
            "self move corrupted mapping");

        TempFile empty_file{""};

        qwen3::MappedFile empty{
            empty_file.path()};

        require(
            empty.empty(),
            "empty file was not detected");

        require(
            empty.size_bytes() == 0,
            "empty file size must be zero");

        require(
            empty.data() == nullptr,
            "empty file data should be null");

        const auto empty_view =
            empty.bytes().subview(0, 0);

        require(
            empty_view.empty(),
            "empty subview should be empty");

        std::filesystem::path missing_path;

        {
            TempFile disappearing{""};
            missing_path = disappearing.path();
        }

        require_throws<std::system_error>(
            [&]
            {
                qwen3::MappedFile missing{
                    missing_path};
            },
            "missing file was accepted");

        require_throws<std::invalid_argument>(
            []
            {
                qwen3::MappedFile directory{
                    std::filesystem::
                        temp_directory_path()};
            },
            "directory was accepted as a file");

        std::cout
            << "MappedFile test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "MappedFile test failed: "
            << error.what()
            << '\n';

        return EXIT_FAILURE;
    }
}