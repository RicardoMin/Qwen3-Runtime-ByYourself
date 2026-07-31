#pragma once

#include <cstddef>
#include <filesystem>

namespace qwen3
{
    class MappedFile;

    // 类似于 C++20 的 span<const std::byte>
    // 它不拥有内存，不能比对应的 MappedFile 活得更久
    class ConstByteView final
    {
    public:
        ConstByteView() noexcept = default;

        [[nodiscard]]
        const std::byte *data() const noexcept;

        [[nodiscard]]
        std::size_t size_bytes() const noexcept;

        [[nodiscard]]
        bool empty() const noexcept;

        [[nodiscard]]
        ConstByteView subview(std::size_t offset, std::size_t count) const;

    private:
        friend class MappedFile;

        ConstByteView(const std::byte *data, std::size_t size_bytes) noexcept;

    private:
        const std::byte *data_ = nullptr;
        std::size_t size_bytes_ = 0;
    };

    class MappedFile final
    {
    public:
        explicit MappedFile(const std::filesystem::path &path);

        ~MappedFile() noexcept;

        MappedFile(const MappedFile &) = delete;
        MappedFile &operator=(const MappedFile &) = delete;

        MappedFile(MappedFile &&other) noexcept;
        MappedFile &operator=(MappedFile &&other) noexcept;

        [[nodiscard]]
        const std::byte *data() const noexcept;

        [[nodiscard]]
        std::size_t size_bytes() const noexcept;

        [[nodiscard]]
        bool empty() const noexcept;

        [[nodiscard]]
        ConstByteView bytes() const noexcept;

    private:
        void unmap_noexcept() noexcept;

    private:
        const std::byte *data_ = nullptr;
        std::size_t size_bytes_ = 0;
    };
}   // namespace qwen3