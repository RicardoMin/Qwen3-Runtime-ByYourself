#include "qwen3/io/mapped_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace qwen3
{
    namespace 
    {
        [[noreturn]]
        void throw_errno(int error, const std::string &operation)
        {
            throw std::system_error(error, std::generic_category(), operation);
        }

        void report_cleanup_error(int error, const char *operation) noexcept
        {
            std::fprintf(stderr, "%s failed during cleanup: %s\n", operation, std::strerror(error));
        }

        // 局部使用文件描述符 RAII 对象
        class UniqueFileDescriptor final
        {
        public:
            explicit UniqueFileDescriptor(int descriptor) noexcept
                : descriptor_(descriptor)
            {
            }

            ~UniqueFileDescriptor() noexcept
            {
                if (descriptor_ >= 0 && ::close(descriptor_) != 0)
                {
                    // close 失败后不要重试，fd 可能已经被系统释放并复用
                    report_cleanup_error(errno, "close");
                }
            }

            UniqueFileDescriptor(const UniqueFileDescriptor &) = delete;

            UniqueFileDescriptor &operator=(const UniqueFileDescriptor &) = delete;

            [[nodiscard]]
            int get() const noexcept
            {
                return descriptor_;
            }

        private:
            int descriptor_ = -1;
        };
    }   // namespace

    ConstByteView::ConstByteView(const std::byte *data, std::size_t size_bytes) noexcept
        : data_(data)
        , size_bytes_(size_bytes)
    {
    }

    const std::byte *ConstByteView::data() const noexcept
    {
        return data_;
    }

    std::size_t ConstByteView::size_bytes() const noexcept
    {
        return size_bytes_;
    }

    bool ConstByteView::empty() const noexcept
    {
        return size_bytes_ == 0;
    }

    ConstByteView ConstByteView::subview(std::size_t offset, std::size_t count) const
    {
        if (!data_) {
            return ConstByteView{data_, size_bytes_};
        }
        // 不写 offset + count > size_bytes_
        // 因为可能溢出
        if (offset > size_bytes_ || count > size_bytes_ - offset)
        {
            throw std::out_of_range("byte subview is out of range");
        }

        const std::byte *subview_data = data_;

        subview_data += offset;

        return ConstByteView{subview_data, count};
    }

    MappedFile::MappedFile(const std::filesystem::path &path)
    {
        const std::string path_text = path.string();

        const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);

        if (descriptor < 0) {
            throw_errno(errno, "open " + path_text);
        }

        UniqueFileDescriptor file{descriptor};

        struct stat status{};

        if(::fstat(file.get(), &status) != 0)
        {
            throw_errno(errno, "fstat " + path_text);
        }

        if (!S_ISREG(status.st_mode)) {
            throw std::invalid_argument("mapped path is not regular file: " + path_text);
        }

        if (status.st_size < 0) {
            throw std::runtime_error("file has a negative size: " + path_text);
        }

        const auto file_size = static_cast<std::uintmax_t>(status.st_size);
        const auto maximum_size = static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max());

        if(file_size > maximum_size) {
            throw std::overflow_error("file is too large for this provess: " + path_text);
        }

        const std::size_t mapping_size = static_cast<std::size_t>(file_size);

        // mmap 的长度不能为0
        if (mapping_size == 0) {
            return;
        }

        void *mapped_address = ::mmap(nullptr, mapping_size, PROT_READ, MAP_PRIVATE, file.get(), 0);

        if (mapped_address == MAP_FAILED) {
            throw_errno(errno, "mmap " + path_text);
        }

        // mmap 成功后，下面只有不抛异常的赋值操作
        data_ = static_cast<const std::byte *>(mapped_address);

        size_bytes_ = mapping_size;

        // 构造函数结束时 file 析构并关闭 fd
        // 已建立的 mmap 仍然有效
    }

    MappedFile::~MappedFile() noexcept
    {
        unmap_noexcept();
    }

    MappedFile::MappedFile(MappedFile &&other) noexcept
        : data_(std::exchange(other.data_, nullptr))
        , size_bytes_(std::exchange(other.size_bytes_, 0))
    {
    }

    MappedFile &MappedFile::operator=(MappedFile &&other) noexcept
    {
        if (this == &other)
            return *this;
        
        unmap_noexcept();

        data_ = std::exchange(other.data_, nullptr);
        size_bytes_ = std::exchange(other.size_bytes_, 0);

        return *this;
    }

    const std::byte *MappedFile::data() const noexcept
    {
        return data_;
    }

    std::size_t MappedFile::size_bytes() const noexcept
    {
        return size_bytes_;
    }

    bool MappedFile::empty() const noexcept
    {
        return size_bytes_ == 0;
    }

    ConstByteView MappedFile::bytes() const noexcept
    {
        return ConstByteView{
            data_,
            size_bytes_};
    }

    void MappedFile::unmap_noexcept() noexcept
    {
        if (size_bytes_ == 0) {
            data_ = nullptr;
            return;
        }

        if (::munmap(const_cast<std::byte *>(data_), size_bytes_) != 0)
        {
            report_cleanup_error(errno, "munmap");
        }

        data_ = nullptr;
        size_bytes_ = 0;
    }
}   // namespace qwen3