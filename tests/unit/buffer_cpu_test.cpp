#include "qwen3/core/buffer.h"
#include "qwen3/core/tensor_view.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace
{

    void require(bool condition, const std::string &message)
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

} // namespace

static_assert(
    !std::is_copy_constructible_v<qwen3::Buffer>);

static_assert(
    !std::is_copy_assignable_v<qwen3::Buffer>);

static_assert(
    std::is_nothrow_move_constructible_v<qwen3::Buffer>);

static_assert(
    std::is_nothrow_move_assignable_v<qwen3::Buffer>);

static_assert(
    std::is_nothrow_destructible_v<qwen3::Buffer>);

int main()
{
    try
    {
        qwen3::Buffer default_buffer;

        require(default_buffer.empty(),
                "default Buffer should be empty");
        require(default_buffer.data() == nullptr,
                "default Buffer data should be null");
        require(default_buffer.device().is_cpu(),
                "default Buffer should be CPU");

        qwen3::Buffer zero_buffer{
            0,
            qwen3::Device::cpu()};

        require(zero_buffer.empty(),
                "zero-byte Buffer should be empty");
        require(zero_buffer.data() == nullptr,
                "zero-byte Buffer should have null data");

        for (const std::size_t size : {1, 2, 63, 64, 65})
        {
            qwen3::Buffer buffer{
                size,
                qwen3::Device::cpu()};

            require(buffer.data() != nullptr,
                    "CPU allocation returned null");

            const auto address =
                reinterpret_cast<std::uintptr_t>(
                    buffer.data());

            require(
                address % qwen3::Buffer::kCpuAlignment == 0,
                "CPU Buffer is not 64-byte aligned");

            std::memset(buffer.data(), 0x5A, size);

            const auto *bytes =
                static_cast<const unsigned char *>(
                    buffer.data());

            for (std::size_t index = 0;
                 index < size;
                 ++index)
            {
                require(
                    bytes[index] == 0x5A,
                    "CPU Buffer read/write failed");
            }
        }

        qwen3::Buffer source{
            128,
            qwen3::Device::cpu()};

        void *original_pointer = source.data();

        qwen3::Buffer moved{std::move(source)};

        require(moved.data() == original_pointer,
                "move constructor changed pointer");
        require(moved.size_bytes() == 128,
                "move constructor lost size");
        require(source.data() == nullptr,
                "moved-from Buffer still owns data");
        require(source.size_bytes() == 0,
                "moved-from Buffer still has size");

        qwen3::Buffer assigned{
            32,
            qwen3::Device::cpu()};

        qwen3::Buffer assignment_source{
            64,
            qwen3::Device::cpu()};

        void *assignment_pointer =
            assignment_source.data();

        assigned = std::move(assignment_source);

        require(assigned.data() == assignment_pointer,
                "move assignment changed pointer");
        require(assigned.size_bytes() == 64,
                "move assignment lost size");
        require(assignment_source.empty(),
                "move-assigned source is not empty");

        assigned = std::move(assigned);

        require(assigned.data() == assignment_pointer,
                "self move assignment corrupted Buffer");

        qwen3::Buffer copy_source{
            16,
            qwen3::Device::cpu()};

        qwen3::Buffer copy_destination{
            16,
            qwen3::Device::cpu()};

        auto *source_bytes =
            static_cast<unsigned char *>(
                copy_source.data());

        for (std::size_t index = 0; index < 16; ++index)
        {
            source_bytes[index] =
                static_cast<unsigned char>(index);
        }

        qwen3::copy_buffer(
            copy_destination,
            copy_source);

        require(
            std::memcmp(
                copy_destination.data(),
                copy_source.data(),
                16) == 0,
            "CPU Buffer copy failed");

        require_throws<std::invalid_argument>(
            [&]
            {
                qwen3::Buffer wrong_size{
                    8,
                    qwen3::Device::cpu()};

                qwen3::copy_buffer(
                    wrong_size,
                    copy_source);
            },
            "different-size Buffer copy was accepted");

        require_throws<std::out_of_range>(
            [&]
            {
                qwen3::copy_buffer(
                    copy_destination,
                    12,
                    copy_source,
                    0,
                    8);
            },
            "out-of-range Buffer copy was accepted");

        qwen3::Buffer tensor_storage{
            24 * sizeof(float),
            qwen3::Device::cpu()};

        qwen3::TensorView tensor_view{
            tensor_storage.data(),
            qwen3::DType::Float32,
            tensor_storage.device(),
            qwen3::Shape{2, 3, 4}};

        require(
            tensor_view.nbytes() ==
                tensor_storage.size_bytes(),
            "TensorView and Buffer sizes do not match");

        std::cout << "CPU Buffer test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "CPU Buffer test failed: "
            << error.what() << '\n';

        return EXIT_FAILURE;
    }
}