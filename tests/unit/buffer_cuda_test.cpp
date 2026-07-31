#include "qwen3/core/buffer.h"
#include "qwen3/cuda/cuda_check.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
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

int main()
{
    try
    {
        int device_count = 0;
        QWEN3_CUDA_CHECK(
            cudaGetDeviceCount(&device_count));

        require(
            device_count > 0,
            "no visible CUDA device");

        int device_before = 0;
        QWEN3_CUDA_CHECK(
            cudaGetDevice(&device_before));

        qwen3::Buffer zero_cuda{
            0,
            qwen3::Device::cuda(0)};

        require(zero_cuda.empty(),
                "zero-byte CUDA Buffer should be empty");
        require(zero_cuda.data() == nullptr,
                "zero-byte CUDA Buffer should be null");

        constexpr std::size_t element_count = 8;
        constexpr std::size_t byte_count =
            element_count * sizeof(std::int32_t);

        qwen3::Buffer host_source{
            byte_count,
            qwen3::Device::cpu()};

        qwen3::Buffer host_destination{
            byte_count,
            qwen3::Device::cpu()};

        qwen3::Buffer cuda_buffer{
            byte_count,
            qwen3::Device::cuda(0)};

        auto *source_values =
            static_cast<std::int32_t *>(
                host_source.data());

        for (std::size_t index = 0;
             index < element_count;
             ++index)
        {
            source_values[index] =
                static_cast<std::int32_t>(
                    index * 10 + 7);
        }

        cudaPointerAttributes attributes{};

        QWEN3_CUDA_CHECK(
            cudaPointerGetAttributes(
                &attributes,
                cuda_buffer.data()));

        require(
            attributes.type == cudaMemoryTypeDevice,
            "CUDA Buffer is not device memory");

        require(
            attributes.device == 0,
            "CUDA Buffer allocated on wrong device");

        qwen3::copy_buffer(
            cuda_buffer,
            host_source);

        qwen3::copy_buffer(
            host_destination,
            cuda_buffer);

        const auto *destination_values =
            static_cast<const std::int32_t *>(
                host_destination.data());

        for (std::size_t index = 0;
             index < element_count;
             ++index)
        {
            require(
                destination_values[index] ==
                    source_values[index],
                "CPU-CUDA-CPU round trip failed");
        }

        qwen3::Buffer second_cuda{
            byte_count,
            qwen3::Device::cuda(0)};

        qwen3::copy_buffer(
            second_cuda,
            cuda_buffer);

        qwen3::Buffer second_host{
            byte_count,
            qwen3::Device::cpu()};

        qwen3::copy_buffer(
            second_host,
            second_cuda);

        const auto *second_values =
            static_cast<const std::int32_t *>(
                second_host.data());

        for (std::size_t index = 0;
             index < element_count;
             ++index)
        {
            require(
                second_values[index] ==
                    source_values[index],
                "CUDA-CUDA copy failed");
        }

        void *cuda_pointer = second_cuda.data();

        qwen3::Buffer moved_cuda{
            std::move(second_cuda)};

        require(
            moved_cuda.data() == cuda_pointer,
            "CUDA Buffer move changed pointer");

        require(
            second_cuda.empty(),
            "moved-from CUDA Buffer is not empty");

        int device_after = 0;
        QWEN3_CUDA_CHECK(
            cudaGetDevice(&device_after));

        require(
            device_after == device_before,
            "Buffer changed the current CUDA device");

        require_throws<std::runtime_error>(
            [device_count]
            {
                qwen3::Buffer invalid{
                    16,
                    qwen3::Device::cuda(device_count)};
            },
            "invalid CUDA device was accepted");

        // 清理可能保存在 CUDA last-error slot 中的错误。
        static_cast<void>(cudaGetLastError());

        QWEN3_CUDA_CHECK(
            cudaGetDevice(&device_after));

        require(
            device_after == device_before,
            "failed allocation changed current device");

        std::cout << "CUDA Buffer test passed.\n";
        std::cout << "Visible CUDA devices: "
                  << device_count << '\n';

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "CUDA Buffer test failed: "
            << error.what() << '\n';

        return EXIT_FAILURE;
    }
}