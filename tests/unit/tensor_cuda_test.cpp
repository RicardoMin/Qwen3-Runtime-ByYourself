#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cuda_check.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

    void require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

} // namespace

int main()
{
    try
    {
        int device_count = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDeviceCount(&device_count));

        require(device_count > 0,
                "no visible CUDA device");

        int device_before = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDevice(&device_before));

        constexpr std::size_t element_count = 8;

        qwen3::Tensor cpu_tensor =
            qwen3::Tensor::empty(
                qwen3::Shape{element_count},
                qwen3::DType::Int32,
                qwen3::Device::cpu());

        auto *input =
            static_cast<std::int32_t *>(
                cpu_tensor.data());

        for (std::size_t index = 0;
             index < element_count;
             ++index)
        {
            input[index] =
                static_cast<std::int32_t>(
                    index * 11 + 3);
        }

        qwen3::Tensor cuda_tensor =
            cpu_tensor.copy_to(
                qwen3::Device::cuda(0));

        require(cuda_tensor.device().is_cuda(),
                "copy_to did not create CUDA Tensor");
        require(cuda_tensor.device().index() == 0,
                "CUDA Tensor is on wrong device");
        require(cuda_tensor.shape() == cpu_tensor.shape(),
                "copy_to changed shape");
        require(cuda_tensor.dtype() == cpu_tensor.dtype(),
                "copy_to changed dtype");

        cudaPointerAttributes attributes{};

        QWEN3_CUDA_CHECK(
            cudaPointerGetAttributes(
                &attributes,
                cuda_tensor.data()));

        require(
            attributes.type == cudaMemoryTypeDevice,
            "CUDA Tensor does not own device memory");

        qwen3::Tensor round_trip =
            cuda_tensor.copy_to(
                qwen3::Device::cpu());

        const auto *output =
            static_cast<const std::int32_t *>(
                round_trip.data());

        for (std::size_t index = 0;
             index < element_count;
             ++index)
        {
            require(
                output[index] == input[index],
                "Tensor CPU-CUDA-CPU round trip failed");
        }

        qwen3::Tensor cuda_clone =
            cuda_tensor.clone();

        require(
            cuda_clone.data() != cuda_tensor.data(),
            "CUDA clone shares storage with source");

        qwen3::Tensor clone_on_cpu =
            cuda_clone.copy_to(
                qwen3::Device::cpu());

        const auto *clone_values =
            static_cast<const std::int32_t *>(
                clone_on_cpu.data());

        for (std::size_t index = 0;
             index < element_count;
             ++index)
        {
            require(
                clone_values[index] == input[index],
                "CUDA clone content is incorrect");
        }

        int device_after = 0;

        QWEN3_CUDA_CHECK(
            cudaGetDevice(&device_after));

        require(
            device_after == device_before,
            "Tensor operations changed current CUDA device");

        std::cout << "CUDA Tensor test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "CUDA Tensor test failed: "
            << error.what() << '\n';

        return EXIT_FAILURE;
    }
}