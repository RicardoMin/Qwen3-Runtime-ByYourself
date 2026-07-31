#include "qwen3/core/tensor_view.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
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
        float storage[24]{};

        qwen3::TensorView tensor{
            storage,
            qwen3::DType::Float32,
            qwen3::Device::cpu(),
            qwen3::Shape{2, 3, 4}};

        require(tensor.data() == storage, "data pointer is incorrect");
        require(tensor.rank() == 3, "rank is incorrect");
        require(tensor.numel() == 24, "numel is incorrect");
        require(tensor.nbytes() == 24 * sizeof(float),
                "byte size is incorrect");

        require(tensor.dim(0) == 2, "dimension 0 is incorrect");
        require(tensor.dim(1) == 3, "dimension 1 is incorrect");
        require(tensor.dim(2) == 4, "dimension 2 is incorrect");

        require(tensor.stride(0) == 12, "stride 0 is incorrect");
        require(tensor.stride(1) == 4, "stride 1 is incorrect");
        require(tensor.stride(2) == 1, "stride 2 is incorrect");

        require(tensor.is_contiguous(),
                "new tensor should be contiguous");

        const auto transposed = tensor.transpose(0, 2);

        require(
            transposed.shape() == qwen3::Shape{4, 3, 2},
            "transposed shape is incorrect");

        require(transposed.stride(0) == 1,
                "transposed stride 0 is incorrect");
        require(transposed.stride(1) == 4,
                "transposed stride 1 is incorrect");
        require(transposed.stride(2) == 12,
                "transposed stride 2 is incorrect");

        require(transposed.data() == storage,
                "transpose must not copy data");

        require(!transposed.is_contiguous(),
                "transposed tensor should not be contiguous");

        std::uint16_t bf16_storage[8]{};

        qwen3::TensorView bf16_tensor{
            bf16_storage,
            qwen3::DType::BFloat16,
            qwen3::Device::cpu(),
            qwen3::Shape{8}};

        require(bf16_tensor.nbytes() == 16,
                "BF16 byte size is incorrect");

        qwen3::TensorView empty_tensor{
            nullptr,
            qwen3::DType::Float32,
            qwen3::Device::cpu(),
            qwen3::Shape{2, 0, 4}};

        require(empty_tensor.is_empty(),
                "empty tensor was not detected");
        require(empty_tensor.numel() == 0,
                "empty tensor numel must be zero");
        require(empty_tensor.nbytes() == 0,
                "empty tensor nbytes must be zero");

        require_throws<std::invalid_argument>(
            []
            {
                qwen3::TensorView invalid{
                    nullptr,
                    qwen3::DType::Float32,
                    qwen3::Device::cpu(),
                    qwen3::Shape{1}};
            },
            "non-empty tensor accepted null data");

        require_throws<std::invalid_argument>(
            [&storage]
            {
                qwen3::TensorView invalid{
                    storage,
                    qwen3::DType::Float32,
                    qwen3::Device::cpu(),
                    qwen3::Shape{2, 3},
                    std::vector<std::int64_t>{3}};
            },
            "tensor accepted incorrect stride rank");

        require_throws<std::invalid_argument>(
            [&storage]
            {
                qwen3::TensorView invalid{
                    storage,
                    qwen3::DType::Float32,
                    qwen3::Device::cpu(),
                    qwen3::Shape{2, 3},
                    std::vector<std::int64_t>{3, -1}};
            },
            "tensor accepted negative stride");

        require_throws<std::out_of_range>(
            [&tensor]
            {
                static_cast<void>(tensor.stride(3));
            },
            "invalid stride axis was accepted");

        std::byte dummy{};

        qwen3::TensorView huge_tensor{
            &dummy,
            qwen3::DType::Float32,
            qwen3::Device::cpu(),
            qwen3::Shape{
                std::numeric_limits<std::int64_t>::max()}};

        require_throws<std::overflow_error>(
            [&huge_tensor]
            {
                static_cast<void>(huge_tensor.nbytes());
            },
            "tensor byte-size overflow was not detected");

        std::cout << "TensorView test passed.\n";
        std::cout << "shape: "
                  << tensor.shape().to_string() << '\n';
        std::cout << "numel: " << tensor.numel() << '\n';
        std::cout << "nbytes: " << tensor.nbytes() << '\n';

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr << "TensorView test failed: "
                  << error.what() << '\n';

        return EXIT_FAILURE;
    }
}