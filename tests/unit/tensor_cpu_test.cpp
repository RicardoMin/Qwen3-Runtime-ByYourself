#include "qwen3/core/tensor.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
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
    !std::is_default_constructible_v<qwen3::Tensor>);

static_assert(
    !std::is_copy_constructible_v<qwen3::Tensor>);

static_assert(
    !std::is_copy_assignable_v<qwen3::Tensor>);

static_assert(
    std::is_nothrow_move_constructible_v<qwen3::Tensor>);

static_assert(
    std::is_nothrow_move_assignable_v<qwen3::Tensor>);

int main()
{
    try
    {
        qwen3::Tensor tensor =
            qwen3::Tensor::empty(
                qwen3::Shape{2, 3, 4},
                qwen3::DType::Float32,
                qwen3::Device::cpu());

        require(tensor.data() != nullptr,
                "Tensor allocation returned null");
        require(tensor.device().is_cpu(),
                "Tensor device is incorrect");
        require(tensor.dtype() == qwen3::DType::Float32,
                "Tensor dtype is incorrect");
        require(tensor.shape() == qwen3::Shape{2, 3, 4},
                "Tensor shape is incorrect");
        require(tensor.numel() == 24,
                "Tensor numel is incorrect");
        require(tensor.nbytes() == 24 * sizeof(float),
                "Tensor byte size is incorrect");

        auto view = tensor.view();

        require(view.data() == tensor.data(),
                "TensorView points to wrong memory");
        require(view.shape() == tensor.shape(),
                "TensorView shape is incorrect");
        require(view.dtype() == tensor.dtype(),
                "TensorView dtype is incorrect");
        require(view.device() == tensor.device(),
                "TensorView device is incorrect");
        require(view.is_contiguous(),
                "owning Tensor view should be contiguous");

        auto transposed = view.transpose(0, 2);

        require(transposed.data() == tensor.data(),
                "transpose should not copy memory");
        require(
            transposed.shape() == qwen3::Shape{4, 3, 2},
            "transposed shape is incorrect");
        require(!transposed.is_contiguous(),
                "transposed view should not be contiguous");

        // owning Tensor 自己没有被转置。
        require(
            tensor.shape() == qwen3::Shape{2, 3, 4},
            "transpose changed owning Tensor");

        qwen3::Tensor bf16 =
            qwen3::Tensor::empty(
                qwen3::Shape{8},
                qwen3::DType::BFloat16,
                qwen3::Device::cpu());

        require(bf16.nbytes() == 16,
                "BF16 Tensor byte size is incorrect");

        // Shape{} 是 rank-0 标量，不是空张量。
        qwen3::Tensor scalar =
            qwen3::Tensor::empty(
                qwen3::Shape{},
                qwen3::DType::Float32,
                qwen3::Device::cpu());

        require(scalar.numel() == 1,
                "scalar Tensor must have one element");
        require(scalar.nbytes() == sizeof(float),
                "scalar Tensor byte size is incorrect");
        require(scalar.data() != nullptr,
                "scalar Tensor should own memory");

        qwen3::Tensor empty =
            qwen3::Tensor::empty(
                qwen3::Shape{2, 0, 4},
                qwen3::DType::Float32,
                qwen3::Device::cpu());

        require(empty.is_empty(),
                "zero-dimension Tensor should be empty");
        require(empty.numel() == 0,
                "empty Tensor numel must be zero");
        require(empty.nbytes() == 0,
                "empty Tensor nbytes must be zero");
        require(empty.data() == nullptr,
                "empty Tensor data should be null");

        qwen3::Tensor source =
            qwen3::Tensor::empty(
                qwen3::Shape{8},
                qwen3::DType::Float32,
                qwen3::Device::cpu());

        auto *source_values =
            static_cast<float *>(source.data());

        for (std::size_t index = 0; index < 8; ++index)
        {
            source_values[index] =
                static_cast<float>(index) + 0.5F;
        }

        qwen3::Tensor cloned = source.clone();

        require(cloned.data() != source.data(),
                "clone must allocate independent memory");

        require(
            std::memcmp(
                cloned.data(),
                source.data(),
                source.nbytes()) == 0,
            "clone content is incorrect");

        source_values[0] = 999.0F;

        const auto *cloned_values =
            static_cast<const float *>(cloned.data());

        require(cloned_values[0] == 0.5F,
                "clone shares storage with source");

        qwen3::Tensor wrong_shape =
            qwen3::Tensor::empty(
                qwen3::Shape{4},
                qwen3::DType::Float32,
                qwen3::Device::cpu());

        require_throws<std::invalid_argument>(
            [&]
            {
                wrong_shape.copy_from(source);
            },
            "copy_from accepted mismatching shapes");

        qwen3::Tensor wrong_dtype =
            qwen3::Tensor::empty(
                qwen3::Shape{8},
                qwen3::DType::Int32,
                qwen3::Device::cpu());

        require_throws<std::invalid_argument>(
            [&]
            {
                wrong_dtype.copy_from(source);
            },
            "copy_from accepted mismatching dtypes");

        require_throws<std::overflow_error>(
            []
            {
                static_cast<void>(
                    qwen3::Tensor::empty(
                        qwen3::Shape{
                            std::numeric_limits<
                                std::int64_t>::max()},
                        qwen3::DType::Float32,
                        qwen3::Device::cpu()));
            },
            "Tensor byte-size overflow was not detected");

        void *original_pointer = tensor.data();

        qwen3::Tensor moved{
            std::move(tensor)};

        require(moved.data() == original_pointer,
                "Tensor move changed storage pointer");

        moved = std::move(moved);

        require(moved.data() == original_pointer,
                "Tensor self-move corrupted storage");

        // moved-from tensor 只保证可以析构或重新赋值，
        // 不应继续访问它的 data/shape/view。

        std::cout << "CPU Tensor test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "CPU Tensor test failed: "
            << error.what() << '\n';

        return EXIT_FAILURE;
    }
}