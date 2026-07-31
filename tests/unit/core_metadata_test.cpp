#include "qwen3/core/device.h"
#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename ExpectedException, typename Function>
void require_throws(
    Function&& function,
    const char* message
) {
    bool expected_exception_thrown = false;

    try {
        std::forward<Function>(function)();
    } catch (const ExpectedException&) {
        expected_exception_thrown = true;
    }

    require(expected_exception_thrown, message);
}

}  // namespace

int main() {
    try {
        // DType
        require(
            qwen3::dtype_size(qwen3::DType::Float32) == 4,
            "float32 should occupy four bytes"
        );

        require(
            qwen3::dtype_size(qwen3::DType::BFloat16) == 2,
            "bfloat16 should occupy two bytes"
        );

        require(
            qwen3::dtype_name(qwen3::DType::Int32) == "int32",
            "int32 name is incorrect"
        );

        // Device
        const qwen3::Device cpu = qwen3::Device::cpu();
        const qwen3::Device cuda = qwen3::Device::cuda(0);

        require(cpu.is_cpu(), "CPU device is incorrect");
        require(cuda.is_cuda(), "CUDA device is incorrect");
        require(cuda.index() == 0, "CUDA index is incorrect");
        require(cuda.to_string() == "cuda: 0", "CUDA name is incorrect");
        require(cpu != cuda, "CPU and CUDA must differ");

        require_throws<std::invalid_argument>(
            [] {
                const auto invalid = qwen3::Device::cuda(-1);
                (void)invalid;
            },
            "negative CUDA index was accepted"
        );

        require_throws<std::invalid_argument>(
            [] {
                const qwen3::Device invalid{
                    qwen3::DeviceType::Cpu,
                    1
                };
                (void)invalid;
            },
            "nonzero CPU index was accepted"
        );

        // Shape
        const qwen3::Shape hidden_states{2, 8, 1024};

        require(
            hidden_states.rank() == 3,
            "shape rank is incorrect"
        );

        require(
            hidden_states[2] == 1024,
            "shape indexing is incorrect"
        );

        require(
            hidden_states.numel() == 2 * 8 * 1024,
            "shape element count is incorrect"
        );

        const qwen3::Shape scalar{};

        require(scalar.is_scalar(), "scalar detection failed");
        require(scalar.numel() == 1, "scalar should have one element");

        const qwen3::Shape empty{2, 0, 4};

        require(
            empty.has_zero_dimension(),
            "zero dimension was not detected"
        );

        require(empty.numel() == 0, "empty shape should have no elements");

        require_throws<std::invalid_argument>(
            [] {
                const qwen3::Shape invalid{2, -1, 4};
                (void)invalid;
            },
            "negative shape dimension was accepted"
        );

        require_throws<std::overflow_error>(
            [] {
                const qwen3::Shape huge{
                    std::numeric_limits<std::int64_t>::max(),
                    2
                };

                (void)huge.numel();
            },
            "shape overflow was not detected"
        );

        std::cout << "Core metadata test passed.\n";
        std::cout
            << "dtype: "
            << qwen3::dtype_name(qwen3::DType::BFloat16)
            << '\n';
        std::cout
            << "device: "
            << cuda.to_string()
            << '\n';
        std::cout
            << "shape: "
            << hidden_states.to_string()
            << '\n';
        std::cout
            << "numel: "
            << hidden_states.numel()
            << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr
            << "Core metadata test failed: "
            << error.what()
            << '\n';

        return EXIT_FAILURE;
    }
}