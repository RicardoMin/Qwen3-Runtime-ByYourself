#pragma once

#include "qwen3/core/device.h"

#include <cublas_v2.h>

namespace qwen3::cuda
{
    void check_cublas(cublasStatus_t status, const char *expression, const char *file, int line);

    class CublasHandle final
    {
    public:
        explicit CublasHandle(Device device);

        ~CublasHandle() noexcept;

        CublasHandle(const CublasHandle &) = delete;

        CublasHandle &operator=(const CublasHandle &) = delete;

        CublasHandle(CublasHandle &&) = delete;

        CublasHandle &operator=(CublasHandle &&) = delete;

        [[nodiscard]]
        cublasHandle_t get() const noexcept;

        [[nodiscard]]
        const Device &device() const noexcept;

    private:
        Device device_;
        cublasHandle_t handle_ = nullptr;
    };

}   // namespace qwen3::cuda

#define QWEN3_CUBLAS_CHECK(expression)  \
    ::qwen3::cuda::check_cublas((expression), #expression, __FILE__, __LINE__)