#include "qwen3/cuda/cublas_handle.h"

#include "qwen3/cuda/cuda_check.h"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace qwen3::cuda
{
    namespace
    {
        const char *cublas_status_name(cublasStatus_t status) noexcept
        {
            switch (status)
            {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";

            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";

            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";

            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";

            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";

            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";

            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";

            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";

            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";

            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
            }

            return "CUBLAS_STATUS_UNKNOWN";
        }

        void report_cuda_cleanup_error(cudaError_t status, const char *operation) noexcept
        {
            if (status == cudaSuccess) {
                return;
            }

            std::fprintf(stderr, "CUDA cleanup failed during %s: %s\n", operation, cudaGetErrorString(status));
        }

        void report_cublas_cleanup_error(cublasStatus_t status, const char *operation) noexcept
        {
            if (status == CUBLAS_STATUS_SUCCESS) {
                return;
            }

            std::fprintf(stderr, "cuBLAS cleanup failed during %s: %s\n", operation, cublas_status_name(status));
        }

        class ScopedCudaDevice final
        {
        public:
            explicit ScopedCudaDevice(int target_device)
            {
                QWEN3_CUDA_CHECK(cudaGetDevice(&previous_device_));

                if (previous_device_ != target_device) {
                    QWEN3_CUDA_CHECK(cudaSetDevice(target_device));

                    changed_ = true;
                }
            }

            ~ScopedCudaDevice() noexcept
            {
                if (changed_) 
                {
                    report_cuda_cleanup_error(cudaSetDevice(previous_device_), "restoring CUDA device");
                }
            }

        private:
            int previous_device_ = 0;
            bool changed_ = false;
        };
    }   // namespace

    void check_cublas(cublasStatus_t status, const char *expression, const char *file, int line)
    {
        if (status == CUBLAS_STATUS_SUCCESS) {
            return;
        }

        std::ostringstream message;

        message << "cuBLAS call failed: " << expression << '\n' << cublas_status_name(status) << '\n' << "at " << file << ":" << line;

        throw std::runtime_error(message.str());
    }

    CublasHandle::CublasHandle(Device device)
        : device_(device)
    {
        if (!device_.is_cuda()) {
            throw std::invalid_argument("CublasHandle requires a CUDA device");
        }

        ScopedCudaDevice guard{device_.index()};

        cublasHandle_t created_handle = nullptr;

        QWEN3_CUBLAS_CHECK(cublasCreate(&created_handle));

        try
        {
            // linear() 中 alpha 和 beta 使用 host float
            QWEN3_CUBLAS_CHECK(cublasSetPointerMode(created_handle, CUBLAS_POINTER_MODE_HOST));

            // 当前所有自写 kernel 都使用默认的 stream
            // 因此 cuBLAS 也先绑定默认 stream
            QWEN3_CUBLAS_CHECK(cublasSetStream(created_handle, nullptr));

            const auto math_mode = static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);

            QWEN3_CUBLAS_CHECK(cublasSetMathMode(created_handle, math_mode));
        }
        catch (...)
        {
            report_cublas_cleanup_error(cublasDestroy(created_handle), "destroying partially initialized handle");
            throw;
        }

        handle_ = created_handle;       
    }

    CublasHandle::~CublasHandle() noexcept
    {
        if (handle_ == nullptr) {
            return;
        }

        int previous_device = 0;

        const cudaError_t get_status = cudaGetDevice(&previous_device);

        bool changed_device = false;

        if (get_status == cudaSuccess && previous_device != device_.index())
        {
            const cudaError_t set_status = cudaSetDevice(device_.index());

            if (set_status == cudaSuccess) {
                changed_device = true;
            }else {
                report_cuda_cleanup_error(set_status, "selecting cuBLAS device");
            }
        }
        else if (get_status != cudaSuccess) 
        {
            report_cuda_cleanup_error(get_status, "getting current CUDA device");
        }

        report_cublas_cleanup_error(cublasDestroy(handle_), "cublasDestroy");

        handle_ = nullptr;

        if (changed_device) {
            report_cuda_cleanup_error(cudaSetDevice(previous_device), "restoring CUDA device");
        }
    }

    cublasHandle_t CublasHandle::get() const noexcept
    {
        return handle_;
    }

    const Device &CublasHandle::device() const noexcept
    {
        return device_;
    }

}   // namespace qwen3::cuda