#include "qwen3/ops/linear.h"

#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/cuda/cuda_check.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qwen3::ops
{
    namespace
    {
        void validate_linear_argument(const Tensor &input, const Tensor &weight, const cuda::CublasHandle &handle)
        {
            if (input.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16)
            {
                throw std::invalid_argument(
                    "linear requires BFloat16 tensors");
            }

            if (!input.device().is_cuda() || !weight.device().is_cuda())
            {
                throw std::invalid_argument(
                    "linear currently requires CUDA tensors");
            }

            if (input.device() != weight.device())
            {
                throw std::invalid_argument(
                    "linear input and weight must be "
                    "on the same CUDA device");
            }

            if (input.device() != handle.device())
            {
                throw std::invalid_argument(
                    "linear tensors and CublasHandle "
                    "must use the same CUDA device");
            }

            if (input.shape().rank() == 0)
            {
                throw std::invalid_argument(
                    "linear input cannot be scalar");
            }

            if (weight.shape().rank() != 2)
            {
                throw std::invalid_argument(
                    "linear weight must have shape "
                    "[out_features, in_features]");
            }

            const std::int64_t input_features =
                input.shape()[input.shape().rank() - 1];

            if (input_features <= 0)
            {
                throw std::invalid_argument(
                    "linear input feature size "
                    "must be positive");
            }

            if (weight.shape()[0] <= 0 ||
                weight.shape()[1] <= 0)
            {
                throw std::invalid_argument(
                    "linear weight dimensions "
                    "must be positive");
            }

            if (weight.shape()[1] != input_features)
            {
                throw std::invalid_argument(
                    "linear input last dimension "
                    "does not match weight in_features");
            }
        }

        int check_cublas_dimension(std::int64_t value, const char *name) 
        {
            if (value < 0 || value > std::numeric_limits<int>::max()) {
                throw std::overflow_error(std::string(name) + " exceeds cuBLAS int range");
            }

            return static_cast<int>(value);
        }
    }   // namespace

    Tensor linear(const Tensor &input, const Tensor &weight, cuda::CublasHandle &handle)
    {
        validate_linear_argument(input, weight, handle);

        const std::int64_t input_features = input.shape()[input.shape().rank() - 1];

        const std::int64_t output_features = weight.shape()[0];

        // 把 input 的所有前导维展平为矩阵的 M
        //
        // [batch, sequence, hidden]
        //             to
        // [batch * sequence, hidden]
        const std::int64_t row_count = input.numel() / input_features;

        std::vector<Shape::Dimension> output_dimensions = input.shape().dimensions();

        output_dimensions.back() = output_features;

        Tensor output = Tensor::empty(Shape{std::move(output_dimensions)}, DType::BFloat16, input.device());

        if (row_count == 0) {
            return output;
        }

        const int n = check_cublas_dimension(output_features, "linear output_features");

        const int m = check_cublas_dimension(row_count, "linear row_count");

        const int k = check_cublas_dimension(input_features, "linear input_features");

        int previous_device = 0;

        QWEN3_CUDA_CHECK(cudaGetDevice(&previous_device));

        const bool changed_device = previous_device != input.device().index();

        if (changed_device) {
            QWEN3_CUDA_CHECK(cudaSetDevice(input.device().index()));
        }

        const float alpha = 1.0F;
        const float beta = 0.0F;
        /*
        * 我们的 Tensor 是 row-major：
        *
        * input  = X，shape [M, K]
        * weight = W，shape [N, K]
        * output = Y，shape [M, N]
        *
        * 目标：
        *
        * Y = X × W^T
        *
        * cuBLAS 默认把内存解释为 column-major。
        * 同一块 row-major 内存会被解释成：
        *
        * X row-major [M,K] -> X^T column-major [K,M]
        * W row-major [N,K] -> W^T column-major [K,N]
        * Y row-major [M,N] -> Y^T column-major [N,M]
        *
        * 所以让 cuBLAS 计算：
        *
        * Y^T = W × X^T
        * 因为gemm是按照列看待矩阵，所以生成的是列意义上的转置答案，实际就是我们需要的答案
        */
       const cublasStatus_t gemm_status = cublasGemmEx(
                    handle.get(),

                    CUBLAS_OP_T,
                    CUBLAS_OP_N,

                    n,  // N
                    m,  // M
                    k,  // K

                    &alpha,

                    weight.data(),
                    CUDA_R_16BF,
                    k,

                    input.data(),
                    CUDA_R_16BF,
                    k,

                    &beta,

                    output.data(),
                    CUDA_R_16BF,
                    n,

                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT
       );

        // const auto *input_data = static_cast<const __nv_bfloat16 *>(input.data());

        // auto *output_data = static_cast<__nv_bfloat16 *>(output.data());

        // cublasStatus_t gemm_status = CUBLAS_STATUS_SUCCESS;

        // for (int row = 0; row < m; ++row) 
        // {
        //     gemm_status = cublasGemmEx(
        //         handle.get(),

        //         CUBLAS_OP_T,
        //         CUBLAS_OP_N,

        //         n,
        //         1,
        //         k,

        //         &alpha,

        //         weight.data(),
        //         CUDA_R_16BF,
        //         k,

        //         input_data + static_cast<std::size_t>(row) * static_cast<std::size_t>(k),
        //         CUDA_R_16BF,
        //         k,

        //         &beta,

        //         output_data + static_cast<std::size_t>(row) * static_cast<std::size_t>(n),
        //         CUDA_R_16BF,
        //         n,

        //         CUBLAS_COMPUTE_32F,
        //         CUBLAS_GEMM_DEFAULT_TENSOR_OP
        //     );

        //     if (gemm_status != CUBLAS_STATUS_SUCCESS) {
        //         break;
        //     }
        // }

       cudaError_t restore_status = cudaSuccess;

       if (changed_device) {
            restore_status = cudaSetDevice(previous_device);
       }

       QWEN3_CUBLAS_CHECK(gemm_status);

       QWEN3_CUDA_CHECK(restore_status);

       return output;
    }
}   // namespace qwen3::ops