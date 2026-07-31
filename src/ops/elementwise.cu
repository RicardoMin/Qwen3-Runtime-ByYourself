#include "qwen3/ops/elementwise.h"

#include "qwen3/core/dtype.h"
#include "qwen3/cuda/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace qwen3::ops
{
    namespace
    {
        constexpr int kThreadsPerBlock = 256;
        constexpr std::int64_t kMaximumBlockCount = 65535;

        __global__ void add_inplace_bfloat16_kernel(
            __nv_bfloat16 *destination, const __nv_bfloat16 *source, std::int64_t element_count)
        {
            const std::int64_t start = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

            const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;

            for (std::int64_t index = start; index < element_count; index += stride)
            {
                const float destination_value = __bfloat162float(destination[index]);

                const float source_value = __bfloat162float(source[index]);

                destination[index] = __float2bfloat16_rn(destination_value + source_value);
            }
        }

        __global__ void sliu_multiply_bfloat16_kernel(
            const __nv_bfloat16 *gate, const __nv_bfloat16 *up, __nv_bfloat16 *output, std::int64_t element_count)
        {
            const std::int64_t start = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

            const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;

            for (std::int64_t index = start; index < element_count; index += stride)
            {
                const float gate_value = __bfloat162float(gate[index]);

                const float up_value = __bfloat162float(up[index]);

                const float silu_value = gate_value / (1.0F + expf(-gate_value));

                // PyTorch 的执行顺序相当于
                //
                // gaye_silu = silu（gate）     -> BF16
                // output = gate_silu * up      -> BF16
                //
                // 虽然我们将两步操作合进一个 kernel
                // 但是仍然需要中间的 BF16 舍入
                // 所以这里将silu结果舍入为BF16，再变成F32 继续下一步计算
                const __nv_bfloat16 silu_bfloat16 = __float2bfloat16_rn(silu_value);

                const float rounded_silu = __bfloat162float(silu_bfloat16);

                output[index] = __float2bfloat16_rn(rounded_silu * up_value);
            }
        }

        void validate_binary_bfloat16_cuda(const Tensor &lhs, const Tensor &rhs, const char *operation)
        {
            if (lhs.dtype() != DType::BFloat16 || rhs.dtype() != DType::BFloat16)
            {
                throw std::invalid_argument(
                    std::string(operation) +
                    " requires BFloat16 tensors");
            }

            if (!lhs.device().is_cuda() || !rhs.device().is_cuda())
            {
                throw std::invalid_argument(
                    std::string(operation) +
                    " currently requires CUDA tensors");
            }

            if (lhs.device() != rhs.device())
            {
                throw std::invalid_argument(
                    std::string(operation) +
                    " requires tensors on the same device");
            }

            if (lhs.shape() != rhs.shape())
            {
                throw std::invalid_argument(
                    std::string(operation) +
                    " requires identical tensor shapes");
            }
        }

        int block_count_for (std::int64_t element_count)
        {
            // 调用方必须先保证 element_count > 0
            const std::int64_t required_blocks = (element_count -1) / kThreadsPerBlock + 1;

            return static_cast<int>(std::min(required_blocks, kMaximumBlockCount));
        }

        int activate_cuda_device(const Device &device, bool &changed_device)
        {
            int previous_device = 0;

            QWEN3_CUDA_CHECK(cudaGetDevice(&previous_device));

            changed_device = previous_device !=  device.index();

            if (changed_device) {
                QWEN3_CUDA_CHECK(cudaSetDevice(device.index()));
            }

            return previous_device;
        }

        cudaError_t restore_cuda_device(int previous_device, bool changed_device)
        {
            if (!changed_device) {
                return cudaSuccess;
            }

            return cudaSetDevice(previous_device);
        }
    }   // namespace

    void add_inplace(Tensor &destination, const Tensor &source)
    {
        validate_binary_bfloat16_cuda(destination, source, "add_inplace");

        const std::int64_t element_count = destination.numel();

        if (element_count == 0) {
            return;
        }

        bool changed_device = false;

        const int previous_device = activate_cuda_device(destination.device(), changed_device);

        add_inplace_bfloat16_kernel<<<block_count_for(element_count), kThreadsPerBlock>>>(
            static_cast<__nv_bfloat16 *>(destination.data()), static_cast<const __nv_bfloat16 *>(source.data()), element_count
        );

        const cudaError_t launch_status = cudaGetLastError();

        const cudaError_t restore_status = restore_cuda_device(previous_device, changed_device);

        QWEN3_CUDA_CHECK(launch_status);
        QWEN3_CUDA_CHECK(restore_status);
    }

    Tensor silu_multiply(const Tensor &gate, const Tensor &up) 
    {
        validate_binary_bfloat16_cuda(gate, up, "silu_multiply");

        Tensor output = Tensor::empty(gate.shape(), gate.dtype(), gate.device());

        const std::int64_t element_count = gate.numel();

        if (element_count == 0) {
            return output;
        }

        bool changed_device = false;

        const int previous_device = activate_cuda_device(gate.device(), changed_device);

        sliu_multiply_bfloat16_kernel<<<block_count_for(element_count), kThreadsPerBlock>>> (
            static_cast<const __nv_bfloat16 *>(gate.data()),
            static_cast<const __nv_bfloat16 *>(up.data()),
            static_cast<__nv_bfloat16 *>(output.data()),
            element_count
        );

        const cudaError_t launch_status = cudaGetLastError();

        const cudaError_t restore_status = restore_cuda_device(previous_device, changed_device);

        QWEN3_CUDA_CHECK(launch_status);
        QWEN3_CUDA_CHECK(restore_status);

        return output;
    }

}   // namespace qwen3::ops