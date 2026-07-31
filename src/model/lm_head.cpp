#include "qwen3/model/lm_head.h"

#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/cuda/cuda_check.h"
#include "qwen3/ops/linear.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qwen3
{
    namespace
    {
        std::size_t checked_to_size(std::int64_t value, const char *name)
        {
            if (value <= 0) {
                throw std::invalid_argument(std::string{name} + "must be positive");
            }

            const auto unsigned_value = static_cast<std::uint64_t>(value);

            if (unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                throw std::overflow_error(std::string{name} + " exceeds size_t range");
            }

            return static_cast<std::size_t>(unsigned_value);
        }

        std::size_t checked_multiply(std::size_t lhs, std::size_t rhs)
        {
            if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
                throw std::overflow_error("LM Head by size overflow");
            }

            return lhs * rhs;
        }

        void validate_hidden_states(const Tensor &hidden_states)
        {
            if (hidden_states.dtype() != DType::BFloat16) {
                throw std::invalid_argument("LM Head hidden states must be BF16");
            }

            if (!hidden_states.device().is_cuda()) {
                throw std::invalid_argument("LM Head hidden states must be on CUDA");
            }

            if (hidden_states.shape().rank() != 3)
            {
                throw std::invalid_argument(
                    "LM Head hidden states must have shape "
                    "[batch, sequence, hidden]");
            }

            if (hidden_states.shape()[0] <= 0 ||
                hidden_states.shape()[1] <= 0 ||
                hidden_states.shape()[2] <= 0)
            {
                throw std::invalid_argument(
                    "LM Head hidden-state dimensions "
                    "must be positive");
            }
        }

        void validate_lm_head_arguments(const Tensor &hidden_states, const Tensor &output_weight, const cuda::CublasHandle &handle)
        {
            validate_hidden_states(hidden_states);

            if (output_weight.dtype() !=
                DType::BFloat16)
            {
                throw std::invalid_argument(
                    "LM Head output weight must be BF16");
            }

            if (!output_weight.device().is_cuda())
            {
                throw std::invalid_argument(
                    "LM Head output weight must be on CUDA");
            }

            if (output_weight.device() !=
                hidden_states.device())
            {
                throw std::invalid_argument(
                    "LM Head hidden states and output "
                    "weight must use the same device");
            }

            if (handle.device() !=
                hidden_states.device())
            {
                throw std::invalid_argument(
                    "LM Head cuBLAS handle uses the "
                    "wrong device");
            }

            if (output_weight.shape().rank() != 2)
            {
                throw std::invalid_argument(
                    "LM Head output weight must have shape "
                    "[vocab_size, hidden_size]");
            }

            const std::int64_t hidden_size =
                hidden_states.shape()[2];

            if (output_weight.shape()[0] <= 0 ||
                output_weight.shape()[1] != hidden_size)
            {
                throw std::invalid_argument(
                    "LM Head output weight shape "
                    "does not match hidden states");
            }
        }
    }   // namespace

    Tensor select_last_token_hidden(const Tensor &hidden_states)
    {
        validate_hidden_states(hidden_states);

        const std::int64_t batch_size = hidden_states.shape()[0];

        const std::int64_t sequence_length = hidden_states.shape()[1];

        const std::int64_t hidden_size = hidden_states.shape()[2];

        Tensor output = Tensor::empty(Shape{batch_size, hidden_size}, DType::BFloat16, hidden_states.device());

        const std::size_t batch_count = checked_to_size(batch_size, "batch_size");

        const std::size_t sequence_count = checked_to_size(sequence_length, "sequence_length");

        const std::size_t hidden_count = checked_to_size(hidden_size, "hidden_size");

        const std::size_t row_bytes = checked_multiply(hidden_count, dtype_size(DType::BFloat16));

        // 一个 batch 的完整 [S, H] 占用多少字节
        const std::size_t source_pitch_bytes = checked_multiply(sequence_count, row_bytes);

        // 第一个 batch 最后一个 token 的起始偏移
        const std::size_t source_offset_bytes = checked_multiply(sequence_count - 1, row_bytes);

        int previous_device = 0;

        QWEN3_CUDA_CHECK(cudaGetDevice(&previous_device));

        const int target_device = hidden_states.device().index();

        const bool changed_device = previous_device != target_device;

        if (changed_device) {
            QWEN3_CUDA_CHECK(cudaSetDevice(target_device));
        }

        const auto *source = static_cast<const std::byte *>(hidden_states.data()) + source_offset_bytes;

        // 每次复制 H 个 BF16
        // source 每跨一行，跳过一个完整的 [S, H] batch
        // destination 每跨一行，只跳过一个[H]
        const cudaError_t copy_status = cudaMemcpy2DAsync(
            output.data(), row_bytes, source, source_pitch_bytes,
            row_bytes, batch_count, cudaMemcpyDeviceToDevice, nullptr
        );

        cudaError_t restore_status = cudaSuccess;

        if (changed_device) {
            restore_status = cudaSetDevice(previous_device);
        }

        QWEN3_CUDA_CHECK(copy_status);
        QWEN3_CUDA_CHECK(restore_status);

        return output;
    }

    Tensor qwen3_lm_head(const Tensor &final_hidden_states, const Tensor &output_weight, cuda::CublasHandle &cublas_handle)
    {
        validate_lm_head_arguments(final_hidden_states, output_weight, cublas_handle);

        // [B, S, H] -> [B, H]
        Tensor last_hidden = select_last_token_hidden(final_hidden_states);

        // [B, H] x [V, H] ^ T -> [B, V]
        return ops::linear(last_hidden, output_weight, cublas_handle);
    }
}   // namespace qwen