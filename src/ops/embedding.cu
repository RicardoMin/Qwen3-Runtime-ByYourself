#include "qwen3/ops/embedding.h"

#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/cuda/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qwen3::ops
{
    namespace 
    {
        constexpr int kThreadsPerBlock = 256;
        constexpr std::int64_t kMaximumBlockCount = 65535;

        __global__ void embedding_bfloat16_kernel(const std::int32_t *token_ids, const __nv_bfloat16 *embedding_weight
                                                , __nv_bfloat16 *output, std::int64_t token_count, 
                                                std::int64_t vocab_size, std::int64_t hidden_size)
                    {
                        // 一个 block 负责一个或多个 token
                        // block 数不足时通过 grid-stride 方式继续处理后续 token
                        for (std::int64_t token_index = static_cast<std::int64_t>(blockIdx.x); token_index < token_count; 
                            token_index += static_cast<std::int64_t>(gridDim.x))
                        {
                            const std::int32_t token_id = token_ids[token_index];

                            const std::int64_t output_offset = token_index * hidden_size;

                            // 正常推理时，tokenizer 和采样器必须保证 ID 合法
                            // 这里仍然做边界保护，防止非法 ID 导致越界读取
                            if (token_id < 0 || static_cast<std::int64_t>(token_id) >= vocab_size) {
                                for (std::int64_t hidden_index = threadIdx.x; hidden_index < hidden_size; hidden_index += blockDim.x)
                                {
                                    output[output_offset + hidden_index] = __float2bfloat16(0.0F);
                                }

                                continue;
                            }

                            const std::int64_t weight_offset = static_cast<std::int64_t>(token_id) * hidden_size;

                            // 同一个 block 中的线程并行复制这一行的不同元素
                            for (std::int64_t hidden_index = threadIdx.x; hidden_index < hidden_size; hidden_index += blockDim.x)
                            {
                                output[output_offset + hidden_index] = embedding_weight[weight_offset + hidden_index];
                            }
                        }
                    }

        void validate_embedding_arguments(const Tensor& token_ids, const Tensor &embedding_weight)
        {
            if (token_ids.dtype() != DType::Int32) {
                throw std::invalid_argument("embedding token_ids must have Int32 dtype");
            }

            if (!token_ids.device().is_cuda() || !embedding_weight.device().is_cuda()) {
                throw std::invalid_argument("embedding currently requires CUDA tensors");
            }

            if (token_ids.device() != embedding_weight.device()) {
                throw std::invalid_argument("embedding token_ids ans weight must be on the same CUDA device");
            }

            if (token_ids.shape().rank() == 0) {
                throw std::invalid_argument("embedding token_ids cannot be scalar");
            }

            if (embedding_weight.shape().rank() != 2) {
                throw std::invalid_argument("embedding weight must have shape [vocab_size, hidden_size]");
            }

            if (embedding_weight.shape()[0] <= 0 || embedding_weight.shape()[1] <= 0) {
                throw std::invalid_argument("embedding weight dimensions must be positive");
            }
        }
    }   // namespace

    Tensor embedding_lookup(const Tensor &token_ids, const Tensor &embedding_weight)
    {
        validate_embedding_arguments(token_ids, embedding_weight);

        const std::int64_t token_count = token_ids.numel();

        const std::int64_t vocab_size = embedding_weight.shape()[0];

        const std::int64_t hidden_size = embedding_weight.shape()[1];

        // 假设 token_ids.shape 是 [batch, sequence]
        // 在末尾添加 hidden_size 后得到：
        // [batch, sequence, hidden_size]
        std::vector<Shape::Dimension> output_dimensions = token_ids.shape().dimensions();

        output_dimensions.push_back(hidden_size);

        Shape output_shape{std::move(output_dimensions)};

        Tensor output = Tensor::empty(std::move(output_shape), DType::BFloat16, token_ids.device());

        if (token_count == 0) {
            return output;
        }

        int previous_device = 0;

        QWEN3_CUDA_CHECK(cudaGetDevice(&previous_device));

        const int target_device = token_ids.device().index();

        const bool changed_device = previous_device != target_device;

        if (changed_device) 
        {
            QWEN3_CUDA_CHECK(cudaSetDevice(target_device));
        }

        const std::int64_t requested_blocks = std::min(token_count, kMaximumBlockCount);

        const int block_count = static_cast<int>(requested_blocks);

        embedding_bfloat16_kernel<<<block_count, kThreadsPerBlock>>>(
            static_cast<const std::int32_t *>(token_ids.data()),
            static_cast<const __nv_bfloat16 *>(embedding_weight.data()),
            static_cast<__nv_bfloat16 *>(output.data()),
            token_count,
            vocab_size,
            hidden_size
        );

        // 只检查 kernel 是否成功提交，不在算子内部同步 GPU
        const cudaError_t launch_status = cudaGetLastError();

        cudaError_t restore_status = cudaSuccess;

        if (changed_device) {
            restore_status = cudaSetDevice(previous_device);
        }

        QWEN3_CUDA_CHECK(launch_status);
        QWEN3_CUDA_CHECK(restore_status);

        return output;
    }
}   // namesapce qwen3::ops