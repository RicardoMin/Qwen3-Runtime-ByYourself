#include "qwen3/model/mlp.h"

#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/ops/linear.h"
#include "qwen3/ops/elementwise.h"

#include <stdexcept>
#include <string>

namespace qwen3
{
    namespace 
    {
        void require_shape(const Tensor &tensor, const Shape &expected, const char *tensor_name)
        {
            if (tensor.shape() != expected) {
                throw std::invalid_argument(std::string(tensor_name) + " has shape " + tensor.shape().to_string() +
                                            ", expected " + expected.to_string());
            }
        }

        void validate_mlp_arguments(const Tensor &input, const Qwen3LayerWeights &weights, const ModelConfig &config)
        {
            if (input.dtype() != DType::BFloat16) {
                throw std::invalid_argument("Qwen3 MLP input must have BFloat16 dtype");
            }

             if (!input.device().is_cuda())
            {
                throw std::invalid_argument(
                    "Qwen3 MLP input must be on CUDA");
            }

            if (input.shape().rank() != 3)
            {
                throw std::invalid_argument(
                    "Qwen3 MLP input must have shape "
                    "[batch, sequence, hidden_size]");
            }

            if (input.shape()[2] != config.hidden_size)
            {
                throw std::invalid_argument(
                    "Qwen3 MLP input hidden dimension "
                    "does not match ModelConfig");
            }

            require_shape(
                weights.gate_proj,
                Shape{
                    config.intermediate_size,
                    config.hidden_size},
                "gate_proj");

            require_shape(
                weights.up_proj,
                Shape{
                    config.intermediate_size,
                    config.hidden_size},
                "up_proj");

            require_shape(
                weights.down_proj,
                Shape{
                    config.hidden_size,
                    config.intermediate_size},
                "down_proj");

            const Tensor *used_weights[] = {
                &weights.gate_proj,
                &weights.up_proj,
                &weights.down_proj};

            for (const Tensor *weight : used_weights)
            {
                if (weight->dtype() != DType::BFloat16)
                {
                    throw std::invalid_argument(
                        "Qwen3 MLP weights must have "
                        "BFloat16 dtype");
                }

                if (weight->device() != input.device())
                {
                    throw std::invalid_argument(
                        "Qwen3 MLP input and weights must "
                        "be on the same CUDA device");
                }
            }
        }
    }   // namespace

    Tensor qwen3_mlp(const Tensor &normalized_hidden_states, const Qwen3LayerWeights &weights, const ModelConfig &config, cuda::CublasHandle &cublas_handle)
    {
        validate_mlp_arguments(normalized_hidden_states, weights, config);

        // [B, S, H] x [I, H]^T
        // -> [B, S, I]

        Tensor gate = ops::linear(normalized_hidden_states, weights.gate_proj, cublas_handle);

        // [B, S, H] x [I, H]^T
        // -> [B, S, I]
        Tensor up = ops::linear(normalized_hidden_states, weights.up_proj, cublas_handle);

        // SwiGLU:
        // SILU(gate) ⨀ up
        //
        // [B, S, I] -> [B, S, I]
        Tensor activated = ops::silu_multiply(gate, up);

        // [B, S, I] x [H, I] ^T
        // -> [B, S, H]
        return ops::linear(activated, weights.down_proj, cublas_handle);
    }
}   // namespace qwen3