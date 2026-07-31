#include "qwen3/core/device.h"
#include "qwen3/io/safetensors_loader.h"
#include "qwen3/model/config.h"
#include "qwen3/model/weights.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace
{

void require(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc != 3)
        {
            printf("argc=%d\n", argc);
            std::cerr
                << "Usage: qwen3_weights_smoke "
                << "<model.safetensors>\n";

            return 1;
        }

        // 使用前面已经测试通过的默认Qwen3-0.6B配置。
        const qwen3::ModelConfig config = qwen3::load_model_config(argv[2]);

        qwen3::SafetensorsCheckpoint checkpoint{argv[1]};

        std::size_t free_before = 0;
        std::size_t total_memory = 0;

        cudaMemGetInfo(
            &free_before,
            &total_memory);

        qwen3::Qwen3Weights weights =
            qwen3::Qwen3Weights::load(
                checkpoint,
                config,
                qwen3::Device::cuda(0));

        std::size_t free_after = 0;

        cudaMemGetInfo(
            &free_after,
            &total_memory);

        require(
            weights.layer_count() ==
                static_cast<std::size_t>(
                    config.num_hidden_layers),
            "incorrect number of model layers");

        require(
            weights.token_embedding()
                    .shape()
                    .numel() ==
                static_cast<std::size_t>(
                    config.vocab_size) *
                static_cast<std::size_t>(
                    config.hidden_size),
            "embedding shape is incorrect");

        require(
            weights.final_norm()
                    .shape()
                    .numel() ==
                static_cast<std::size_t>(
                    config.hidden_size),
            "final norm shape is incorrect");

        require(
            weights.output_weight().data() ==
                weights.token_embedding().data(),
            "output weight is not tied to embedding");

        const auto &layer0 =
            weights.layer(0);

        require(
            layer0.q_proj.shape().numel() ==
                config.q_projection_size() *
                static_cast<std::size_t>(
                    config.hidden_size),
            "layer 0 q_proj is incorrect");

        const double allocated_gib =
            static_cast<double>(
                free_before - free_after) /
            (1024.0 * 1024.0 * 1024.0);

        std::cout
            << "Qwen3 weights smoke passed.\n"
            << "Layers: "
            << weights.layer_count() << '\n'
            << "Embedding elements: "
            << weights.token_embedding()
                       .shape()
                       .numel()
            << '\n'
            << "Approximate GPU allocation: "
            << allocated_gib
            << " GiB\n";

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Qwen3 weights smoke failed: "
            << error.what() << '\n';

        return 1;
    }
}