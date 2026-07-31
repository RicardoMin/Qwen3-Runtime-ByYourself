#include "qwen3/io/safetensors_file.h"
#include "qwen3/io/safetensors_metadata.h"

#include <cstddef>
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
        if (argc != 2)
        {
            std::cerr
                << "Usage: safetensors_metadata_smoke "
                << "<model.safetensors>\n";
            return 1;
        }

        const qwen3::SafetensorsFile file{argv[1]};
        const qwen3::SafetensorsMetadata metadata{file};

        require(
            metadata.contains("model.embed_tokens.weight"),
            "embedding weight is missing");

        require(
            metadata.contains(
                "model.layers.0.self_attn.q_proj.weight"),
            "layer 0 q_proj weight is missing");

        require(
            metadata.contains("model.norm.weight"),
            "final norm weight is missing");

        const auto &embedding =
            metadata.tensor("model.embed_tokens.weight");

        require(
            embedding.dtype == qwen3::DType::BFloat16,
            "embedding dtype is not BF16");

        require(
            embedding.shape.rank() == 2,
            "embedding rank is not 2");

        require(
            embedding.shape.numel() ==
                std::size_t{151936} * 1024,
            "embedding shape is incorrect");

        require(
            embedding.size_bytes() ==
                std::size_t{151936} * 1024 * 2,
            "embedding byte size is incorrect");

        std::cout
            << "Safetensors metadata smoke passed.\n"
            << "Tensor count: "
            << metadata.tensor_count() << '\n'
            << "Embedding elements: "
            << embedding.shape.numel() << '\n'
            << "Embedding bytes: "
            << embedding.size_bytes() << '\n';

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Safetensors metadata smoke failed: "
            << error.what() << '\n';

        return 1;
    }
}