#include "qwen3/model/config.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 3) {
            std::cerr
                << "Usage: model_config_test "
                << "<qwen3-0.6b-config> "
                << "<qwen3-8b-config>\n";
            return EXIT_FAILURE;
        }

        const qwen3::ModelConfig small =
            qwen3::load_model_config(argv[1]);

        require(
            small.hidden_size == 1024,
            "0.6B hidden_size is incorrect");
        require(
            small.num_hidden_layers == 28,
            "0.6B layer count is incorrect");
        require(
            small.q_projection_size() == 2048,
            "0.6B Q projection size is incorrect");
        require(
            small.kv_projection_size() == 1024,
            "0.6B KV projection size is incorrect");
        require(
            small.query_heads_per_kv_head() == 2,
            "0.6B GQA group size is incorrect");
        require(
            small.tie_word_embeddings,
            "0.6B should use tied embeddings");

        const qwen3::ModelConfig large =
            qwen3::load_model_config(argv[2]);

        require(
            large.hidden_size == 4096,
            "8B hidden_size is incorrect");
        require(
            large.num_hidden_layers == 36,
            "8B layer count is incorrect");
        require(
            large.q_projection_size() == 4096,
            "8B Q projection size is incorrect");
        require(
            large.kv_projection_size() == 1024,
            "8B KV projection size is incorrect");
        require(
            large.query_heads_per_kv_head() == 4,
            "8B GQA group size is incorrect");
        require(
            !large.tie_word_embeddings,
            "8B should use an independent lm_head");

        qwen3::ModelConfig invalid = large;
        invalid.num_key_value_heads = 6;

        bool rejected = false;

        try {
            invalid.validate();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }

        require(
            rejected,
            "invalid GQA configuration was accepted");

        std::cout
            << "Qwen3 Dense config test passed.\n"
            << "0.6B: layers="
            << small.num_hidden_layers
            << ", hidden=" << small.hidden_size
            << ", tied=" << small.tie_word_embeddings
            << '\n'
            << "8B: layers="
            << large.num_hidden_layers
            << ", hidden=" << large.hidden_size
            << ", tied=" << large.tie_word_embeddings
            << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr
            << "ModelConfig test failed: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}