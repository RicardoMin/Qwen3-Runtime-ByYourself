#include "qwen3/model/config.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace qwen3
{

    namespace
    {

        using Json = nlohmann::json;

        void require_positive(std::int32_t value, const char *name)
        {
            if (value <= 0)
            {
                throw std::invalid_argument(
                    std::string(name) + "must be positive");
            }
        }

        const Json &required_field(const Json &root, const char *key)
        {
            const auto iterator = root.find(key);

            if (iterator == root.end())
            {
                throw std::runtime_error("missing required config field '" + std::string{key} + "'");
            }

            return *iterator;
        }

        std::int32_t read_int32(const Json &root, const char *key)
        {
            const Json &value = required_field(root, key);

            if (!value.is_number_integer() && !value.is_number_unsigned())
            {
                throw std::runtime_error("config field '" + std::string{key} + "' must be an integar");
            }

            const std::int64_t number = value.get<std::int64_t>();

            if (number <
                    std::numeric_limits<std::int32_t>::min() ||
                number >
                    std::numeric_limits<std::int32_t>::max())
            {
                throw std::runtime_error(
                    "config field '" + std::string{key} +
                    "' is outside the int32 range");
            }

            return static_cast<std::int32_t>(number);
        }

        float read_float(
            const Json &root,
            const char *key)
        {
            const Json &value = required_field(root, key);

            if (!value.is_number())
            {
                throw std::runtime_error(
                    "config field '" + std::string{key} +
                    "' must be numeric");
            }

            const double number = value.get<double>();

            if (!std::isfinite(number) ||
                std::fabs(number) >
                    std::numeric_limits<float>::max())
            {
                throw std::runtime_error(
                    "config field '" + std::string{key} +
                    "' cannot be represented as float");
            }

            return static_cast<float>(number);
        }

        bool read_bool(
            const Json &root,
            const char *key)
        {
            const Json &value = required_field(root, key);

            if (!value.is_boolean())
            {
                throw std::runtime_error(
                    "config field '" + std::string{key} +
                    "' must be boolean");
            }

            return value.get<bool>();
        }

        std::string read_string(
            const Json &root,
            const char *key)
        {
            const Json &value = required_field(root, key);

            if (!value.is_string())
            {
                throw std::runtime_error(
                    "config field '" + std::string{key} +
                    "' must be a string");
            }

            return value.get<std::string>();
        }

        void reject_non_null_field(const Json &root, const char *key)
        {
            const auto iterator = root.find(key);

            if (iterator != root.end() && !iterator->is_null())
            {
                throw std::invalid_argument("config field '" + std::string{key} + "' is not support yet");
            }
        }

        void require_qwen3_architecture(const Json &root)
        {
            const Json &architectures = required_field(root, "architectures");

            if (!architectures.is_array())
            {
                throw std::runtime_error("config field 'architechtures' must be an array");
            }

            for (const Json &architecture : architectures)
            {
                if (architecture.is_string() && architecture.get<std::string>() == "Qwen3ForCausalLM")
                {
                    return;
                }
            }

            throw std::invalid_argument("only Qwen3ForCausalLM is supported");
        }

    } // namespace

    std::int64_t ModelConfig::q_projection_size() const noexcept
    {
        return static_cast<std::int64_t>(num_attention_heads) * head_dim;
    }

    std::int64_t ModelConfig::kv_projection_size() const noexcept
    {
        return static_cast<std::int64_t>(num_key_value_heads) * head_dim;
    }

    std::int32_t ModelConfig::query_heads_per_kv_head() const noexcept
    {
        if (num_key_value_heads == 0)
        {
            return 0;
        }

        return num_attention_heads / num_key_value_heads;
    }

    void ModelConfig::validate() const
    {
        require_positive(vocab_size, "vocab_size");
        require_positive(hidden_size, "hidden_size");
        require_positive(
            intermediate_size,
            "intermediate_size");
        require_positive(
            num_hidden_layers,
            "num_hidden_layers");
        require_positive(
            num_attention_heads,
            "num_attention_heads");
        require_positive(
            num_key_value_heads,
            "num_key_value_heads");
        require_positive(head_dim, "head_dim");
        require_positive(
            max_position_embeddings,
            "max_position_embeddings");

        if (num_attention_heads %
                num_key_value_heads !=
            0)
        {
            throw std::invalid_argument(
                "num_attention_heads must be divisible by "
                "num_key_value_heads");
        }

        if (head_dim % 2 != 0)
        {
            throw std::invalid_argument(
                "head_dim must be even for RoPE");
        }

        if (!std::isfinite(rope_theta) ||
            rope_theta <= 0.0F)
        {
            throw std::invalid_argument(
                "rope_theta must be finite and positive");
        }

        if (!std::isfinite(rms_norm_eps) ||
            rms_norm_eps <= 0.0F)
        {
            throw std::invalid_argument(
                "rms_norm_eps must be finite and positive");
        }

        if (bos_token_id < 0 ||
            bos_token_id >= vocab_size)
        {
            throw std::invalid_argument(
                "bos_token_id is outside the vocabulary");
        }

        if (eos_token_id < 0 ||
            eos_token_id >= vocab_size)
        {
            throw std::invalid_argument(
                "eos_token_id is outside the vocabulary");
        }

        if (attention_bias)
        {
            throw std::invalid_argument(
                "attention bias is not supported");
        }

        if (use_sliding_window)
        {
            throw std::invalid_argument(
                "sliding-window attention is not supported");
        }
    }

    ModelConfig load_model_config(const std::filesystem::path &config_path)
    {
        std::ifstream input{config_path};

        if (!input)
        {
            throw std::runtime_error("cannot open model config: " + config_path.string());
        }

        Json root;

        try
        {
            input >> root;
        }
        catch (const Json::exception &error)
        {
            throw std::runtime_error("cannot parse model config '" + config_path.string() + "': " + error.what());
        }

        if (!root.is_object())
        {
            throw std::runtime_error("model config root must be a JSON object");
        }

        if (read_string(root, "model_type") != "qwen3")
        {
            throw std::invalid_argument(
                "only Qwen3 Dense models are supported");
        }
        
        require_qwen3_architecture(root);

        if (read_string(root, "hidden_act") != "silu")
        {
            throw std::invalid_argument(
                "only the SiLU activation is supported");
        }

        if (read_string(root, "torch_dtype") !=
            "bfloat16")
        {
            throw std::invalid_argument(
                "only BF16 Qwen3 checkpoints are supported");
        }

        // 这些特性以后单独实现
        reject_non_null_field(root, "rope_scaling");
        reject_non_null_field(root, "sliding_window");
        reject_non_null_field(root, "quantization_config");

        ModelConfig config;

        config.vocab_size =
            read_int32(root, "vocab_size");
        config.hidden_size =
            read_int32(root, "hidden_size");
        config.intermediate_size =
            read_int32(root, "intermediate_size");
        config.num_hidden_layers =
            read_int32(root, "num_hidden_layers");

        config.num_attention_heads =
            read_int32(root, "num_attention_heads");
        config.num_key_value_heads =
            read_int32(root, "num_key_value_heads");
        config.head_dim =
            read_int32(root, "head_dim");

        config.max_position_embeddings =
            read_int32(root, "max_position_embeddings");

        config.rope_theta =
            read_float(root, "rope_theta");
        config.rms_norm_eps =
            read_float(root, "rms_norm_eps");

        config.bos_token_id =
            read_int32(root, "bos_token_id");
        config.eos_token_id =
            read_int32(root, "eos_token_id");

        config.attention_bias =
            read_bool(root, "attention_bias");
        config.tie_word_embeddings =
            read_bool(root, "tie_word_embeddings");
        config.use_sliding_window =
            read_bool(root, "use_sliding_window");

        config.validate();
        return config;
    }

    ModelConfig make_qwen3_0_6b_config()
    {
        ModelConfig config;

        config.vocab_size = 151936;
        config.hidden_size = 1024;
        config.intermediate_size = 3072;
        config.num_hidden_layers = 28;

        config.num_attention_heads = 16;
        config.num_key_value_heads = 8;
        config.head_dim = 128;

        config.max_position_embeddings = 40960;
        config.rope_theta = 1'000'000.0F;
        config.rms_norm_eps = 1.0e-6F;

        config.bos_token_id = 151643;
        config.eos_token_id = 151645;

        config.attention_bias = false;
        config.tie_word_embeddings = true;
        config.use_sliding_window = false;

        config.validate();
        return config;
    }

} // namespace qwen3
