#include "qwen3/tokenizer/tokenizer.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace qwen3
{
    namespace
    {
        using Json = nlohmann::json;

        const Json &required_field(const Json &object, const char *name)
        {
            const auto iterator = object.find(name);

            if (iterator == object.end())
            {
                throw std::runtime_error("missing tokenizer field '" + std::string{name} + "'");
            }

            return *iterator;
        }

        std::int32_t read_token_id(const Json &value, const char *field_name)
        {
            if (!value.is_number_integer() && !value.is_number_unsigned())
            {
                throw std::runtime_error(std::string{field_name} + " must be an integar");
            }

            const std::int64_t number = value.get<std::int64_t>();

            if (number < 0 || number > std::numeric_limits<std::int32_t>::max())
            {
                throw std::runtime_error(std::string{field_name} + " is outside the supported token ID range");
            }

            return static_cast<int32_t>(number);
        }

        // 严格读取一个 UTF-8 Unicode scalar
        char32_t read_utf8_scalar(std::string_view text, std::size_t &position)
        {
            if (position >= text.size())
            {
                throw std::runtime_error("unexpected end of UTF-8 string");
            }

            const auto byte = [&](std::size_t index) -> std::uint8_t
            {
                return static_cast<std::uint8_t>(text[index]);
            };

            const std::uint8_t first = byte(position);

            if (first <= 0x7FU)
            {
                ++position;
                return first;
            }

            int continuation_count = 0;
            char32_t code_point = 0;

            if (first >= 0xC2U && first <= 0xDFU)
            {
                continuation_count = 1;
                code_point = first & 0x1FU;
            }
            else if (first >= 0xE0U && first <= 0xEFU)
            {
                continuation_count = 2;
                code_point = first & 0x0FU;
            }
            else if (first >= 0xF0U && first <= 0xF4U)
            {
                continuation_count = 3;
                code_point = first & 0x07U;
            }
            else
            {
                throw std::runtime_error(
                    "invalid UTF-8 leading byte");
            }

            if (position +
                    static_cast<std::size_t>(
                        continuation_count) >=
                text.size())
            {
                throw std::runtime_error(
                    "truncated UTF-8 sequence");
            }

            for (int index = 1;
                 index <= continuation_count;
                 ++index)
            {
                const std::uint8_t next =
                    byte(position +
                         static_cast<std::size_t>(index));

                if ((next & 0xC0U) != 0x80U)
                {
                    throw std::runtime_error(
                        "invalid UTF-8 continuation byte");
                }

                code_point =
                    (code_point << 6U) |
                    static_cast<char32_t>(
                        next & 0x3FU);
            }

            if ((continuation_count == 2 &&
                 code_point < 0x800U) ||
                (continuation_count == 3 &&
                 code_point < 0x10000U))
            {
                throw std::runtime_error(
                    "overlong UTF-8 sequence");
            }

            if (code_point >= 0xD800U &&
                code_point <= 0xDFFFU)
            {
                throw std::runtime_error(
                    "UTF-8 sequence contains a surrogate");
            }

            if (code_point > 0x10FFFFU)
            {
                throw std::runtime_error(
                    "UTF-8 code point is out of range");
            }

            position +=
                static_cast<std::size_t>(
                    continuation_count + 1);

            return code_point;
        }

        // 构造 GPT-2/Qwen3 ByteLevel 反向映射：
        //
        // ByteLevel Unicode scalar -> 原始 byte
        std::unordered_map<char32_t, std::uint8_t> build_byte_decoder()
        {
            std::array<bool, 256> directly_mapped{};

            for (std::int32_t value = 33; value <= 126; ++value)
            {
                directly_mapped[static_cast<std::size_t>(value)] = true;
            }

            for (std::int32_t value = 161; value <= 172; ++value)
            {
                directly_mapped[static_cast<std::size_t>(value)] = true;
            }

            for (std::int32_t value = 174; value <= 255; ++value)
            {
                directly_mapped[static_cast<std::size_t>(value)] = true;
            }

            std::unordered_map<char32_t, std::uint8_t> decoder;

            decoder.reserve(256);

            char32_t extra_code_point = 256;

            for (std::int32_t value = 0; value <= 255; ++value)
            {
                char32_t encoded_code_point = 0;

                if (directly_mapped[static_cast<std::size_t>(value)])
                {
                    encoded_code_point = static_cast<char32_t>(value);
                }
                else
                {
                    encoded_code_point = extra_code_point++;
                }

                decoder.emplace(encoded_code_point, static_cast<std::uint8_t>(value));
            }

            return decoder;
        }

        std::string decode_bytelevel_token(std::string_view encoded_token, const std::unordered_map<char32_t, std::uint8_t> &byte_decoder)
        {
            std::string decoded;
            decoded.reserve(encoded_token.size());

            std::size_t position = 0;

            while (position < encoded_token.size())
            {
                const char32_t code_point = read_utf8_scalar(encoded_token, position);

                const auto iterator = byte_decoder.find(code_point);

                if (iterator == byte_decoder.end())
                {
                    throw std::runtime_error("base vocabulary token is not valid ByteLevel data");
                }

                decoded.push_back(static_cast<char>(iterator->second));
            }

            return decoded;
        }

        void append_utf8_scalar(std::string &output, char32_t code_point)
        {
            if (code_point <= 0x7FU)
            {
                output.push_back(static_cast<char>(code_point));
            }
            else if (code_point <= 0x7FFU)
            {
                output.push_back(static_cast<char>(0xC0U | (code_point >> 6U)));

                output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
            }
            else if (code_point <= 0xFFFFU)
            {
                output.push_back(
                    static_cast<char>(
                        0xE0U | (code_point >> 12U)));

                output.push_back(
                    static_cast<char>(
                        0x80U |
                        ((code_point >> 6U) & 0x3FU)));

                output.push_back(
                    static_cast<char>(
                        0x80U | (code_point & 0x3FU)));
            }
            else
            {
                output.push_back(
                    static_cast<char>(
                        0xF0U | (code_point >> 18U)));

                output.push_back(
                    static_cast<char>(
                        0x80U |
                        ((code_point >> 12U) & 0x3FU)));

                output.push_back(
                    static_cast<char>(
                        0x80U |
                        ((code_point >> 6U) & 0x3FU)));

                output.push_back(
                    static_cast<char>(
                        0x80U | (code_point & 0x3FU)));
            }
        }

        std::array<std::string, 256> build_byte_encoder()
        {
            std::array<bool, 256> directly_mapped{};

            for (std::int32_t value = 33; value <= 126; ++value)
            {
                directly_mapped[static_cast<std::size_t>(value)] = true;
            }

            for (std::int32_t value = 161; value <= 172; ++value)
            {
                directly_mapped[static_cast<std::size_t>(value)] = true;
            }

            for (std::int32_t value = 174; value <= 255; ++value)
            {
                directly_mapped[static_cast<std::size_t>(value)] = true;
            }

            std::array<std::string, 256> encoder;

            char32_t extra_code_point = 256;

            for (std::int32_t value = 0; value <= 255; ++value)
            {
                char32_t encoded_code_point = 0;

                if (directly_mapped[static_cast<std::size_t>(value)])
                {
                    encoded_code_point = static_cast<char32_t>(value);
                }
                else
                {
                    encoded_code_point = extra_code_point++;
                }

                append_utf8_scalar(encoder[static_cast<std::size_t>(value)], encoded_code_point);
            }

            return encoder;
        }

        bool read_required_bool(const nlohmann::json &object, const char *field_name)
        {
            const nlohmann::json &value = required_field(object, field_name);

            if (!value.is_boolean())
            {
                throw std::runtime_error(std::string{field_name} + " must be boolean");
            }

            return value.get<bool>();
        }

    } // namespace

    Tokenizer Tokenizer::load(const std::filesystem::path &tokenizer_json_path)
    {
        std::ifstream input(tokenizer_json_path, std::ios::binary);

        if (!input)
        {
            throw std::runtime_error("failed to open tokenizer file: " + tokenizer_json_path.string());
        }

        Json root;

        try
        {
            input >> root;
        }
        catch (const Json::exception &error)
        {
            throw std::runtime_error("failed to parse tokenizer JSON:" + std::string{error.what()});
        }

        const Json &model = required_field(root, "model");

        const Json &model_type = required_field(model, "type");

        if (!model_type.is_string() || model_type.get<std::string>() != "BPE")
        {
            throw std::runtime_error("tokenizer model must be BPE");
        }

        const Json &decoder = required_field(root, "decoder");

        const Json &decoder_type = required_field(decoder, "type");

        if (!decoder_type.is_string() || decoder_type.get<std::string>() != "ByteLevel")
        {
            throw std::runtime_error("tokenzier decoder must be ByteLevel");
        }

        const Json &normalizer = required_field(root, "normalizer");

        const Json &normalizer_type = required_field(normalizer, "type");

        if (!normalizer_type.is_string() || normalizer_type.get<std::string>() != "NFC")
        {
            throw std::runtime_error("Qwen3 tokenizer normalizer must be NFC");
        }

        const Json &pre_tokenizer = required_field(root, "pre_tokenizer");

        if (required_field(pre_tokenizer, "type").get<std::string>() != "Sequence")
        {
            throw std::runtime_error("Qwen3 pre-tokenizer must be a Sequence");
        }

        const Json &pretokenizers = required_field(pre_tokenizer, "pretokenizers");

        if (!pretokenizers.is_array() || pretokenizers.size() != 2)
        {
            throw std::runtime_error("Qwen3 pre-tokenizer sequence must contain Split and ByteLevel");
        }

        const Json &split = pretokenizers[0];
        const Json &byte_level = pretokenizers[1];

        if (required_field(split, "type").get<std::string>() != "Split")
        {
            throw std::runtime_error("first Qwen3 pre-tokenizer must be Split");
        }

        if (required_field(split, "behavior").get<std::string>() != "Isolated")
        {
            throw std::runtime_error(
                "Qwen3 Split behavior must be Isolated");
        }

        if (read_required_bool(split, "invert"))
        {
            throw std::runtime_error(
                "inverted tokenizer Split is unsupported");
        }

        const Json &pattern =
            required_field(split, "pattern");

        const Json &regex =
            required_field(pattern, "Regex");

        if (!regex.is_string())
        {
            throw std::runtime_error(
                "tokenizer Split regex must be a string");
        }

        if (required_field(byte_level, "type").get<std::string>() != "ByteLevel")
        {
            throw std::runtime_error(
                "second Qwen3 pre-tokenizer must be ByteLevel");
        }

        if (read_required_bool(
                byte_level,
                "add_prefix_space") ||
            read_required_bool(
                byte_level,
                "trim_offsets") ||
            read_required_bool(
                byte_level,
                "use_regex"))
        {
            throw std::runtime_error(
                "unsupported Qwen3 ByteLevel configuration");
        }

        const Json &vocabulary = required_field(model, "vocab");

        if (!vocabulary.is_object())
        {
            throw std::runtime_error("tokenizer model.vocab must be an object");
        }

        const Json &added_tokens = required_field(root, "added_tokens");

        if (!added_tokens.is_array())
        {
            throw std::runtime_error("tokenizer added_tokens must be an array");
        }

        Tokenizer tokenizer;

        tokenizer.pretokenizer_pattern_ = regex.get<std::string>();

        tokenizer.byte_token_ids_.fill(-1);

        tokenizer.base_vocab_size_ = static_cast<std::int32_t>(vocabulary.size());

        tokenizer.added_token_count_ = static_cast<std::int32_t>(added_tokens.size());

        tokenizer.token_to_id_.reserve(vocabulary.size() + added_tokens.size());

        const auto byte_decoder = build_byte_decoder();

        const auto resize_for_id = [&](std::int32_t token_id)
        {
            const std::size_t required_size = static_cast<std::size_t>(token_id) + 1;

            if (tokenizer.id_to_piece_.size() < required_size)
            {
                tokenizer.id_to_piece_.resize(required_size);

                tokenizer.id_is_present_.resize(required_size, false);

                tokenizer.id_is_special_.resize(required_size, false);
            }

            if (token_id > tokenizer.max_token_id_)
            {
                tokenizer.max_token_id_ = token_id;
            }
        };

        // =======================================================================================
        // 1. 加载普通 ByteLevel BPE vocabulary
        // =======================================================================================

        for (auto iterator = vocabulary.begin(); iterator != vocabulary.end(); ++iterator)
        {
            const std::string encoded_token = iterator.key();

            const std::int32_t token_id = read_token_id(iterator.value(), "vocabulary token ID");

            resize_for_id(token_id);

            if (tokenizer.id_is_present_[static_cast<std::size_t>(token_id)])
            {
                throw std::runtime_error("duplicate token ID in vocabulary");
            }

            tokenizer.id_to_piece_[static_cast<std::size_t>(token_id)] = decode_bytelevel_token(encoded_token, byte_decoder);

            tokenizer.id_is_present_[static_cast<std::size_t>(token_id)] = true;

            tokenizer.id_is_special_[static_cast<std::size_t>(token_id)] = false;

            const auto insertion = tokenizer.token_to_id_.emplace(encoded_token, token_id);

            if (!insertion.second)
            {
                throw std::runtime_error("duplicate token string in vocabulary");
            }

            ++tokenizer.token_count_;
        }

        const auto byte_encoder = build_byte_encoder();

        for (std::int32_t byte_value = 0; byte_value <= 255; ++byte_value)
        {
            const std::string &encoded_byte = byte_encoder[static_cast<std::size_t>(byte_value)];

            const auto iterator = tokenizer.token_to_id_.find(encoded_byte);

            if (iterator == tokenizer.token_to_id_.end())
            {
                throw std::runtime_error("tokenzier vocabulary is missing a ByteLevel base token");
            }

            tokenizer.byte_token_ids_[static_cast<std::size_t>(byte_value)] = iterator->second;
        }


        const Json &merges = required_field(model, "merges");

        if (!merges.is_array())
        {
            throw std::runtime_error("tokenizer model.merges must be an array");
        }

        if (merges.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        {
            throw std::runtime_error("tokenizer contains too many merge rules");
        }

        tokenizer.merge_rules_.reserve(merges.size());


        for (std::size_t rank = 0; rank < merges.size(); ++rank)
        {
            const Json &merge = merges[rank];

            if (!merge.is_array() || merge.size() != 2 || !merge[0].is_string() || !merge[1].is_string())
            {
                throw std::runtime_error("each BPE merge must contain two token strings");
            }

            const std::string left = merge[0].get<std::string>();

            const std::string right = merge[1].get<std::string>();

            const auto left_iterator = tokenizer.token_to_id_.find(left);

            const auto right_iterator = tokenizer.token_to_id_.find(right);

            const auto result_iterator = tokenizer.token_to_id_.find(left + right);

            if (left_iterator == tokenizer.token_to_id_.end() ||
                right_iterator == tokenizer.token_to_id_.end() ||
                result_iterator == tokenizer.token_to_id_.end())
            {
                throw std::runtime_error("BPE merge references an unknown vocabulary token");
            }

            const TokenPair pair{left_iterator->second, right_iterator->second};

            const MergeRule rule{static_cast<std::int32_t>(rank), result_iterator->second};

            const auto insertion = tokenizer.merge_rules_.emplace(pair, rule);

            if (!insertion.second)
            {
                throw std::runtime_error("duplicate BPE merge pair");
            }
        }


        tokenizer.merge_count_ = static_cast<std::int32_t>(merges.size());

        // =========================================================================================
        // 2. 加载 added tokens
        // =========================================================================================

        for (const Json &entry : added_tokens)
        {
            if (!entry.is_object())
            {
                throw std::runtime_error("added token entry must be an object");
            }

            const std::int32_t token_id = read_token_id(required_field(entry, "id"), "added token ID");

            const Json &content_value = required_field(entry, "content");

            const Json &special_value = required_field(entry, "special");

            const bool single_word = read_required_bool(entry, "single_word");

            const bool lstrip = read_required_bool(entry, "lstrip");

            const bool rstrip = read_required_bool(entry, "rstrip");

            const bool normalized = read_required_bool(entry, "normalized");

            if (!content_value.is_string())
            {
                throw std::runtime_error("added token content must be a string");
            }

            if (!special_value.is_boolean())
            {
                throw std::runtime_error("added token special flag must be boolean");
            }

            if (single_word || lstrip || rstrip || normalized)
            {
                throw std::runtime_error(
                    "this Tokenizer currently supports only "
                    "Qwen3 AddedToken entries with "
                    "single_word=false, lstrip=false, "
                    "rstrip=false and normalized=false");
            }

            const std::string content = content_value.get<std::string>();

            if (content.empty())
            {
                throw std::runtime_error(
                    "added token content cannot be empty");
            }

            const bool is_special = special_value.get<bool>();

            tokenizer.added_tokens_.push_back(AddedTokenInfo{content, token_id, is_special});

            resize_for_id(token_id);

            if (tokenizer.id_is_present_[static_cast<std::size_t>(token_id)])
            {
                throw std::runtime_error("duplicate added token ID");
            }

            // AddedToken 不是 ByteLevel 编码后的字符串
            // 解码时应该直接输出原始 content
            tokenizer.id_to_piece_[static_cast<std::size_t>(token_id)] = content;

            tokenizer.id_is_present_[static_cast<std::size_t>(token_id)] = true;

            tokenizer.id_is_special_[static_cast<std::size_t>(token_id)] = is_special;

            // AddedToken 应覆盖同名普通查找结果
            tokenizer.token_to_id_[content] = token_id;

            ++tokenizer.token_count_;
        }

        if (tokenizer.token_count_ == 0)
        {
            throw std::runtime_error("tokenizer contains no tokens");
        }

        return tokenizer;
    }

    std::int64_t Tokenizer::token_count() const noexcept
    {
        return token_count_;
    }

    std::int32_t Tokenizer::base_vocab_size() const noexcept
    {
        return base_vocab_size_;
    }

    std::int32_t Tokenizer::added_token_count() const noexcept
    {
        return added_token_count_;
    }

    std::int32_t Tokenizer::max_token_id() const noexcept
    {
        return max_token_id_;
    }

    std::int32_t Tokenizer::merge_count() const noexcept
    {
        return merge_count_;
    }

    bool Tokenizer::contains_token_id(
        std::int32_t token_id) const noexcept
    {
        if (token_id < 0)
        {
            return false;
        }

        const std::size_t index =
            static_cast<std::size_t>(token_id);

        return index < id_is_present_.size() &&
               id_is_present_[index];
    }

    bool Tokenizer::is_special_token(
        std::int32_t token_id) const
    {
        if (!contains_token_id(token_id))
        {
            throw std::out_of_range(
                "token ID has no tokenizer mapping: " +
                std::to_string(token_id));
        }

        return id_is_special_[static_cast<std::size_t>(token_id)];
    }

    std::int32_t Tokenizer::token_to_id(
        std::string_view token) const
    {
        const auto iterator =
            token_to_id_.find(std::string{token});

        if (iterator == token_to_id_.end())
        {
            throw std::out_of_range(
                "token string is not present in tokenizer");
        }

        return iterator->second;
    }

    const std::string &Tokenizer::token_piece(
        std::int32_t token_id) const
    {
        if (!contains_token_id(token_id))
        {
            throw std::out_of_range(
                "token ID has no tokenizer mapping: " +
                std::to_string(token_id));
        }

        return id_to_piece_[static_cast<std::size_t>(token_id)];
    }

    std::string Tokenizer::decode(const std::vector<std::int32_t> &token_ids, bool skip_special_tokens) const
    {
        std::size_t total_bytes = 0;

        for (const std::int32_t token_id : token_ids)
        {
            if (!contains_token_id(token_id))
            {
                throw std::out_of_range("token ID has no tokenizer mappint: " + std::to_string(token_id));
            }

            if (skip_special_tokens && is_special_token(token_id))
            {
                continue;
            }

            total_bytes += id_to_piece_[static_cast<std::size_t>(token_id)].size();
        }

        std::string decoded;
        decoded.reserve(total_bytes);

        for (const std::int32_t token_id : token_ids)
        {
            if (skip_special_tokens && is_special_token(token_id))
            {
                continue;
            }

            decoded += token_piece(token_id);
        }

        return decoded;
    }
}