#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace qwen3
{
    class Tokenizer final
    {
    public:
        [[nodiscard]]
        static Tokenizer load(const std::filesystem::path &tokenizer_json_path);

        ~Tokenizer() = default;

        Tokenizer(const Tokenizer &) = delete;
        Tokenizer &operator=(const Tokenizer &) = delete;

        Tokenizer(Tokenizer &&) noexcept = default;
        Tokenizer &operator=(Tokenizer &&) noexcept = default;

        // tokenizer 中实际存在映射的 token 数量
        //
        // Qwen3 是 151669，不等于模型 vocab_size 151936
        [[nodiscard]]
        std::int64_t token_count() const noexcept;

        [[nodiscard]]
        std::int32_t base_vocab_size() const noexcept;

        [[nodiscard]]
        std::int32_t merge_count() const noexcept;

        [[nodiscard]]
        std::int32_t added_token_count() const noexcept;

        [[nodiscard]]
        std::int32_t max_token_id() const noexcept;

        [[nodiscard]]
        bool contains_token_id(std::int32_t token_id) const noexcept;

        [[nodiscard]]
        bool is_special_token(std::int32_t token_id) const;

        // 查找 tokenizer.json 中原始 token 字符串
        //
        // 主要用于：
        // tokenizer.token_to_id("<|im_start|>")
        [[nodiscard]]
        std::int32_t token_to_id(std::string_view token) const;

        // 返回该 token 解码后的原始字节片段
        //
        // ByteLevel token 的片段不保证单独构成完整的 UTF-8 字符
        // 因此正常显示是应该使用 decode（）解码完整token 序列
        [[nodiscard]]
        const std::string &token_piece(std::int32_t token_id) const;

        // 将一组 token ID 解码乘 UTF-8 字符串
        //
        // skip_special_tokens=true 时：
        // - 跳过 <|im_start|>,<|im_end|> 等 special=true token
        // - 不跳过 <think>,</think>, 因为他们special=false
        [[nodiscard]]
        std::string decode(const std::vector<std::int32_t> &token_ids, bool skip_special_tokens = false) const;

        // 不自动添加 BOS/EOS
        [[nodiscard]]
        std::vector<std::int32_t> encode(std::string_view text) const;

    private:
        struct TokenPair final
        {
            std::int32_t left = -1;
            std::int32_t right = -1;

            bool operator==(const TokenPair &other) const noexcept
            {
                return left == other.left && right == other.right;
            }
        };

        struct TokenPairHash final
        {
            std::size_t operator()(const TokenPair &pair) const noexcept
            {
                const std::uint64_t packed = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(pair.left)) << 32U) |
                                            static_cast<std::uint32_t>(pair.right);
                        
                return std::hash<std::uint64_t>{}(packed);
            }
        };
        
        struct MergeRule final
        {
            std::int32_t rank = -1;
            std::int32_t result_token_id = -1;
        };

        struct AddedTokenInfo final{
            std::string content;
            std::int32_t token_id = -1;
            bool special = false;
        };

    private:
        Tokenizer() = default;

        void encode_normal_text(std::string_view text, std::vector<std::int32_t> &output) const;

        void encode_pretoken(std::string_view text, std::vector<std::int32_t> &output) const;

    private:
        std::vector<std::string> id_to_piece_;
        std::vector<bool> id_is_present_;
        std::vector<bool> id_is_special_;

        std::unordered_map<std::string, std::int32_t> token_to_id_;

        // 原始 byte 0..255 对应的基础 token ID
        std::array<std::int32_t, 256> byte_token_ids_{};

        std::unordered_map<TokenPair, MergeRule, TokenPairHash> merge_rules_;

        std::vector<AddedTokenInfo> added_tokens_;

        // 从 tokenizer.json 原样读取
        std::string pretokenizer_pattern_;

        std::int64_t token_count_ = 0;
        std::int32_t base_vocab_size_ = 0;
        std::int32_t added_token_count_ = 0;
        std::int32_t merge_count_ = 0;
        std::int32_t max_token_id_ = -1;
    };
}