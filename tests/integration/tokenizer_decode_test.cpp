#include "qwen3/tokenizer/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void require(
        bool condition,
        const char *message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 2)
        {
            std::cerr
                << "Usage: tokenizer_decode_test "
                << "<tokenizer.json>\n";

            return 1;
        }

        qwen3::Tokenizer tokenizer =
            qwen3::Tokenizer::load(argv[1]);

        require(
            tokenizer.base_vocab_size() == 151643,
            "base vocabulary size is incorrect");

        require(
            tokenizer.added_token_count() == 26,
            "added token count is incorrect");

        require(
            tokenizer.token_count() == 151669,
            "mapped token count is incorrect");

        require(
            tokenizer.max_token_id() == 151668,
            "maximum tokenizer token ID is incorrect");

        require(
            tokenizer.token_to_id("<|endoftext|>") ==
                151643,
            "endoftext token ID is incorrect");

        require(
            tokenizer.token_to_id("<|im_start|>") ==
                151644,
            "im_start token ID is incorrect");

        require(
            tokenizer.token_to_id("<|im_end|>") ==
                151645,
            "im_end token ID is incorrect");

        require(
            tokenizer.token_to_id("<think>") ==
                151667,
            "think token ID is incorrect");

        // hello + 空格world + !
        const std::string english =
            tokenizer.decode(
                {14990, 1879, 0});

        require(
            english == "hello world!",
            "English ByteLevel decoding failed");

        const std::string chinese =
            tokenizer.decode(
                {56568, 52801});

        require(
            chinese == "你好",
            "Chinese ByteLevel decoding failed");

        const std::vector<std::int32_t> special_case{
            151644, // <|im_start|>
            14990,  // hello
            151645, // <|im_end|>
            151667, // <think>, special=false
            151668, // </think>, special=false
        };

        const std::string decoded_with_special =
            tokenizer.decode(
                special_case,
                false);

        require(
            decoded_with_special ==
                "<|im_start|>hello<|im_end|>"
                "<think></think>",
            "raw added-token decoding failed");

        const std::string decoded_without_special =
            tokenizer.decode(
                special_case,
                true);

        // im_start/im_end 被移除，
        // 但 think 标签不是 special=true，必须保留。
        require(
            decoded_without_special ==
                "hello<think></think>",
            "skip_special_tokens behavior is incorrect");

        // 模型 vocabulary 是 151936，
        // 但 tokenizer 只映射到 151668。
        require(
            !tokenizer.contains_token_id(151669),
            "padded vocabulary ID should not have "
            "a tokenizer mapping");

        bool unknown_id_rejected = false;

        try
        {
            (void)tokenizer.decode({151669});
        }
        catch (const std::out_of_range &)
        {
            unknown_id_rejected = true;
        }

        require(
            unknown_id_rejected,
            "unmapped padded token ID was not rejected");

        std::cout
            << "Tokenizer decode test passed.\n"
            << "Base vocabulary: "
            << tokenizer.base_vocab_size()
            << '\n'
            << "Added tokens: "
            << tokenizer.added_token_count()
            << '\n'
            << "Mapped tokens: "
            << tokenizer.token_count()
            << '\n'
            << "English: "
            << english
            << '\n'
            << "Chinese: "
            << chinese
            << '\n';

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Tokenizer decode test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}