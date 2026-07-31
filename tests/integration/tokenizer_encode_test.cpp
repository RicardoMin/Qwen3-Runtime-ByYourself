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

    void require_equal(
        const std::vector<std::int32_t> &actual,
        const std::vector<std::int32_t> &expected,
        const char *message)
    {
        if (actual != expected)
        {
            std::cerr << message << "\nactual: ";

            for (const auto value : actual)
            {
                std::cerr << value << ' ';
            }

            std::cerr << "\nexpected: ";

            for (const auto value : expected)
            {
                std::cerr << value << ' ';
            }

            std::cerr << '\n';

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
                << "Usage: tokenizer_encode_test "
                << "<tokenizer.json>\n";

            return 1;
        }

        qwen3::Tokenizer tokenizer =
            qwen3::Tokenizer::load(argv[1]);

        require(
            tokenizer.merge_count() == 151387,
            "merge count is incorrect");
        
        require(
            tokenizer.encode("").empty(),
            "empty input must not add BOS or EOS");

        require_equal(
            tokenizer.encode("hello world!"),
            {14990, 1879, 0},
            "ASCII BPE encoding failed");

        // Qwen3 的正则是单个 \p{N}，
        // 因此连续数字分别进行预分词。
        require_equal(
            tokenizer.encode("1234"),
            {16, 17, 18, 19},
            "Qwen3 digit pre-tokenization failed");

        require_equal(
            tokenizer.encode(
                "<|im_start|>hello<|im_end|>"),
            {151644, 14990, 151645},
            "AddedToken encoding failed");

        const std::vector<std::int32_t> composed =
            tokenizer.encode("café");

        const std::vector<std::int32_t> decomposed =
            tokenizer.encode(
                "cafe\xCC\x81");

        require(
            composed == decomposed,
            "NFC normalization failed");

        const std::string chinese =
            "你好，世界！";

        const auto chinese_ids =
            tokenizer.encode(chinese);

        require(
            !chinese_ids.empty(),
            "Chinese encoding returned no tokens");

        require(
            tokenizer.decode(chinese_ids) == chinese,
            "Chinese encode/decode round-trip failed");

        const std::string thinking_text =
            "<think>分析</think>";

        const auto thinking_ids =
            tokenizer.encode(thinking_text);

        require(
            !thinking_ids.empty(),
            "thinking text encoding failed");

        require(
            thinking_ids.front() == 151667,
            "<think> was not encoded atomically");

        require(
            thinking_ids.back() == 151668,
            "</think> was not encoded atomically");

        require(
            tokenizer.decode(
                thinking_ids,
                true) == thinking_text,
            "non-special AddedToken was incorrectly removed");

        std::cout
            << "Tokenizer encode test passed.\n"
            << "Merge rules: "
            << tokenizer.merge_count()
            << '\n'
            << "Chinese token count: "
            << chinese_ids.size()
            << '\n';

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Tokenizer encode test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}