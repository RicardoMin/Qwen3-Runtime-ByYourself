#include "qwen3/tokenizer/tokenizer.h"

#include <unicode/normalizer2.h>
#include <unicode/parseerr.h>
#include <unicode/regex.h>
#include <unicode/unistr.h>
#include <unicode/ustring.h>
#include <unicode/utypes.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace qwen3
{
    namespace
    {
        [[noreturn]]
        void throw_icu_error(
            const char *operation,
            UErrorCode status)
        {
            throw std::runtime_error(
                std::string{operation} +
                " failed: " +
                u_errorName(status));
        }

        icu::UnicodeString strict_unicode_from_utf8(
            std::string_view text)
        {
            if (text.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max()))
            {
                throw std::length_error(
                    "Tokenizer input is too large");
            }

            const std::int32_t source_length =
                static_cast<std::int32_t>(
                    text.size());

            UErrorCode status = U_ZERO_ERROR;
            std::int32_t required_length = 0;

            u_strFromUTF8(
                nullptr,
                0,
                &required_length,
                text.data(),
                source_length,
                &status);

            if (status == U_BUFFER_OVERFLOW_ERROR)
            {
                status = U_ZERO_ERROR;
            }
            else if (U_FAILURE(status))
            {
                throw std::invalid_argument(
                    std::string{
                        "Tokenizer input is not valid UTF-8: "} +
                    u_errorName(status));
            }

            std::vector<UChar> buffer(
                static_cast<std::size_t>(
                    required_length) + 1);

            std::int32_t actual_length = 0;

            u_strFromUTF8(
                buffer.data(),
                static_cast<std::int32_t>(
                    buffer.size()),
                &actual_length,
                text.data(),
                source_length,
                &status);

            if (U_FAILURE(status))
            {
                throw std::invalid_argument(
                    std::string{
                        "Tokenizer input is not valid UTF-8: "} +
                    u_errorName(status));
            }

            return icu::UnicodeString(
                buffer.data(),
                actual_length);
        }

        std::string unicode_to_utf8(
            const icu::UnicodeString &text)
        {
            std::string output;
            text.toUTF8String(output);
            return output;
        }
    }

    std::vector<std::int32_t> Tokenizer::encode(
        std::string_view text) const
    {
        std::vector<std::int32_t> output;

        if (text.empty())
        {
            return output;
        }

        std::size_t cursor = 0;

        while (cursor < text.size())
        {
            std::size_t best_position =
                std::string_view::npos;

            std::size_t best_added_index =
                std::string_view::npos;

            for (std::size_t index = 0;
                 index < added_tokens_.size();
                 ++index)
            {
                const AddedTokenInfo &added =
                    added_tokens_[index];

                const std::size_t position =
                    text.find(
                        added.content,
                        cursor);

                if (position ==
                    std::string_view::npos)
                {
                    continue;
                }

                const bool earlier =
                    best_position ==
                        std::string_view::npos ||
                    position < best_position;

                const bool same_position_but_longer =
                    position == best_position &&
                    best_added_index !=
                        std::string_view::npos &&
                    added.content.size() >
                        added_tokens_[
                            best_added_index]
                            .content.size();

                if (earlier ||
                    same_position_but_longer)
                {
                    best_position = position;
                    best_added_index = index;
                }
            }

            if (best_position ==
                std::string_view::npos)
            {
                encode_normal_text(
                    text.substr(cursor),
                    output);

                break;
            }

            if (best_position > cursor)
            {
                encode_normal_text(
                    text.substr(
                        cursor,
                        best_position - cursor),
                    output);
            }

            const AddedTokenInfo &added =
                added_tokens_[best_added_index];

            output.push_back(
                added.token_id);

            cursor =
                best_position +
                added.content.size();
        }

        return output;
    }

    void Tokenizer::encode_normal_text(
        std::string_view text,
        std::vector<std::int32_t> &output) const
    {
        if (text.empty())
        {
            return;
        }

        const icu::UnicodeString source =
            strict_unicode_from_utf8(text);

        UErrorCode status = U_ZERO_ERROR;

        const icu::Normalizer2 *nfc =
            icu::Normalizer2::getNFCInstance(
                status);

        if (U_FAILURE(status) || nfc == nullptr)
        {
            throw_icu_error(
                "ICU NFC initialization",
                status);
        }

        icu::UnicodeString normalized;

        nfc->normalize(
            source,
            normalized,
            status);

        if (U_FAILURE(status))
        {
            throw_icu_error(
                "ICU NFC normalization",
                status);
        }

        const icu::UnicodeString pattern_text =
            strict_unicode_from_utf8(
                pretokenizer_pattern_);

        UParseError parse_error{};

        std::unique_ptr<icu::RegexPattern> pattern{
            icu::RegexPattern::compile(
                pattern_text,
                0,
                parse_error,
                status)
        };

        if (U_FAILURE(status) || !pattern)
        {
            throw std::runtime_error(
                "failed to compile tokenizer regex at "
                "UTF-16 offset " +
                std::to_string(parse_error.offset) +
                ": " +
                u_errorName(status));
        }

        std::unique_ptr<icu::RegexMatcher> matcher{
            pattern->matcher(
                normalized,
                status)
        };

        if (U_FAILURE(status) || !matcher)
        {
            throw_icu_error(
                "ICU regex matcher creation",
                status);
        }

        std::int32_t cursor = 0;

        while (matcher->find(status))
        {
            if (U_FAILURE(status))
            {
                throw_icu_error(
                    "ICU regex matching",
                    status);
            }

            const std::int32_t begin =
                matcher->start(status);

            const std::int32_t end =
                matcher->end(status);

            if (U_FAILURE(status))
            {
                throw_icu_error(
                    "ICU regex match range",
                    status);
            }

            if (begin < cursor ||
                end <= begin)
            {
                throw std::logic_error(
                    "tokenizer regex produced an invalid range");
            }

            // Split behavior=Isolated：
            // 不能把匹配之间的 gap 丢掉。
            if (begin > cursor)
            {
                const icu::UnicodeString gap(
                    normalized,
                    cursor,
                    begin - cursor);

                const std::string gap_utf8 =
                    unicode_to_utf8(gap);

                encode_pretoken(
                    gap_utf8,
                    output);
            }

            const icu::UnicodeString match(
                normalized,
                begin,
                end - begin);

            const std::string match_utf8 =
                unicode_to_utf8(match);

            encode_pretoken(
                match_utf8,
                output);

            cursor = end;
        }

        if (U_FAILURE(status))
        {
            throw_icu_error(
                "ICU regex matching",
                status);
        }

        if (cursor < normalized.length())
        {
            const icu::UnicodeString tail(
                normalized,
                cursor,
                normalized.length() - cursor);

            const std::string tail_utf8 =
                unicode_to_utf8(tail);

            encode_pretoken(
                tail_utf8,
                output);
        }
    }

    void Tokenizer::encode_pretoken(
        std::string_view text,
        std::vector<std::int32_t> &output) const
    {
        if (text.empty())
        {
            return;
        }

        std::vector<std::int32_t> symbols;
        symbols.reserve(text.size());

        // ByteLevel：
        // 每个 UTF-8 原始 byte 先转成一个基础 token ID。
        for (const char character : text)
        {
            const std::uint8_t byte =
                static_cast<std::uint8_t>(
                    character);

            const std::int32_t token_id =
                byte_token_ids_[
                    static_cast<std::size_t>(byte)];

            if (token_id < 0)
            {
                throw std::logic_error(
                    "ByteLevel token table is incomplete");
            }

            symbols.push_back(token_id);
        }

        while (symbols.size() >= 2)
        {
            bool found_rule = false;

            TokenPair best_pair;
            MergeRule best_rule;

            for (std::size_t index = 0;
                 index + 1 < symbols.size();
                 ++index)
            {
                const TokenPair pair{
                    symbols[index],
                    symbols[index + 1]
                };

                const auto iterator =
                    merge_rules_.find(pair);

                if (iterator ==
                    merge_rules_.end())
                {
                    continue;
                }

                if (!found_rule ||
                    iterator->second.rank <
                        best_rule.rank)
                {
                    found_rule = true;
                    best_pair = pair;
                    best_rule = iterator->second;
                }
            }

            if (!found_rule)
            {
                break;
            }

            // 一轮中合并选中 pair 的所有非重叠出现。
            std::vector<std::int32_t> merged;
            merged.reserve(symbols.size());

            std::size_t index = 0;

            while (index < symbols.size())
            {
                if (index + 1 < symbols.size() &&
                    symbols[index] == best_pair.left &&
                    symbols[index + 1] == best_pair.right)
                {
                    merged.push_back(
                        best_rule.result_token_id);

                    index += 2;
                }
                else
                {
                    merged.push_back(
                        symbols[index]);

                    ++index;
                }
            }

            if (merged.size() >= symbols.size())
            {
                throw std::logic_error(
                    "BPE merge did not reduce symbol count");
            }

            symbols = std::move(merged);
        }

        output.insert(
            output.end(),
            symbols.begin(),
            symbols.end());
    }
}