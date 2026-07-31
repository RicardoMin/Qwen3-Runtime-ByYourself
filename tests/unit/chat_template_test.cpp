#include "qwen3/chat/chat_template.h"

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
        const std::string &actual,
        const std::string &expected,
        const char *test_name)
    {
        if (actual != expected)
        {
            throw std::runtime_error(
                std::string{test_name} +
                " produced an incorrect prompt\n"
                "actual:\n" +
                actual +
                "\nexpected:\n" +
                expected);
        }
    }

} // namespace

int main()
{
    using qwen3::chat::Message;
    using qwen3::chat::Role;
    using qwen3::chat::TemplateOptions;
    using qwen3::chat::apply_chat_template;

    try
    {
        // ---------------------------------------------------------
        // 1. 单轮普通思考模式
        // ---------------------------------------------------------

        const std::vector<Message> single_turn{
            {Role::User, "你好"},
        };

        require_equal(
            apply_chat_template(single_turn),
            "<|im_start|>user\n"
            "你好<|im_end|>\n"
            "<|im_start|>assistant\n",
            "single-turn thinking template");

        // ---------------------------------------------------------
        // 2. 关闭 thinking
        // ---------------------------------------------------------

        TemplateOptions no_thinking;
        no_thinking.enable_thinking = false;

        require_equal(
            apply_chat_template(single_turn, no_thinking),
            "<|im_start|>user\n"
            "你好<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n",
            "single-turn non-thinking template");

        // ---------------------------------------------------------
        // 3. system + user
        // ---------------------------------------------------------

        const std::vector<Message> with_system{
            {Role::System, "You are helpful."},
            {Role::User, "Hello"},
        };

        require_equal(
            apply_chat_template(with_system, no_thinking),
            "<|im_start|>system\n"
            "You are helpful.<|im_end|>\n"
            "<|im_start|>user\n"
            "Hello<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n",
            "system template");

        // ---------------------------------------------------------
        // 4. 多轮对话
        // ---------------------------------------------------------

        const std::vector<Message> multi_turn{
            {Role::System, "S"},
            {Role::User, "Q1"},
            {Role::Assistant, "A1"},
            {Role::User, "Q2"},
        };

        require_equal(
            apply_chat_template(multi_turn),
            "<|im_start|>system\n"
            "S<|im_end|>\n"
            "<|im_start|>user\n"
            "Q1<|im_end|>\n"
            "<|im_start|>assistant\n"
            "A1<|im_end|>\n"
            "<|im_start|>user\n"
            "Q2<|im_end|>\n"
            "<|im_start|>assistant\n",
            "multi-turn template");

        // ---------------------------------------------------------
        // 5. 非法消息顺序
        // ---------------------------------------------------------

        bool invalid_order_rejected = false;

        try
        {
            const std::vector<Message> invalid_messages{
                {Role::User, "Q1"},
                {Role::User, "Q2"},
            };

            (void)apply_chat_template(invalid_messages);
        }
        catch (const std::invalid_argument &)
        {
            invalid_order_rejected = true;
        }

        require(
            invalid_order_rejected,
            "invalid message order was not rejected");

        std::cout << "Chat template test passed.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Chat template test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}