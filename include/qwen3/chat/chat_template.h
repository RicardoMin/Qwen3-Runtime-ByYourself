#pragma once

#include <string>
#include <vector>

namespace qwen3::chat
{
    enum class Role
    {
        System,
        User,
        Assistant,
    };

    struct Message
    {
        Role role;
        std::string content;
    };

    struct TemplateOptions
    {
        // 是否在末尾添加
        // <|im_start|>assistant\n
        bool add_generation_prompt = true;

        // true:
        // 让模型自己生成 <think>...</think>
        //
        // false:
        // 在 prompt 中预先添加一个空的思考块
        // 强制模型直接生成最终回答
        bool enable_thinking = true;
    };

    [[nodiscard]]
    std::string apply_chat_template(const std::vector<Message> &message, const TemplateOptions &options = {});
}   // namespace qwen3::chat