#include "qwen3/chat/chat_template.h"

#include <stdexcept>

namespace qwen3::chat
{
    namespace 
    {
        const char *role_name(Role role)
        {
            switch (role)
            {
            case Role::System:
                return "system";

            case Role::User:
                return "user";

            case Role::Assistant:
                return "assistant";
            }

            throw std::invalid_argument("unknow chat role");
        }

        void validate_messages(const std::vector<Message> &messages, const TemplateOptions &options)
        {
            if (messages.empty())
            {
                throw std::invalid_argument(
                    "chat messages cannot be empty");
            }

            std::size_t index = 0;

            // 普通对话允许第一条是 system。
            if (messages.front().role == Role::System)
            {
                index = 1;
            }

            if (index == messages.size())
            {
                throw std::invalid_argument(
                    "chat must contain at least one user message");
            }

            Role expected_role = Role::User;

            for (; index < messages.size(); ++index)
            {
                const Role current_role = messages[index].role;

                if (current_role == Role::System)
                {
                    throw std::invalid_argument(
                        "system message is only supported "
                        "at the beginning of the conversation");
                }

                if (current_role != expected_role)
                {
                    throw std::invalid_argument(
                        "chat messages must alternate between "
                        "user and assistant");
                }

                expected_role =
                    current_role == Role::User
                        ? Role::Assistant
                        : Role::User;
            }

            // 如果需要让模型继续生成 assistant，
            // 那么历史消息最后必须停在 user。
            if (options.add_generation_prompt &&
                messages.back().role != Role::User)
            {
                throw std::invalid_argument(
                    "generation prompt requires the last "
                    "message to have role=user");
            }
        }
    }   // namespace

    std::string apply_chat_template(const std::vector<Message> &messages, const TemplateOptions &options)
    {
        validate_messages(messages, options);

        std::size_t estimated_size = 64;

        for (const Message &message : messages) {
            estimated_size += message.content.size() + 32;
        }

        std::string prompt;
        prompt.reserve(estimated_size);

        for (const Message &message : messages)
        {
            prompt += "<|im_start|>";
            prompt += role_name(message.role);
            prompt += '\n';
            prompt += message.content;
            prompt += "<|im_end|>\n";
        }

        if (options.add_generation_prompt) {
            prompt += "<|im_start|>assistant\n";

            if (!options.enable_thinking) {
                prompt += "<think>\n\n</think>\n\n";
            }
        }

        return prompt;
    }
}   // namespace qwen3::chat