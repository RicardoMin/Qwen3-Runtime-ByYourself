#include "qwen3/chat/chat_template.h"

#include "qwen3/core/device.h"
#include "qwen3/core/dtype.h"
#include "qwen3/core/shape.h"
#include "qwen3/core/tensor.h"

#include "qwen3/cuda/cublas_handle.h"
#include "qwen3/generation/generator.h"
#include "qwen3/generation/sampler.h"
#include "qwen3/io/safetensors_checkpoint.h"
#include "qwen3/model/config.h"
#include "qwen3/model/kv_cache.h"
#include "qwen3/model/weights.h"
#include "qwen3/ops/rope.h"
#include "qwen3/tokenizer/tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>

#include <clocale>
#include <termios.h>
#include <unistd.h>

namespace
{
    // 这是应用层主动设置的显存上限
    // 不是 Qwen3 模型自身的最大上下文长度
    constexpr std::int64_t kDefaultMaxContext = 4096;

    constexpr std::int64_t kDefaultMaxNewTokens = 512;

    struct AppOptions
    {
        std::filesystem::path model_directory;

        // 逻辑 CUDA 设备编号
        std::int32_t gpu_index = 0;

        std::int64_t max_context = kDefaultMaxContext;
        bool max_context_was_set = false;

        std::int64_t max_new_tokens = kDefaultMaxNewTokens;

        bool greedy = false;

        float temperature = 0.6F;
        std::int32_t top_k = 20;
        float top_p = 0.95F;
        std::uint64_t seed = 42;

        bool enable_thinking = true;

        // 只控制终端显示，不影响模型推理
        bool show_thinking = true;

        std::string system_prompt;

        // 在 Qwen3 默认 stop token 之外追加的停止 ID
        std::vector<std::int32_t> additional_stop_token_ids;

        bool show_help = false;
    };

    void print_usage(const char *program_name)
    {
        std::cout
            << "Usage:\n"
            << "  " << program_name
            << " MODEL_DIRECTORY [OPTIONS]\n\n"

            << "  " << program_name
            << " --model MODEL_DIRECTORY [OPTIONS]\n\n"

            << "Options:\n"
            << "  --model PATH\n"
            << "      Qwen3 model directory.\n\n"

            << "  --gpu INDEX\n"
            << "      Logical CUDA device index.\n"
            << "      Default: 0\n\n"

            << "  --max-context TOKENS\n"
            << "      KV cache and RoPE context capacity.\n"
            << "      Default: 4096\n\n"

            << "  --max-new-tokens TOKENS\n"
            << "      Maximum tokens generated per response.\n"
            << "      Default: 512\n\n"

            << "  --sample\n"
            << "      Use temperature/Top-k/Top-p sampling.\n"
            << "      This is the default mode.\n\n"

            << "  --greedy\n"
            << "      Always select the highest-logit token.\n\n"

            << "  --temperature VALUE\n"
            << "      Sampling temperature.\n"
            << "      Default: 0.6\n\n"

            << "  --top-k VALUE\n"
            << "      Top-k sampling size. 0 disables Top-k.\n"
            << "      Default: 20\n\n"

            << "  --top-p VALUE\n"
            << "      Top-p probability threshold.\n"
            << "      1.0 disables Top-p.\n"
            << "      Default: 0.95\n\n"

            << "  --seed VALUE\n"
            << "      Random sampling seed.\n"
            << "      Default: 42\n\n"

            << "  --thinking\n"
            << "      Enable Qwen3 thinking mode.\n"
            << "      This is the default.\n\n"

            << "  --no-thinking\n"
            << "      Prefill an empty thinking block and ask\n"
            << "      the model to answer directly.\n\n"

            << "  --show-thinking\n"
            << "      Display the model's thinking content.\n"
            << "      This is the default.\n\n"

            << "  --hide-thinking\n"
            << "      Let the model think, but display only\n"
            << "      the final answer.\n\n"

            << "  --system TEXT\n"
            << "      Optional system prompt.\n\n"

            << "  --stop-token ID\n"
            << "      Add an extra stop token ID.\n"
            << "      This option may be repeated.\n\n"

            << "  -h, --help\n"
            << "      Show this help message.\n\n"

            << "Interactive commands:\n"
            << "  /exit       Exit the program\n"
            << "  /clear      Clear conversation history\n"
            << "  /think on   Enable thinking mode\n"
            << "  /think off  Disable thinking mode\n"
            << "  /help       Show interactive help\n";
    }

    std::string require_option_value(int argc, char **argv, int &index, const std::string &option_name)
    {
        if (index + 1 >= argc)
        {
            throw std::invalid_argument("missing value after " + option_name);
        }

        ++index;
        return argv[index];
    }

    std::int64_t parse_int64(const std::string &text, const std::string &option_name)
    {
        std::size_t consumed = 0;
        long long value = 0;

        try
        {
            value = std::stoll(text, &consumed, 10);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument(option_name + " requires a valid integer");
        }

        if (consumed != text.size())
        {
            throw std::invalid_argument(option_name + " contains invalid characters");
        }

        return static_cast<std::int64_t>(value);
    }

    std::int64_t parse_positive_int64(
        const std::string &text,
        const std::string &option_name)
    {
        const std::int64_t value =
            parse_int64(text, option_name);

        if (value <= 0)
        {
            throw std::invalid_argument(
                option_name +
                " must be positive");
        }

        return value;
    }

    std::int32_t parse_non_negative_int32(
        const std::string &text,
        const std::string &option_name)
    {
        const std::int64_t value =
            parse_int64(text, option_name);

        if (value < 0 ||
            value >
                std::numeric_limits<std::int32_t>::max())
        {
            throw std::invalid_argument(
                option_name +
                " is outside the non-negative int32 range");
        }

        return static_cast<std::int32_t>(value);
    }

    std::uint64_t parse_uint64(
        const std::string &text,
        const std::string &option_name)
    {
        if (text.empty() || text.front() == '-')
        {
            throw std::invalid_argument(
                option_name +
                " requires a non-negative integer");
        }

        std::size_t consumed = 0;
        unsigned long long value = 0;

        try
        {
            value = std::stoull(
                text,
                &consumed,
                10);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument(
                option_name +
                " requires a valid unsigned integer");
        }

        if (consumed != text.size())
        {
            throw std::invalid_argument(
                option_name +
                " contains invalid characters");
        }

        return static_cast<std::uint64_t>(value);
    }

    float parse_float(
        const std::string &text,
        const std::string &option_name)
    {
        std::size_t consumed = 0;
        float value = 0.0F;

        try
        {
            value = std::stof(
                text,
                &consumed);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument(
                option_name +
                " requires a valid floating-point value");
        }

        if (consumed != text.size() ||
            !std::isfinite(value))
        {
            throw std::invalid_argument(
                option_name +
                " requires a finite floating-point value");
        }

        return value;
    }

    AppOptions parse_arguments(int argc, char **argv)
    {
        AppOptions options;

        for (int index = 1; index < argc; ++index)
        {
            const std::string argument{argv[index]};

            if (argument == "-h" || argument == "--help")
            {
                options.show_help = true;
                continue;
            }

            if (argument == "--model")
            {
                if (!options.model_directory.empty())
                {
                    throw std::invalid_argument("model directory was provided more than once");
                }

                options.model_directory = require_option_value(argc, argv, index, argument);

                continue;
            }

            if (argument == "--gpu")
            {
                options.gpu_index = parse_non_negative_int32(require_option_value(argc, argv, index, argument), argument);

                continue;
            }

            if (argument == "--max-context")
            {
                options.max_context = parse_positive_int64(require_option_value(argc, argv, index, argument), argument);

                options.max_context_was_set = true;

                continue;
            }

            if (argument == "--max-new-tokens")
            {
                options.max_new_tokens =
                    parse_positive_int64(
                        require_option_value(
                            argc,
                            argv,
                            index,
                            argument),
                        argument);

                continue;
            }

            if (argument == "--greedy")
            {
                options.greedy = true;
                continue;
            }

            if (argument == "--sample")
            {
                options.greedy = false;
                continue;
            }

            if (argument == "--temperature")
            {
                options.temperature =
                    parse_float(
                        require_option_value(
                            argc,
                            argv,
                            index,
                            argument),
                        argument);

                continue;
            }

            if (argument == "--top-k")
            {
                options.top_k =
                    parse_non_negative_int32(
                        require_option_value(
                            argc,
                            argv,
                            index,
                            argument),
                        argument);

                continue;
            }

            if (argument == "--top-p")
            {
                options.top_p =
                    parse_float(
                        require_option_value(
                            argc,
                            argv,
                            index,
                            argument),
                        argument);

                continue;
            }

            if (argument == "--seed")
            {
                options.seed =
                    parse_uint64(
                        require_option_value(
                            argc,
                            argv,
                            index,
                            argument),
                        argument);

                continue;
            }

            if (argument == "--thinking")
            {
                options.enable_thinking = true;
                continue;
            }

            if (argument == "--no-thinking")
            {
                options.enable_thinking = false;
                continue;
            }

            if (argument == "--show-thinking")
            {
                options.show_thinking = true;
                continue;
            }

            if (argument == "--hide-thinking")
            {
                options.show_thinking = false;
                continue;
            }

            if (argument == "--system")
            {
                options.system_prompt =
                    require_option_value(
                        argc,
                        argv,
                        index,
                        argument);

                continue;
            }

            if (argument == "--stop-token")
            {
                options.additional_stop_token_ids.push_back(
                    parse_non_negative_int32(
                        require_option_value(
                            argc,
                            argv,
                            index,
                            argument),
                        argument));

                continue;
            }

            if (!argument.empty() &&
                argument.front() == '-')
            {
                throw std::invalid_argument(
                    "unknown option: " + argument);
            }

            // 第一个非 option 参数作为模型目录。
            if (options.model_directory.empty())
            {
                options.model_directory = argument;
                continue;
            }

            throw std::invalid_argument(
                "unexpected positional argument: " +
                argument);
        }

        if (!options.show_help &&
            options.model_directory.empty())
        {
            throw std::invalid_argument(
                "Qwen3 model directory is required");
        }

        if (options.temperature <= 0.0F)
        {
            throw std::invalid_argument(
                "--temperature must be greater than zero");
        }

        if (options.top_p <= 0.0F ||
            options.top_p > 1.0F)
        {
            throw std::invalid_argument(
                "--top-p must be in the interval (0, 1]");
        }

        return options;
    }

    qwen3::Tensor make_token_ids(const std::vector<std::int32_t> &values, const qwen3::Device &device)
    {
        if (values.empty())
        {
            throw std::invalid_argument("prompt token IDs cannot be empty");
        }

        const qwen3::Shape shape{
            1, static_cast<std::int64_t>(values.size())};

        qwen3::Tensor cpu = qwen3::Tensor::empty(shape, qwen3::DType::Int32, qwen3::Device::cpu());

        auto *destination = static_cast<std::int32_t *>(cpu.data());

        for (std::size_t index = 0; index < values.size(); ++index)
        {
            destination[index] = values[index];
        }

        return cpu.copy_to(device);
    }

    void append_unique_token(std::vector<std::int32_t> &token_ids, std::int32_t token_id)
    {
        if (std::find(token_ids.begin(), token_ids.end(), token_id) == token_ids.end())
        {
            token_ids.push_back(token_id);
        }
    }

    bool is_stop_token(std::int32_t token_id, const std::vector<std::int32_t> &stop_token_ids)
    {
        return std::find(stop_token_ids.begin(), stop_token_ids.end(), token_id) != stop_token_ids.end();
    }

    void reset_history(std::vector<qwen3::chat::Message> &history, const std::string &system_prompt)
    {
        history.clear();

        if (!system_prompt.empty())
        {
            history.push_back(qwen3::chat::Message{qwen3::chat::Role::System, system_prompt});
        }
    }

    void print_interactive_help()
    {
        std::cout << "\nCommands:\n"
                  << "   /exit       Exit the program\n"
                  << "   /clear      Clear conversation history\n"
                  << "   /think on   Enable thinking mode\n"
                  << "   /think off  Disable thinking mode\n"
                  << "   /help       Show this help\n\n";
    }

    void configure_terminal_input()
    {
        std::setlocale(LC_ALL, "");

#ifdef IUTF8
        if (::isatty(STDIN_FILENO))
        {
            termios state{};

            if (::tcgetattr(STDIN_FILENO, &state) == 0)
            {
                state.c_iflag |= IUTF8;
                ::tcsetattr(STDIN_FILENO, TCSANOW, &state);
            }
        }
#endif
    }

    struct ParsedAssistantAnswer final
    {
        std::string final_answer;

        bool had_thinking = false;

        // false 表示出现了 <think>
        // 但没有生成 </think>
        bool thinking_complete = true;
    };

    ParsedAssistantAnswer parse_assistant_answer(std::string_view raw_answer)
    {
        constexpr std::string_view opening_tag{
            "<think>"};

        constexpr std::string_view closing_tag{
            "</think>"};

        const std::size_t closing_position =
            raw_answer.rfind(closing_tag);

        if (closing_position !=
            std::string_view::npos)
        {
            std::string_view final_answer =
                raw_answer.substr(
                    closing_position +
                    closing_tag.size());

            // 与 Qwen3 官方模板一致，
            // 只清理标签后面的换行
            while (!final_answer.empty() &&
                   (final_answer.front() == '\n' ||
                    final_answer.front() == '\r'))
            {
                final_answer.remove_prefix(1);
            }

            return ParsedAssistantAnswer{
                std::string{final_answer},
                true,
                true};
        }

        // 出现开始标签，却没有结束标签：
        // 模型只生成了思考，没有生成最终回答
        if (raw_answer.find(opening_tag) !=
            std::string_view::npos)
        {
            return ParsedAssistantAnswer{
                {},
                true,
                false};
        }

        // no-thinking 模式或普通回答
        return ParsedAssistantAnswer{
            std::string{raw_answer},
            false,
            true};
    }
} // namespace

int main(int argc, char **argv)
{
    // 打开UTF-8 退格支持
    configure_terminal_input();
    try
    {
        // ======================================================================================================
        // 1. 解析命令行参数
        // ======================================================================================================
        const AppOptions options = parse_arguments(argc, argv);

        if (options.show_help)
        {
            print_usage(argv[0]);
            return 0;
        }

        // =======================================================================================================
        // 2. 检查模型目录并确定 CUDA 设备
        // =======================================================================================================

        if (!std::filesystem::exists(options.model_directory) || !std::filesystem::is_directory(options.model_directory))
        {
            throw std::invalid_argument("model directory does not exist: " + options.model_directory.string());
        }

        // --gpu 使用逻辑 CUDA 设备编号
        //
        // 如果设置：
        // CUDA_VISIBLE_DEVICE=？
        // 那么物理 GPU ？ 会映射为逻辑GPU 0
        // 此时应该使用 --gpu 0
        const qwen3::Device device = qwen3::Device::cuda(options.gpu_index);

        // =======================================================================================================
        // 3. 加载模型配置并确定实际上下文长度
        // =======================================================================================================

        const qwen3::ModelConfig config = qwen3::load_model_config(options.model_directory / "config.json");

        std::int64_t max_sequence_length = 0;

        if (options.max_context_was_set)
        {
            if (options.max_context > config.max_position_embeddings)
            {
                throw std::invalid_argument("--max-context exceeds the model's max_position_embeddings");
            }

            max_sequence_length = options.max_context;
        }
        else
        {
            // 没有显示指定是，使用应用默认上限
            //
            // 这可以避免直接按照模型的 40960 上下文
            // 分配过大的 KV Cache
            max_sequence_length = std::min<std::int64_t>(kDefaultMaxContext, config.max_position_embeddings);
        }

        // ========================================================================================================
        // 4. 验证采样参数与额外停止 Token
        // ========================================================================================================

        if (options.top_k > config.vocab_size)
        {
            throw std::invalid_argument("--top-k cannot exceed model vocab_size");
        }

        for (const std::int32_t token_id : options.additional_stop_token_ids)
        {
            if (token_id >= config.vocab_size)
            {
                throw std::invalid_argument("--stop-token is outside model vocabulary");
            }
        }

        // =======================================================================================================
        // 5. 加载 Tokenizer
        // =======================================================================================================

        qwen3::Tokenizer tokenizer = qwen3::Tokenizer::load(options.model_directory / "tokenizer.json");

        const std::int64_t tokenizer_vocab_size = static_cast<std::int64_t>(tokenizer.max_token_id()) + 1;

        if (tokenizer.token_count() != tokenizer_vocab_size)
        {
            throw std::runtime_error("sparse tokenizer token IDs are not supported");
        }

        if (tokenizer_vocab_size > config.vocab_size)
        {
            throw std::runtime_error("tokenzier vocabulary exceeds model vocabulary");
        }

        // =======================================================================================================
        // 6. 打开 Safetensors Checkpoint 并加载权重
        // =======================================================================================================

        qwen3::SafetensorsCheckpoint checkpoint{options.model_directory};

        std::cout
            << "Loading Qwen3 weights...\n"
            << "  Model: "
            << options.model_directory.string()
            << '\n'
            << "  Logical GPU: "
            << options.gpu_index
            << '\n'
            << "  Context: "
            << max_sequence_length
            << '\n'
            << "  Max new tokens: "
            << options.max_new_tokens
            << '\n'
            << "  Sampling mode: "
            << (options.greedy
                    ? "greedy"
                    : "sample")
            << '\n';

        if (!options.greedy)
        {
            std::cout
                << "  Temperature: "
                << options.temperature
                << '\n'
                << "  Top-k: "
                << options.top_k
                << '\n'
                << "  Top-p: "
                << options.top_p
                << '\n'
                << "  Seed: "
                << options.seed
                << '\n';
        }

        std::cout
            << "  Thinking: "
            << (options.enable_thinking
                    ? "on"
                    : "off")
            << "\n\n";

        // checkpoint 必须覆盖整个程序生命周期
        // 如果权重 Tensor 引用了 mmap 中的数据
        // 提前销毁 checkpoint 会产生悬空引用
        qwen3::Qwen3Weights weights = qwen3::Qwen3Weights::load(checkpoint, config, device);

        // ==================================================================================================
        // 7. 创建并预计算 RoPE Cache
        // ==================================================================================================
        //
        // RoPE Cache 只与最大序列长度，head_dim 和 rope_theta 有关，创建一次后可以在所有对话轮次中使用

        qwen3::Tensor rope_cache = qwen3::ops::build_rope_cache(max_sequence_length, config.head_dim, config.rope_theta, device);

        // =================================================================================================
        // 8. 创建长期复用的 cuBLAS Handle
        // =================================================================================================

        qwen3::cuda::CublasHandle cublas_handle{device};

        // =================================================================================================
        // 9. 根据命令行参数创建 Sampler
        // =================================================================================================

        qwen3::generation::SamplingConfig sampling_config;

        sampling_config.mode = options.greedy ? qwen3::generation::SamplingMode::Greedy : qwen3::generation::SamplingMode::Sample;

        sampling_config.temperature = options.temperature;
        sampling_config.top_k = options.top_k;
        sampling_config.top_p = options.top_p;
        sampling_config.seed = options.seed;
        sampling_config.valid_vocab_size = tokenizer_vocab_size;

        qwen3::generation::Sampler sampler{sampling_config};

        // ================================================================================================
        // 10. 组合默认停止 Token 和用户额外指定的停止 Token
        // ================================================================================================

        std::vector<std::int32_t> stop_token_ids;

        // Qwen3 默认：
        //
        // eos_token_id= 151645 = <|im_end|>
        // bos_token_id = 151643 = <|endoftext|>
        append_unique_token(stop_token_ids, config.eos_token_id);

        append_unique_token(stop_token_ids, config.bos_token_id);

        for (const std::int32_t token_id : options.additional_stop_token_ids)
        {
            append_unique_token(stop_token_ids, token_id);
        }

        // ===================================================================================================
        // 11. 初始化对话历史与 Thinking 状态
        // ===================================================================================================

        std::vector<qwen3::chat::Message> history;

        // 如果命令行指定了 system prompt
        // reset_history() 会将它作为历史中的第一条消息
        reset_history(history, options.system_prompt);

        bool enable_thinking = options.enable_thinking;

        std::cout << "Qwen3 chat is ready.\n";

        print_interactive_help();

        // ===================================================================================================
        // 12. 进入终端多轮对话循环
        // ===================================================================================================

        while (true)
        {
            std::cout << "user>> " << std::flush;

            std::string user_input;

            if (!std::getline(std::cin, user_input))
            {
                std::cout << '\n';
                break;
            }

            if (user_input.empty())
            {
                continue;
            }

            // ================================================================================================
            // 12.1 处理终端控制命令
            // ================================================================================================

            if (user_input == "/exit" || user_input == "/quit")
            {
                std::cout << "Qwen3 chat is break.\n";
                break;
            }

            if (user_input == "/help")
            {
                print_interactive_help();
                continue;
            }

            if (user_input == "/clear")
            {
                // 清楚 user/assistant 历史
                // 但是保留命令行指定的 system prompt
                reset_history(history, options.system_prompt);

                std::cout << "Conversion history cleared.\n\n";

                continue;
            }

            if (user_input == "/think on")
            {
                enable_thinking = true;

                std::cout << "Thinking mode enabled.\n\n";

                continue;
            }

            if (user_input == "/think off")
            {
                enable_thinking = false;

                std::cout << "Thinking mode disable.\n\n";

                continue;
            }

            // =================================================================================================
            // 12.2 将本轮用户消息加入对话历史
            // =================================================================================================

            history.push_back(qwen3::chat::Message{qwen3::chat::Role::User, user_input});

            try
            {
                // =============================================================================================
                // 12.3 使用 Chat Template 构造模型 Prompt
                // =============================================================================================

                qwen3::chat::TemplateOptions template_options;

                template_options.add_generation_prompt = true;

                template_options.enable_thinking = enable_thinking;

                const std::string prompt_text = qwen3::chat::apply_chat_template(history, template_options);

                // ==============================================================================================
                // 12.4 使用 Tokenizer 将 Prompt 编码为 Token IDs
                // ==============================================================================================
                //
                // Char Template 已经包含必要的控制 Token，所以这里不能再额外添加 BOS 或 EOS

                const std::vector<std::int32_t> prompt_token_ids = tokenizer.encode(prompt_text);

                const std::int64_t prompt_length = static_cast<std::int64_t>(prompt_token_ids.size());

                // =============================================================================================
                // 12.5 检查剩余上下文空间
                // =============================================================================================

                if (prompt_length >= max_sequence_length)
                {
                    throw std::runtime_error("conversation has reached the context limit; use /clear");
                }

                const std::int64_t remaining_length = max_sequence_length - prompt_length;

                // 如果剩余空间不足 max_new_tokens
                // 则自动缩短本轮最大生成长度
                const std::int64_t current_max_new_tokens = std::min(options.max_new_tokens, remaining_length);

                if (current_max_new_tokens <= 0)
                {
                    throw std::runtime_error("no context space remains for generation");
                }

                // ================================================================================================
                // 12.6 将 Prompt Token IDs 从 CPU 搬到 GPU
                // ================================================================================================

                qwen3::Tensor prompt = make_token_ids(prompt_token_ids, device);

                // ================================================================================================
                // 12.7 为本轮对话创建空的 KV Cache
                // ================================================================================================
                //
                // 当前实现每轮都会重新编码完整 history，所以必须使用空 KV Cache
                //
                // 如果继续使用上一轮 Cache，同时又输入完整历史
                // 相同的历史 Token 就会被重复加入 Cache

                qwen3::KvCache kv_cache(config.num_hidden_layers, 1, max_sequence_length, config.num_key_value_heads, config.head_dim, qwen3::DType::BFloat16, device);

                // ================================================================================================
                // 12.8 配置并执行自回归生成
                // ================================================================================================

                qwen3::generation::GenerationOptions generation_options;

                generation_options.stop_token_ids = stop_token_ids;

                generation_options.max_new_tokens = current_max_new_tokens;

                qwen3::generation::GenerationResult result = qwen3::generation::generate_token_ids(prompt, weights, config, rope_cache, kv_cache,
                                                                                                   cublas_handle, sampler, generation_options);

                // Generator 返回的仅是新生成的 Token，不包含输入 Prompt
                std::vector<std::int32_t> generated_token_ids = std::move(result.token_ids);

                // ==================================================================================================
                // 12.9 移除生成结果末尾的停止符号
                // ==================================================================================================
                //
                // Generator 当前会把触发停止的 Token 本身放进GenerationResult
                //
                // 这里仅在最后一个 ID 确实属于停止集合时删除

                if (!generated_token_ids.empty() && is_stop_token(generated_token_ids.back(), stop_token_ids))
                {
                    generated_token_ids.pop_back();
                }

                // =====================================================================================================
                // 12.10 将生成 Token 整体解码为 UTF-8 文本
                // =====================================================================================================
                //
                // 当前不逐 Token 解码，因为单个 Token 可能只包含一个 UTF-8 字符的部分字节
                //
                // 统一解码可以避免中文字符被拆开产生乱码

                const std::string row_answer = tokenizer.decode(generated_token_ids, false);

                const ParsedAssistantAnswer parsed_answer = parse_assistant_answer(row_answer);

                // 模型停在了未闭合的思考过程里
                if (!parsed_answer.thinking_complete)
                {
                    std::cout << "\nassistant>>\n";

                    if (options.show_thinking) {
                        std::cout << row_answer << '\n';
                    }else {
                        std::cout << "[Thinking content hidden.]\n";
                    }

                    std::cout << "[The model stopped before producing a complete final answer. This turn was not added to history.]\n\n";

                    // 因为 history 最后一项是刚加入的 user
                    // 因为没有完整 assistant 回答，所以将其回滚
                    if (!history.empty() && history.back().role == qwen3::chat::Role::User) {
                        history.pop_back();
                    }

                    continue;
                }

                std::cout << "\nassistant>>\n";

                if (options.show_thinking) {
                    // 显示原始结果
                    // <think>...</think> + 最终回答
                    std::cout << row_answer;
                } else {
                    // 只显示最终回答
                    std::cout << parsed_answer.final_answer;
                }

                std::cout << "\n\n";
            
                // =====================================================================================================
                // 12.11 将 Assistant 回答保存进对话历史
                // =====================================================================================================

                history.push_back(qwen3::chat::Message{qwen3::chat::Role::Assistant, std::move(parsed_answer.final_answer)});

                // ====================================================================================================
                // 12.12 报告达到最大生成长度
                // ====================================================================================================

                if (result.stop_reason == qwen3::generation::StopReason::MaxNewTokens)
                {
                    std::cout << "[Reached max-new-tokens.]\n\n";
                }
            }
            catch (const std::exception &error)
            {
                // ===================================================================================================
                // 12.13 本轮失败时回滚对话历史
                // ===================================================================================================
                //
                // 如果本轮生成失败，需要删除刚加入的 User 消息
                // 否则下一轮历史会出现连续的两个 User
                // 破坏 Chat Template 中的角色交替约束

                if (!history.empty() && history.back().role == qwen3::chat::Role::User)
                {
                    history.pop_back();
                }

                std::cerr << "Generation failed: " << error.what() << "\n\n";
            }
        }

        // ===========================================================================================================
        // 13. 正常结束程序
        // ===========================================================================================================

        std::cout << "Bye.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        // ===========================================================================================================
        // 14. 处理程序初始化阶段发生的错误
        // ===========================================================================================================

        std::cerr << "Failed to start qwen_chat " << error.what() << '\n';

        std::cerr << "Use --help to see available options.\n";

        return 1;
    }
}