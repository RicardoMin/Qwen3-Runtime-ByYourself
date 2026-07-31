#pragma once

#include <cstdint>
#include <filesystem>

namespace qwen3 {

    struct ModelConfig {
        // 词表和 Transformer 主体
        std::int32_t vocab_size = 0;
        std::int32_t hidden_size = 0;
        std::int32_t intermediate_size = 0;
        std::int32_t num_hidden_layers = 0;

        // 注意力结构
        std::int32_t num_attention_heads = 0;
        std::int32_t num_key_value_heads = 0;
        std::int32_t head_dim = 0;

        // 上下文和位置编码
        std::int32_t max_position_embeddings = 0;
        float rope_theta = 0.0F;

        // RMSNorm
        float rms_norm_eps = 0.0F;

        // 特殊 token
        std::int32_t bos_token_id = -1;
        std::int32_t eos_token_id = -1;

        // 模型结构开关
        bool attention_bias = false;
        bool tie_word_embeddings = false;
        bool use_sliding_window = false;

        // Q 投影后的总维度
        [[nodiscard]]
        std::int64_t q_projection_size() const noexcept;

        // K 或 V 投影后的总维度
        [[nodiscard]]
        std::int64_t kv_projection_size() const noexcept;

        // 一个 KV head 对应多少个 Q head
        [[nodiscard]]
        std::int32_t query_heads_per_kv_head() const noexcept;
        
        // 检查配置是否合法
        void validate() const;
    };

    // 当前只用于锁定和测试 Qwen3-0.6B 官方参数
    // 后面会由 config.json 加载器替代硬编码
    [[nodiscard]]
    ModelConfig make_qwen3_0_6b_config();

    [[nodiscard]]
    ModelConfig load_model_config(const std::filesystem::path &config_path);

}   // namespace qwen3