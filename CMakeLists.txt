cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(qwen3_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(ICU REQUIRED COMPONENTS uc i18n)

# 我们真正的运行时库。
# 后面 Tensor、算子、模型和生成模块都会逐步加入这个 target。
add_library(qwen3_runtime STATIC
    src/core/dtype.cpp
    src/core/device.cpp
    src/core/shape.cpp
    src/core/tensor_view.cpp
    src/core/buffer.cpp
    src/core/tensor.cpp
    src/io/mapped_file.cpp
    src/io/safetensors_file.cpp
    src/io/safetensors_metadata.cpp
    src/io/safetensors_loader.cpp
    src/io/safetensors_checkpoint.cpp
    src/ops/embedding.cu
    src/ops/rms_norm.cu
    src/ops/elementwise.cu
    src/ops/linear.cpp
    src/ops/rope.cu
    src/ops/gqa_attention.cu
    src/ops/argmax.cu
    src/cuda/cuda_check.cpp
    src/cuda/cublas_handle.cpp
    src/model/weights.cpp
    src/model/config.cpp
    src/model/kv_cache.cpp
    src/model/self_attention.cpp
    src/model/mlp.cpp
    src/model/decoder_layer.cpp
    src/model/model.cpp
    src/model/lm_head.cpp
    src/generation/sampler.cpp
    src/generation/generator.cpp
    src/tokenizer/tokenizer.cpp
    src/tokenizer/tokenizer_encode.cpp
    src/chat/chat_template.cpp
)

target_include_directories(qwen3_runtime
    SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party
)

target_include_directories(qwen3_runtime
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(qwen3_runtime
    PUBLIC cxx_std_17
)

# 运行库链接 CUDA Runtime
target_link_libraries(qwen3_runtime
    PUBLIC 
        CUDA::cudart
        CUDA::cublas
)

# 运行库链接 ICU
target_link_libraries(qwen3_runtime
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        ICU::uc
        ICU::i18n
)

# CUDA 工具链测试
add_executable(cuda_smoke
    tests/toolchain/cuda_smoke.cu
)

target_link_libraries(cuda_smoke
    PRIVATE CUDA::cudart
)

# ModelConfig 单元测试
add_executable(model_config_test
    tests/unit/model_config_test.cpp
)

target_link_libraries(model_config_test
    PRIVATE qwen3_runtime
)

# COREMetadata 测试类型
add_executable(core_metadata_test
    tests/unit/core_metadata_test.cpp
)

target_link_libraries(core_metadata_test
    PRIVATE qwen3_runtime
)

# 测试 Tensro_view 功能
add_executable(tensor_view_test
    tests/unit/tensor_view_test.cpp
)

target_link_libraries(tensor_view_test
    PRIVATE qwen3_runtime
)

# 添加 buffer 关于cpu，cuda 的测试
add_executable(buffer_cpu_test
    tests/unit/buffer_cpu_test.cpp
)

target_link_libraries(buffer_cpu_test
    PRIVATE qwen3_runtime
)

add_executable(buffer_cuda_test
    tests/unit/buffer_cuda_test.cpp
)

target_link_libraries(buffer_cuda_test
    PRIVATE qwen3_runtime
)

# 添加 tensor 关于 cpu，cuda 的测试
add_executable(tensor_cpu_test
    tests/unit/tensor_cpu_test.cpp
)

target_link_libraries(tensor_cpu_test
    PRIVATE qwen3_runtime
)

add_executable(tensor_cuda_test
    tests/unit/tensor_cuda_test.cpp
)

target_link_libraries(tensor_cuda_test
    PRIVATE qwen3_runtime
)

# 添加 权重文件匹配 测试
add_executable(mapped_file_test
    tests/unit/mapped_file_test.cpp
)

target_link_libraries(mapped_file_test
    PRIVATE qwen3_runtime
)

# 添加 safetensors_file 的测试
add_executable(safetensors_file_test
    tests/unit/safetensors_file_test.cpp
)

target_link_libraries(safetensors_file_test
    PRIVATE qwen3_runtime
)

# 添加 safetensors_metadata 的测试
add_executable(safetensors_metadata_smoke
    tests/integration/safetensors_metadata_smoke.cpp
)

target_link_libraries(safetensors_metadata_smoke
    PRIVATE qwen3_runtime
)

# 添加 safertensors_loader 的测试
add_executable(safetensors_loader_smoke
    tests/integration/safetensors_loader_smoke.cpp
)

target_link_libraries(safetensors_loader_smoke
    PRIVATE
        qwen3_runtime
        CUDA::cudart
)

# 添加 safetensors_checkpoint 的测试
add_executable(
    safetensors_checkpoint_test
    tests/unit/safetensors_checkpoint_test.cpp
)

target_link_libraries(
    safetensors_checkpoint_test
    PRIVATE qwen3_runtime
)

# 添加 weights 的测试
add_executable(qwen3_weights_smoke
    tests/integration/qwen3_weights_smoke.cpp
)

target_link_libraries(qwen3_weights_smoke
    PRIVATE
        qwen3_runtime
        CUDA::cudart
)

# 添加 embedding 的测试
add_executable(embedding_cuda_test
    tests/unit/embedding_cuda_test.cpp
)

target_link_libraries(embedding_cuda_test
    PRIVATE
        qwen3_runtime
        CUDA::cudart
)

# 添加 rms_norm 的测试
add_executable(rms_norm_cuda_test
    tests/unit/rms_norm_cuda_test.cpp
)

target_link_libraries(rms_norm_cuda_test
    PRIVATE
        qwen3_runtime
        CUDA::cudart
)

# 添加 elementwise 的测试
add_executable(elementwise_cuda_test
    tests/unit/elementwise_cuda_test.cpp
)

target_link_libraries(elementwise_cuda_test
    PRIVATE
        qwen3_runtime
        CUDA::cudart
)

# 添加 linear 的测试
add_executable(linear_cuda_test
    tests/unit/linear_cuda_test.cpp
)

target_link_libraries(linear_cuda_test
    PRIVATE
        qwen3_runtime
        CUDA::cudart
        CUDA::cublas
)

# 添加 rope 的测试
add_executable(rope_cuda_test
    tests/unit/rope_cuda_test.cpp
)

target_link_libraries(rope_cuda_test
    PRIVATE qwen3_runtime
)

# 添加 kv_cache 的测试
add_executable(kv_cache_cuda_test
    tests/unit/kv_cache_cuda_test.cpp
)

target_link_libraries(kv_cache_cuda_test
    PRIVATE qwen3_runtime
)

# 添加 gqa_attention 的测试
add_executable(gqa_attention_cuda_test
    tests/unit/gqa_attention_cuda_test.cpp
)

target_link_libraries(gqa_attention_cuda_test
    PRIVATE qwen3_runtime
)

# 添加 self_attention 的测试
add_executable(self_attention_smoke
    tests/integration/self_attention_smoke.cpp
)

target_link_libraries(self_attention_smoke
    PRIVATE qwen3_runtime
)

# 添加 mlp 的测试
add_executable(mlp_real_smoke
    tests/integration/mlp_real_smoke.cu
)

target_link_libraries(mlp_real_smoke
    PRIVATE
        qwen3_runtime
        CUDA::cudart
        CUDA::cublas
)

# 添加 decoder_layer 的测试
add_executable(decoder_layer_real_smoke
    tests/integration/decoder_layer_real_smoke.cu
)

target_link_libraries(decoder_layer_real_smoke
    PRIVATE
        qwen3_runtime
        CUDA::cudart
        CUDA::cublas
)

add_executable(model_real_smoke
    tests/integration/model_real_smoke.cu
)

target_link_libraries(model_real_smoke
    PRIVATE
        qwen3_runtime
        CUDA::cudart
        CUDA::cublas
)

add_executable(lm_head_cuda_test
    tests/unit/lm_head_cuda_test.cpp
)

target_link_libraries(lm_head_cuda_test
    PRIVATE
        qwen3_runtime
        CUDA::cudart
        CUDA::cublas
)

add_executable(argmax_cuda_test
    tests/unit/argmax_cuda_test.cpp
)

target_link_libraries(argmax_cuda_test
    PRIVATE
        qwen3_runtime
        CUDA::cudart
)

add_executable(sampler_cuda_test
    tests/unit/sampler_cuda_test.cpp
)

target_link_libraries(sampler_cuda_test
    PRIVATE
        qwen3_runtime
        CUDA::cudart
)

add_executable(generator_real_smoke
    tests/integration/generator_real_smoke.cpp
)

target_link_libraries(generator_real_smoke
    PRIVATE
        qwen3_runtime
        CUDA::cudart
        CUDA::cublas
)

add_executable(tokenizer_decode_test
    tests/integration/tokenizer_decode_test.cpp
)

target_link_libraries(tokenizer_decode_test
    PRIVATE qwen3_runtime
)

add_executable(tokenizer_encode_test
    tests/integration/tokenizer_encode_test.cpp
)

target_link_libraries(tokenizer_encode_test
    PRIVATE qwen3_runtime
)

add_executable(chat_template_test
    tests/unit/chat_template_test.cpp
)

target_link_libraries(chat_template_test
    PRIVATE qwen3_runtime
)

# 启动 qwen3_chat 程序
add_executable(qwen3_chat
    apps/qwen3_chat.cpp
)

target_link_libraries(qwen3_chat
    PRIVATE qwen3_runtime
)

include(CTest)

if(BUILD_TESTING)
    add_test(NAME cuda_smoke COMMAND cuda_smoke)
    add_test(
        NAME model_config_test
        COMMAND model_config_test
            "${PROJECT_SOURCE_DIR}/tests/fixtures/qwen3_0_6b.json"
            "${PROJECT_SOURCE_DIR}/tests/fixtures/qwen3_8b.json"
    )
    add_test(NAME core_metadata_test COMMAND core_metadata_test)
    add_test(NAME tensor_view_test COMMAND tensor_view_test)

    add_test(
        NAME buffer_cpu_test
        COMMAND buffer_cpu_test
    )

    add_test(
        NAME buffer_cuda_test
        COMMAND buffer_cuda_test
    )

    set_tests_properties(
        buffer_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME tensor_cpu_test
        COMMAND tensor_cpu_test
    )

    add_test(
        NAME tensor_cuda_test
        COMMAND tensor_cuda_test
    )

    set_tests_properties(
        tensor_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME mapped_file_test
        COMMAND mapped_file_test
    )

    add_test(
        NAME safetensors_file_test
        COMMAND safetensors_file_test
    )
    add_test(
        NAME safetensors_checkpoint_test
        COMMAND safetensors_checkpoint_test
    )

    add_test(
        NAME embedding_cuda_test
        COMMAND embedding_cuda_test
    )

    set_tests_properties(
        embedding_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME rms_norm_cuda_test
        COMMAND rms_norm_cuda_test
    )

    set_tests_properties(
        rms_norm_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME elementwise_cuda_test
        COMMAND elementwise_cuda_test
    )

    set_tests_properties(
        elementwise_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME linear_cuda_test
        COMMAND linear_cuda_test
    )

    set_tests_properties(
        linear_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME rope_cuda_test
        COMMAND rope_cuda_test
    )

    set_tests_properties(
        rope_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME kv_cache_cuda_test
        COMMAND kv_cache_cuda_test
    )

    set_tests_properties(
        kv_cache_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME gqa_attention_cuda_test
        COMMAND gqa_attention_cuda_test
    )

    set_tests_properties(
        gqa_attention_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME self_attention_smoke
        COMMAND self_attention_smoke
    )

    set_tests_properties(
        self_attention_smoke
        PROPERTIES
            LABELS "gpu;integration"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME lm_head_cuda_test
        COMMAND lm_head_cuda_test
    )

    set_tests_properties(
        lm_head_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME argmax_cuda_test
        COMMAND argmax_cuda_test
    )

    set_tests_properties(
        argmax_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME sampler_cuda_test
        COMMAND sampler_cuda_test
    )

    set_tests_properties(
        sampler_cuda_test
        PROPERTIES
            LABELS "gpu"
            RUN_SERIAL TRUE
    )

    add_test(
        NAME chat_template_test
        COMMAND chat_template_test
    )

endif()