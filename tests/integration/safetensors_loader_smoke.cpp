#include "qwen3/core/device.h"
#include "qwen3/core/tensor.h"
#include "qwen3/cuda/cuda_check.h"
#include "qwen3/io/safetensors_loader.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{

    void require(bool condition, const char *message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc != 2)
        {
            std::cerr
                << "Usage: safetensors_loader_smoke "
                << "<model.safetensors>\n";

            return 1;
        }

        qwen3::SafetensorsLoader loader{argv[1]};

        // 选择很小的权重测试，避免加载几百 MB 的 embedding。
        constexpr const char *weight_name =
            "model.norm.weight";

        const auto &info =
            loader.metadata().tensor(weight_name);

        require(
            info.shape.numel() == 1024,
            "model.norm.weight shape is incorrect");

        require(
            info.size_bytes() == 1024 * 2,
            "model.norm.weight byte size is incorrect");

        // 加载到 CPU。
        qwen3::Tensor cpu_tensor =
            loader.load(
                weight_name,
                qwen3::Device::cpu());

        require(
            cpu_tensor.nbytes() == info.size_bytes(),
            "CPU tensor byte size is incorrect");

        // CUDA_VISIBLE_DEVICES=6 后，程序内部看到的是 cuda:0。
        qwen3::Tensor cuda_tensor =
            loader.load(
                weight_name,
                qwen3::Device::cuda(0));

        require(
            cuda_tensor.nbytes() == info.size_bytes(),
            "CUDA tensor byte size is incorrect");

        // 将 CUDA 权重复制回来，与 CPU 权重逐字节比较。
        std::vector<std::byte> cuda_result(
            cuda_tensor.nbytes());

        QWEN3_CUDA_CHECK(
            cudaMemcpy(
                cuda_result.data(),
                cuda_tensor.data(),
                cuda_tensor.nbytes(),
                cudaMemcpyDeviceToHost));

        QWEN3_CUDA_CHECK(
            cudaDeviceSynchronize());

        require(
            std::memcmp(
                cpu_tensor.data(),
                cuda_result.data(),
                cpu_tensor.nbytes()) == 0,
            "CPU and CUDA weight bytes are different");

        std::cout
            << "Safetensors loader smoke passed.\n"
            << "Weight: " << weight_name << '\n'
            << "Elements: " << info.shape.numel() << '\n'
            << "Bytes: " << info.size_bytes() << '\n';

        const std::vector<std::string> weight_names{
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight"};

        std::vector<qwen3::Tensor> cpu_weights =
            loader.load_many(
                weight_names,
                qwen3::Device::cpu());

        std::vector<qwen3::Tensor> cuda_weights =
            loader.load_many(
                weight_names,
                qwen3::Device::cuda(0));

        require(
            cpu_weights.size() == weight_names.size(),
            "CPU batch size is incorrect");

        require(
            cuda_weights.size() == weight_names.size(),
            "CUDA batch size is incorrect");

        for (std::size_t index = 0;
             index < weight_names.size();
             ++index)
        {
            require(
                cpu_weights[index].nbytes() ==
                    cuda_weights[index].nbytes(),
                "CPU and CUDA tensor sizes differ");

            std::vector<std::byte> copied_back(
                cuda_weights[index].nbytes());

            QWEN3_CUDA_CHECK(
                cudaMemcpy(
                    copied_back.data(),
                    cuda_weights[index].data(),
                    copied_back.size(),
                    cudaMemcpyDeviceToHost));

            require(
                std::memcmp(
                    cpu_weights[index].data(),
                    copied_back.data(),
                    copied_back.size()) == 0,
                "CPU and CUDA tensor bytes differ");
        }

        std::cout << "pass !" << std::endl;

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Safetensors loader smoke failed: "
            << error.what() << '\n';

        return 1;
    }
}