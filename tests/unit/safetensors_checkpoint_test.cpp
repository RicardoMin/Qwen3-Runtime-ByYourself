#include "qwen3/io/safetensors_checkpoint.h"

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace
{

void require(
    bool condition,
    const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(
    Function&& function,
    const std::string& message)
{
    bool caught = false;

    try
    {
        function();
    }
    catch (const Exception&)
    {
        caught = true;
    }

    require(caught, message);
}

std::uint64_t next_id() noexcept
{
    static std::uint64_t value = 0;
    return value++;
}

class TempDirectory final
{
public:
    TempDirectory()
        : path_{
              std::filesystem::temp_directory_path() /
              ("qwen3-checkpoint-" +
               std::to_string(::getpid()) +
               "-" +
               std::to_string(next_id()))}
    {
        if (!std::filesystem::create_directory(
                path_))
        {
            throw std::runtime_error(
                "cannot create temporary directory");
        }
    }

    ~TempDirectory() noexcept
    {
        std::error_code error;
        std::filesystem::remove_all(
            path_,
            error);
    }

    [[nodiscard]]
    const std::filesystem::path&
    path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

std::string encode_u64_le(
    std::uint64_t value)
{
    std::string bytes(8, '\0');

    for (std::size_t index = 0;
         index < 8;
         ++index)
    {
        bytes[index] =
            static_cast<char>(
                (value >> (index * 8)) &
                0xFFU);
    }

    return bytes;
}

void write_file(
    const std::filesystem::path& path,
    const std::string& contents)
{
    std::ofstream output{
        path,
        std::ios::binary |
            std::ios::trunc};

    if (!output)
    {
        throw std::runtime_error(
            "cannot open temporary file");
    }

    output.write(
        contents.data(),
        static_cast<std::streamsize>(
            contents.size()));

    output.close();

    if (!output)
    {
        throw std::runtime_error(
            "cannot write temporary file");
    }
}

template <typename Type>
void append_value(
    std::string& bytes,
    const Type& value)
{
    bytes.append(
        reinterpret_cast<const char*>(&value),
        sizeof(Type));
}

void write_safetensors(
    const std::filesystem::path& path,
    std::string header,
    const std::string& data)
{
    while (header.size() % 8 != 0)
    {
        header.push_back(' ');
    }

    write_file(
        path,
        encode_u64_le(header.size()) +
            header +
            data);
}

template <typename Type>
Type read_value(
    const qwen3::Tensor& tensor,
    std::size_t index)
{
    require(
        tensor.device().is_cpu(),
        "test tensor is not on CPU");

    require(
        (index + 1) * sizeof(Type) <=
            tensor.nbytes(),
        "test read is outside tensor bounds");

    Type value{};

    std::memcpy(
        &value,
        static_cast<const std::byte*>(
            tensor.data()) +
            index * sizeof(Type),
        sizeof(Type));

    return value;
}

void create_single_checkpoint(
    const std::filesystem::path& directory)
{
    std::string data;

    append_value(data, 1.25F);
    append_value(data, -2.5F);
    append_value(
        data,
        std::int32_t{42});

    write_safetensors(
        directory / "model.safetensors",
        R"({"tensor.a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"tensor.c":{"dtype":"I32","shape":[1],"data_offsets":[8,12]}})",
        data);
}

void create_sharded_checkpoint(
    const std::filesystem::path& directory)
{
    std::string shard1_data;

    append_value(shard1_data, 1.25F);
    append_value(shard1_data, -2.5F);
    append_value(
        shard1_data,
        std::int32_t{42});

    write_safetensors(
        directory /
            "model-00001-of-00002.safetensors",
        R"({"tensor.a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"tensor.c":{"dtype":"I32","shape":[1],"data_offsets":[8,12]}})",
        shard1_data);

    std::string shard2_data;

    append_value(shard2_data, 10.0F);
    append_value(shard2_data, 20.0F);
    append_value(shard2_data, 30.0F);

    write_safetensors(
        directory /
            "model-00002-of-00002.safetensors",
        R"({"tensor.b":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}})",
        shard2_data);

    write_file(
        directory /
            "model.safetensors.index.json",
        R"({"metadata":{"total_size":24},"weight_map":{"tensor.a":"model-00001-of-00002.safetensors","tensor.b":"model-00002-of-00002.safetensors","tensor.c":"model-00001-of-00002.safetensors"}})");
}

} // namespace

int main()
{
    try
    {
        TempDirectory single_directory;

        create_single_checkpoint(
            single_directory.path());

        qwen3::SafetensorsCheckpoint single{
            single_directory.path()};

        require(
            !single.is_sharded(),
            "single checkpoint reported sharded");

        require(
            single.shard_count() == 1,
            "single checkpoint shard count is wrong");

        require(
            single.tensor_count() == 2,
            "single checkpoint tensor count is wrong");

        qwen3::Tensor single_tensor =
            single.load(
                "tensor.a",
                qwen3::Device::cpu());

        require(
            read_value<float>(
                single_tensor,
                0) == 1.25F,
            "single checkpoint value is wrong");

        TempDirectory sharded_directory;

        create_sharded_checkpoint(
            sharded_directory.path());

        qwen3::SafetensorsCheckpoint checkpoint{
            sharded_directory.path()};

        require(
            checkpoint.is_sharded(),
            "sharded checkpoint reported single");

        require(
            checkpoint.shard_count() == 2,
            "shard count is incorrect");

        require(
            checkpoint.tensor_count() == 3,
            "tensor count is incorrect");

        require(
            checkpoint.contains("tensor.a") &&
            checkpoint.contains("tensor.b") &&
            checkpoint.contains("tensor.c"),
            "checkpoint routing table is incomplete");

        require(
            !checkpoint.contains("missing"),
            "missing tensor was reported present");

        const auto& tensor_b_info =
            checkpoint.tensor_info("tensor.b");

        require(
            tensor_b_info.dtype ==
                qwen3::DType::Float32 &&
            tensor_b_info.shape ==
                qwen3::Shape{3},
            "tensor.b metadata is incorrect");

        const std::vector<std::string> names{
            "tensor.b",
            "tensor.a",
            "tensor.c",
            "tensor.b"};

        std::vector<qwen3::Tensor> values =
            checkpoint.load_many(
                names,
                qwen3::Device::cpu());

        require(
            values.size() == names.size(),
            "result tensor count is incorrect");

        require(
            read_value<float>(
                values[0],
                0) == 10.0F &&
            read_value<float>(
                values[0],
                2) == 30.0F,
            "cross-shard result 0 is wrong");

        require(
            read_value<float>(
                values[1],
                0) == 1.25F &&
            read_value<float>(
                values[1],
                1) == -2.5F,
            "cross-shard result 1 is wrong");

        require(
            read_value<std::int32_t>(
                values[2],
                0) == 42,
            "cross-shard result 2 is wrong");

        require(
            read_value<float>(
                values[3],
                0) == 10.0F,
            "duplicate request order was not preserved");

        require_throws<std::out_of_range>(
            [&]
            {
                static_cast<void>(
                    checkpoint.load(
                        "missing",
                        qwen3::Device::cpu()));
            },
            "missing tensor was accepted");

        TempDirectory missing_shard;

        write_file(
            missing_shard.path() /
                "model.safetensors.index.json",
            R"({"metadata":{"total_size":8},"weight_map":{"tensor.a":"missing.safetensors"}})");

        require_throws<std::runtime_error>(
            [&]
            {
                qwen3::SafetensorsCheckpoint invalid{
                    missing_shard.path()};
            },
            "missing shard was accepted");

        TempDirectory path_escape;

        write_file(
            path_escape.path() /
                "model.safetensors.index.json",
            R"({"metadata":{"total_size":8},"weight_map":{"tensor.a":"../outside.safetensors"}})");

        require_throws<std::runtime_error>(
            [&]
            {
                qwen3::SafetensorsCheckpoint invalid{
                    path_escape.path()};
            },
            "path traversal was accepted");

        std::cout
            << "SafetensorsCheckpoint test passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "SafetensorsCheckpoint test failed: "
            << error.what()
            << '\n';

        return EXIT_FAILURE;
    }
}