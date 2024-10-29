// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <cstddef>
#include <cstdint>

#include <msml.h>

extern "C" [[nodiscard]] auto msml__sto_read_buffered(msml_ctx_t* ctx, const std::uint8_t* buf, std::size_t size, std::size_t* out_n_tensors) -> msml_tensor_t**; // Imported from msml.c

extern "C" auto LLVMFuzzerTestOneInput(const std::int8_t* data, std::size_t dize) -> int {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    std::size_t n_tensors = 0;
    [[maybe_unused]]
    msml_tensor_t** volatile tensors = msml__sto_read_buffered(ctx, reinterpret_cast<const std::uint8_t*>(data), dize, &n_tensors);
    msml_ctx_destroy(ctx);
    return 0;
}