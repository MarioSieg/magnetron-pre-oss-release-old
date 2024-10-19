#include <array>

#include <msml.h>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

auto main() -> int {
    msml_ctx_t* ctx = msml_ctx_create2(4ull << 30);
    msml_ctx_set_exec_mode(ctx, MSML_EXEC_MODE_DEFERRED);
    msml_tensor_t* A = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 16000, 16000);
    msml_tensor_fill(A, 1.0f);
    msml_tensor_t* B = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 16000, 16000);
    msml_tensor_fill(B, 2.0f);
    std::array<msml_tensor_t*, 2> in {A, B};
    msml_tensor_t* C = msml_tensor_operator(ctx, MSML_OP_ADD, in.data(), in.size(), nullptr);
    msml_compute_graph_t* gra = msml_compute_graph_compile(ctx, C, MSML_GRAPH_EVAL_ORDER_FORWARD, nullptr);

    ankerl::nanobench::Bench().run("Matmul 16000 X 16000", [&] {
        msml_compute_graph_execute(gra);
    });

    ankerl::nanobench::doNotOptimizeAway(gra);

    msml_ctx_destroy(ctx);
}
