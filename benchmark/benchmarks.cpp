// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: linux_prepare_perf.sh to setup the system for performance measurements.

#include <array>
#include <functional>

#include <msml.h>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

static auto run_bench(ankerl::nanobench::Bench& b, const char* name, std::function<auto(msml_ctx_t* ctx) -> msml_tensor_t*>&& callback) -> void;

auto main() -> int {
    ankerl::nanobench::Bench matmul_bench {};
    matmul_bench.title("Matmul")
            .unit("matmul")
            .warmup(100)
            .relative(true);
    matmul_bench.performanceCounters(true);

    run_bench(matmul_bench, "Matmul Large", [](msml_ctx_t* ctx) -> msml_tensor_t* {
        constexpr std::int64_t N = 16384;

        msml_tensor_t* A = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, N, N);
        msml_tensor_fill(A, 1.0f);
        msml_tensor_t* B = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, N, N);
        msml_tensor_fill(B, 2.0f);

        std::array<msml_tensor_t*, 2> in {A, B};
        msml_tensor_t* C = msml_tensor_operator(ctx, MSML_OP_ADD, in.data(), in.size(), nullptr);
        return C;
    });

    ankerl::nanobench::Bench softmax_bench {};
    matmul_bench.title("Softmax")
        .unit("softmax")
        .warmup(100)
        .relative(true);
    matmul_bench.performanceCounters(true);

    run_bench(matmul_bench, "Softmax Large", [](msml_ctx_t* ctx) -> msml_tensor_t* {
        constexpr std::int64_t N = 16384;

        msml_tensor_t* A = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, N, N);
        msml_tensor_fill(A, 1.0f);

        msml_tensor_t* C = msml_tensor_operator(ctx, MSML_OP_SOFTMAX, &A, 1, nullptr);
        return C;
    });
}

static auto run_bench(ankerl::nanobench::Bench& b, const char* name, std::function<auto(msml_ctx_t* ctx) -> msml_tensor_t*>&& callback) -> void {
    msml_ctx_t* ctx = msml_ctx_create2(4ull << 30);
    msml_ctx_set_exec_mode(ctx, MSML_EXEC_MODE_DEFERRED);
    msml_compute_graph_t* gra = msml_compute_graph_compile(ctx, std::invoke(callback, ctx), MSML_GRAPH_EVAL_ORDER_FORWARD, nullptr);
    b.run(name, [gra]() -> void {
        msml_compute_graph_execute(gra);
    });
    ankerl::nanobench::doNotOptimizeAway(gra);
    msml_ctx_destroy(ctx);
}
