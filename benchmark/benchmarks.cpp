// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: linux_prepare_perf.sh to setup the system for performance measurements.

#include <functional>

#include <wavelet.h>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

static auto run_bench(ankerl::nanobench::Bench& b, const char* name, std::function<auto(wl_ctx_t* ctx) -> wl_tensor_t*>&& callback) -> void;

auto main() -> int {
    ankerl::nanobench::Bench matmul_bench {};


    ankerl::nanobench::Bench softmax_bench {};
    matmul_bench.title("Tensor Add")
        .unit("add")
        .warmup(100)
        .relative(true);
    matmul_bench.performanceCounters(true);

    run_bench(matmul_bench, "Tensor Add", [](wl_ctx_t* ctx) -> wl_tensor_t* {
        constexpr std::int64_t N = 128;

        wl_tensor_t* A = wl_tensor_create_2d(ctx, WL_DTYPE_F32, N, N);
        wl_tensor_fill(A, 1.0f);

        wl_tensor_t* B = wl_tensor_create_2d(ctx, WL_DTYPE_F32, N/2, N/2);
        wl_tensor_fill(A, 1.0f);

        wl_tensor_t* inputs[2] = {A, B};
        wl_tensor_t* C = wl_tensor_operator(ctx, WL_OP_ADD, inputs, 2, nullptr);
        return C;
    });
}

static auto run_bench(ankerl::nanobench::Bench& b, const char* name, std::function<auto(wl_ctx_t* ctx) -> wl_tensor_t*>&& callback) -> void {
    wl_ctx_t* ctx = wl_ctx_create2(4ull << 30);
    wl_ctx_set_exec_mode(ctx, WL_EXEC_MODE_DEFERRED);
    wl_compute_graph_t* gra = wl_compute_graph_compile(ctx, std::invoke(callback, ctx), WL_GRAPH_EVAL_ORDER_FORWARD, nullptr);
    b.run(name, [gra]() -> void {
        wl_compute_graph_execute(gra);
    });
    ankerl::nanobench::doNotOptimizeAway(gra);
    wl_ctx_destroy(ctx);
}
