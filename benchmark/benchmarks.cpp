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
        constexpr std::int64_t N = 1024;

        wl_tensor_t* A = wl_tensor_create_2d(ctx, WL_DTYPE_F32, N, N);
        wl_tensor_fill(A, 1.0f);

        wl_tensor_t* B = wl_tensor_create_2d(ctx, WL_DTYPE_F32, N, N);
        wl_tensor_fill(A, 1.0f);

        wl_tensor_t* inputs[2] = {A, B};
        wl_tensor_t* C = wl_tensor_operator(ctx, WL_OP_ADD, false, inputs, 2, nullptr);

        wl_tensor_destroy(A);
        wl_tensor_destroy(B);
        wl_tensor_destroy(C);
        return C;
    });
}

static auto run_bench(ankerl::nanobench::Bench& b, const char* name, std::function<auto(wl_ctx_t* ctx) -> wl_tensor_t*>&& callback) -> void {
    wl_ctx_t* ctx = wl_ctx_create2(WL_COMPUTE_DEVICE_TYPE_CPU);
    b.run(name, [&]() -> void {
        std::invoke(callback, ctx);
    });
    ankerl::nanobench::doNotOptimizeAway(ctx);
    wl_ctx_destroy(ctx);
}
