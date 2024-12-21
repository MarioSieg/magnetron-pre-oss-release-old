// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: linux_prepare_perf.sh to setup the system for performance measurements.

#include <functional>

#include <magnetron.h>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

static auto run_bench(ankerl::nanobench::Bench& b, const char* name, std::function<auto(mag_ctx_t* ctx) -> mag_tensor_t*>&& callback) -> void;

auto main() -> int {
    ankerl::nanobench::Bench matmul_bench {};


    ankerl::nanobench::Bench softmax_bench {};
    matmul_bench.title("Tensor Add")
        .unit("add")
        .warmup(100)
        .relative(true);
    matmul_bench.performanceCounters(true);

    run_bench(matmul_bench, "Tensor Add", [](mag_ctx_t* ctx) -> mag_tensor_t* {
        constexpr std::int64_t N = 1024;

        mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, N, N);
        mag_tensor_fill(A, 1.0f);

        mag_tensor_t* B = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, N, N);
        mag_tensor_fill(A, 1.0f);

        mag_tensor_t* C = mag_add(A, B);

        mag_tensor_decref(A);
        mag_tensor_decref(B);
        mag_tensor_decref(C);
        return C;
    });
}

static auto run_bench(ankerl::nanobench::Bench& b, const char* name, std::function<auto(mag_ctx_t* ctx) -> mag_tensor_t*>&& callback) -> void {
    mag_ctx_t* ctx = mag_ctx_create2(MAG_COMPUTE_DEVICE_TYPE_CPU);
    b.run(name, [&]() -> void {
        std::invoke(callback, ctx);
    });
    ankerl::nanobench::doNotOptimizeAway(ctx);
    mag_ctx_destroy(ctx);
}
