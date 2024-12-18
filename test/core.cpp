// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

#include <numbers>
#include <filesystem>

TEST(core, op_param_int) {
    mag_op_param_t p = mag_op_param_int(1234);
    ASSERT_TRUE(mag_op_param_is_int(p));
    ASSERT_EQ(mag_op_param_unpack_int(p), 1234);
}

TEST(core, op_param_float) {
    mag_op_param_t p = mag_op_param_float(std::numbers::pi_v<float>);
    ASSERT_TRUE(mag_op_param_is_float(p));
    ASSERT_FLOAT_EQ(mag_op_param_unpack_float(p), std::numbers::pi_v<float>);
}

TEST(core, profiler_small_dims) {
    mag_ctx_t* ctx = mag_ctx_create(nullptr);
    mag_ctx_profile_start_recording(ctx);
    for (int i=0; i < 1000000; ++i) {
        mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, 3, 3);
        mag_tensor_t* B =  mag_tensor_emit_op_va(ctx, MAG_OP_SIN, A);
        mag_tensor_t* C =  mag_tensor_emit_op_va(ctx, MAG_OP_COS, B);
        [[maybe_unused]]
        mag_tensor_t* D =  mag_tensor_emit_op_va(ctx, MAG_OP_TANH, C);
        mag_tensor_decref(A);
        mag_tensor_decref(B);
        mag_tensor_decref(C);
        mag_tensor_decref(D);
    }
    mag_ctx_profile_stop_recording(ctx, "perf.csv");
    ASSERT_TRUE(std::filesystem::exists("perf.csv"));
    std::filesystem::remove("perf.csv");
    mag_ctx_destroy(ctx);
}

TEST(core, profiler_big_dims) {
    mag_ctx_t* ctx = mag_ctx_create(nullptr);
    mag_ctx_profile_start_recording(ctx);
    mag_tensor_t* A = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, 32, 32, 4, 4, 4, 4);
    mag_tensor_t* B =  mag_tensor_emit_op_va(ctx, MAG_OP_SIN, A);
    mag_tensor_t* C =  mag_tensor_emit_op_va(ctx, MAG_OP_COS, B);
    [[maybe_unused]]
    mag_tensor_t* D =  mag_tensor_emit_op_va(ctx, MAG_OP_TANH, C);
    mag_ctx_profile_stop_recording(ctx, NULL);
    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(C);
    mag_tensor_decref(D);
    mag_ctx_destroy(ctx);
}

#if 0
TEST(core, crc32) {
    ASSERT_EQ(mag__crc32c("Hello, World!", std::strlen("Hello, World!")), 1297420392);
    uint8_t y = 0x3f;
    ASSERT_EQ(mag__crc32c(& y, sizeof(y)), 1015883460);
    ASSERT_EQ(mag__crc32c(nullptr, 0), 0);
    ASSERT_EQ(mag__crc32c("AB", std::strlen("AB")), 3180610794);
    ASSERT_EQ(mag__crc32c(
            "Ich liebe Berliner Kebap, der ist einfach ultra schmackofatz, gerade um 4 Uhr Morgens nach einer langen Clubnacht.",
            std::strlen(
                    "Ich liebe Berliner Kebap, der ist einfach ultra schmackofatz, gerade um 4 Uhr Morgens nach einer langen Clubnacht.")), 60440201);
    std::vector<std::uint8_t> huge {};
    huge.resize(0xffff);
    for (std::size_t i = 0; i < huge.size(); ++i) {
        huge[i] = i % 0xff;
    }
    ASSERT_EQ(mag__crc32c(huge.data(), huge.size()), 2008503331);
}
#endif