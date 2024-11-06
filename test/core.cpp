// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(core, op_param_int) {
    wl_op_param_t p = wl_op_param_int(1234);
    ASSERT_TRUE(wl_op_param_is_int(p));
    ASSERT_EQ(wl_op_param_unpack_int(p), 1234);
}

TEST(core, profiler_small_dims) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    wl_ctx_profile_start_recording(ctx);
    for (int i=0; i < 1000000; ++i) {
        wl_tensor_t* A = wl_tensor_create_2d(ctx, WL_DTYPE_F32, 3, 3);
        wl_tensor_t* B =  wl_tensor_emit_op_va(ctx, WL_OP_SIN, A);
        wl_tensor_t* C =  wl_tensor_emit_op_va(ctx, WL_OP_COS, B);
        [[maybe_unused]]
        wl_tensor_t* D =  wl_tensor_emit_op_va(ctx, WL_OP_TANH, C);
    }
    wl_ctx_profile_stop_recording(ctx);
    wl_ctx_profile_generate_report(ctx);
    wl_ctx_destroy(ctx);
}

TEST(core, profiler_big_dims) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    wl_ctx_profile_start_recording(ctx);
    wl_tensor_t* A = wl_tensor_create_6d(ctx, WL_DTYPE_F32, 32, 32, 32, 32, 32, 32);
    wl_tensor_t* B =  wl_tensor_emit_op_va(ctx, WL_OP_SIN, A);
    wl_tensor_t* C =  wl_tensor_emit_op_va(ctx, WL_OP_COS, B);
    [[maybe_unused]]
    wl_tensor_t* D =  wl_tensor_emit_op_va(ctx, WL_OP_TANH, C);
    wl_ctx_profile_stop_recording(ctx);
    wl_ctx_profile_generate_report(ctx);
    wl_ctx_destroy(ctx);
}

#if 0
TEST(core, crc32) {
    ASSERT_EQ(wl__crc32c("Hello, World!", std::strlen("Hello, World!")), 1297420392);
    uint8_t y = 0x3f;
    ASSERT_EQ(wl__crc32c(& y, sizeof(y)), 1015883460);
    ASSERT_EQ(wl__crc32c(nullptr, 0), 0);
    ASSERT_EQ(wl__crc32c("AB", std::strlen("AB")), 3180610794);
    ASSERT_EQ(wl__crc32c(
            "Ich liebe Berliner Kebap, der ist einfach ultra schmackofatz, gerade um 4 Uhr Morgens nach einer langen Clubnacht.",
            std::strlen(
                    "Ich liebe Berliner Kebap, der ist einfach ultra schmackofatz, gerade um 4 Uhr Morgens nach einer langen Clubnacht.")), 60440201);
    std::vector<std::uint8_t> huge {};
    huge.resize(0xffff);
    for (std::size_t i = 0; i < huge.size(); ++i) {
        huge[i] = i % 0xff;
    }
    ASSERT_EQ(wl__crc32c(huge.data(), huge.size()), 2008503331);
}
#endif