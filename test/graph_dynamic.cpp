// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(graph_dynamic, simple) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    wl_ctx_set_exec_mode(ctx, WL_EXEC_MODE_EAGER);

    // ((W * X) + B).relu()

    auto* W = wl_tensor_create_2d(ctx, WL_DTYPE_F32, 2, 2);
    wl_tensor_fill(W, 0.6f);
    wl_tensor_set_name(W, "W");

    auto* X = wl_tensor_create_2d(ctx, WL_DTYPE_F32, 2, 2);
    wl_tensor_fill(X, 2.11f);
    wl_tensor_set_name(X, "X");

    auto* WX = wl_tensor_emit_op_va(ctx, WL_OP_MUL, W, X);
    auto* buf = wl_tensor_data_as_f32(WX);
    for (std::int64_t i=0; i < wl_tensor_num_elements(WX); ++i) { // op must already be executed
        ASSERT_EQ(buf[i], 0.6f*2.11f);
    }

    auto* B = wl_tensor_create_2d(ctx, WL_DTYPE_F32, 2, 2);
    wl_tensor_fill(B, 0.1f);
    wl_tensor_set_name(B, "B");

    auto* WXB = wl_tensor_emit_op_va(ctx, WL_OP_ADD, WX, B);
    buf = wl_tensor_data_as_f32(WXB);
    for (std::int64_t i=0; i < wl_tensor_num_elements(WXB); ++i) { // op must already be executed
        ASSERT_EQ(buf[i], 0.6f*2.11f + 0.1f);
    }

    wl_tensor_decref(WXB);
    wl_tensor_decref(B);
    wl_tensor_decref(WX);
    wl_tensor_decref(W);
    wl_tensor_decref(X);

    wl_ctx_destroy(ctx);
}