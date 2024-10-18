// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(graph_static, simple) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_ctx_set_exec_mode(ctx, MSML_EXEC_MODE_DEFERRED);

    // ((W * X) + B).relu()

    auto* W = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_fill(W, 0.6f);
    msml_tensor_set_name(W, "W");

    auto* X = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_fill(X, 2.11f);
    msml_tensor_set_name(X, "X");

    auto* WX = msml_tensor_emit_op_va(ctx, MSML_OP_MUL, W, X);
    auto* buf = msml_tensor_buf_f32(WX);
    for (std::int64_t i=0; i < msml_tensor_buf_len(WX); ++i) { // op must already be executed
        ASSERT_EQ(buf[i], 0.0f);
    }

    auto* B = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_fill(B, 0.1f);
    msml_tensor_set_name(B, "B");

    auto* WXB = msml_tensor_emit_op_va(ctx, MSML_OP_ADD, WX, B);
    buf = msml_tensor_buf_f32(WXB);
    for (std::int64_t i=0; i < msml_tensor_buf_len(WXB); ++i) { // op must already be executed
        ASSERT_EQ(buf[i], 0.0f);
    }

    msml_compute_graph_dump_to_dot(msml_compute_graph_compile(ctx, WXB, MSML_GRAPH_EVAL_ORDER_FORWARD, nullptr), "graph.dot");
    msml_tensor_evaluate_static_graph(WXB, MSML_GRAPH_EVAL_ORDER_FORWARD);

    for (std::int64_t i=0; i < msml_tensor_buf_len(WXB); ++i) { // op must already be executed
        ASSERT_EQ(buf[i], 0.6f*2.11f + 0.1f);
    }

    msml_ctx_destroy(ctx);
}
