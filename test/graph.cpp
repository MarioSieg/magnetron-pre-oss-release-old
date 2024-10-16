// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <gtest/gtest.h>
#include <msml.h>

template <typename... Args>
[[nodiscard]] static auto msml_tensor_emit_op_va(msml_op_t op, Args&&... args) -> msml_tensor_t* {
    std::array<msml_tensor_t*, sizeof...(Args)> tensors {args...};
    return msml_tensor_emit_op(op, tensors.data(), tensors.size());
}

TEST(graph, compile_simple) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);


    // (A + B) @ C.relu()
    // Total nodes: 6
    // Total leaves: 3

    msml_tensor_t* A = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_fill_random(A, 0.0f, 1.0f);

    msml_tensor_t* B = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_fill_random(A, 0.0f, 1.0f);

    msml_tensor_t* A_P_B = msml_tensor_emit_op_va(MSML_OP_ADD, A, B);

    msml_tensor_t* C = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_fill_random(C, 0.0f, 1.0f);

    msml_tensor_t* C_relu = msml_tensor_emit_op_va(MSML_OP_RELU, C);

    msml_tensor_t* result = msml_tensor_emit_op_va(MSML_OP_MATMUL, A_P_B, C_relu);

    msml_compute_graph_t* gra = msml_compute_graph_compile(ctx, result, MSML_GRAPH_EVAL_ORDER_FORWARD, nullptr);
    ASSERT_EQ(msml_compute_graph_get_num_total_nodes(gra), 6);
    ASSERT_EQ(msml_compute_graph_get_num_internal_nodes(gra), 3);
    ASSERT_EQ(msml_compute_graph_get_num_leaf_nodes(gra), 3);
    ASSERT_GT(msml_compute_graph_get_memory_usage(gra), sizeof(void*) * 6);

    const msml_tensor_t** leaves = msml_compute_graph_get_leaf_nodes(gra, nullptr);
    ASSERT_EQ(leaves[0], A);
    ASSERT_EQ(leaves[1], B);
    ASSERT_EQ(leaves[2], C);

    size_t n_i = 0;
    const msml_tensor_t** nodes = msml_compute_graph_get_internal_nodes(gra, &n_i);
    ASSERT_EQ(nodes[0], A_P_B);
    ASSERT_EQ(nodes[1], C_relu);
    ASSERT_EQ(nodes[2], result);
    ASSERT_EQ(nodes[n_i-1], result); // Last node must be evaluation root and result

    msml_ctx_destroy(ctx);
}
