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
    msml_tensor_set_name(A, "A");
    msml_tensor_fill_random(A, 0.0f, 1.0f);

    msml_tensor_t* B = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_set_name(B, "B");
    msml_tensor_fill_random(A, 0.0f, 1.0f);

    msml_tensor_t* A_P_B = msml_tensor_emit_op_va(MSML_OP_ADD, A, B);

    msml_tensor_t* C = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_set_name(C, "C");
    msml_tensor_fill_random(C, 0.0f, 1.0f);

    msml_tensor_t* C_relu = msml_tensor_emit_op_va(MSML_OP_RELU, C);

    msml_tensor_t* result = msml_tensor_emit_op_va(MSML_OP_MATMUL, A_P_B, C_relu);

    msml_compute_graph_t* gra = msml_compute_graph_compile(ctx, result, MSML_GRAPH_EVAL_ORDER_FORWARD, nullptr);
    msml_compute_graph_dump_to_dot(gra, "graph.dot");
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

TEST(graph, compile_complex) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    // Create input tensors
    msml_tensor_t* A = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 64, 64);
    msml_tensor_set_name(A, "A");
    msml_tensor_fill_random(A, -1.0f, 1.0f);

    msml_tensor_t* B = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 64, 64);
    msml_tensor_set_name(B, "B");
    msml_tensor_fill_random(B, -1.0f, 1.0f);

    msml_tensor_t* C = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 64, 64);
    msml_tensor_set_name(C, "C");
    msml_tensor_fill_random(C, -1.0f, 1.0f);

    msml_tensor_t* D = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 64, 64);
    msml_tensor_set_name(D, "D");
    msml_tensor_fill_random(D, -1.0f, 1.0f);

    msml_tensor_t* E = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 64, 64);
    msml_tensor_set_name(E, "E");
    msml_tensor_fill_random(E, -1.0f, 1.0f);

    // Build a complex computation graph
    // H1 = A @ B
    msml_tensor_t* H1 = msml_tensor_emit_op_va(MSML_OP_MATMUL, A, B);

    // H2 = C + D
    msml_tensor_t* H2 = msml_tensor_emit_op_va(MSML_OP_ADD, C, D);

    // H3 = RELU(H1)
    msml_tensor_t* H3 = msml_tensor_emit_op_va(MSML_OP_RELU, H1);

    // H4 = SIGMOID(H2)
    msml_tensor_t* H4 = msml_tensor_emit_op_va(MSML_OP_SIGMOID, H2);

    // H5 = H3 * H4
    msml_tensor_t* H5 = msml_tensor_emit_op_va(MSML_OP_MUL, H3, H4);

    // H6 = SOFTMAX(H5)
    msml_tensor_t* H6 = msml_tensor_emit_op_va(MSML_OP_SOFTMAX, H5);

    // H7 = H6 - E
    msml_tensor_t* H7 = msml_tensor_emit_op_va(MSML_OP_SUB, H6, E);

    // H8 = MATMUL(H7, TRANSPOSE(A))
    msml_tensor_t* A_T = msml_tensor_emit_op_va(MSML_OP_TRANSPOSE, A);
    msml_tensor_t* H8 = msml_tensor_emit_op_va(MSML_OP_MATMUL, H7, A_T);

    // H9 = GELU(H8)
    msml_tensor_t* H9 = msml_tensor_emit_op_va(MSML_OP_GELU, H8);

    // H10 = SILU(H9)
    msml_tensor_t* H10 = msml_tensor_emit_op_va(MSML_OP_SILU, H9);

    // H11 = HARD_SIGMOID(H10)
    msml_tensor_t* H11 = msml_tensor_emit_op_va(MSML_OP_HARD_SIGMOID, H10);

    // H12 = TANH(H11)
    msml_tensor_t* H12 = msml_tensor_emit_op_va(MSML_OP_TANH, H11);

    // H13 = DIV(H12, H5)
    msml_tensor_t* H13 = msml_tensor_emit_op_va(MSML_OP_DIV, H12, H5);

    // H14 = CLONE(H13)
    msml_tensor_t* H14 = msml_tensor_emit_op_va(MSML_OP_CLONE, H13);

    // H15 = CLONE(H14)
    msml_tensor_t* H15 = msml_tensor_emit_op_va(MSML_OP_CLONE, H14);

    // H16 = STEP(H15)
    msml_tensor_t* H16 = msml_tensor_emit_op_va(MSML_OP_STEP, H15);

    // Final result
    msml_tensor_t* result = H16;

    // Compile the computation graph
    msml_compute_graph_t* gra = msml_compute_graph_compile(ctx, result, MSML_GRAPH_EVAL_ORDER_FORWARD, nullptr);
    msml_compute_graph_dump_to_dot(gra, "complex_graph.dot");

    // Assertions to verify the graph structure
    size_t total_nodes = msml_compute_graph_get_num_total_nodes(gra);
    size_t internal_nodes = msml_compute_graph_get_num_internal_nodes(gra);
    size_t leaf_nodes = msml_compute_graph_get_num_leaf_nodes(gra);

    ASSERT_GT(total_nodes, 20);
    ASSERT_GT(internal_nodes, 15);
    ASSERT_GT(leaf_nodes, 5);

    ASSERT_GT(msml_compute_graph_get_memory_usage(gra), sizeof(void*) * total_nodes);

    const msml_tensor_t** leaves = msml_compute_graph_get_leaf_nodes(gra, nullptr);

    // Function to check if a tensor is in the leaves
    auto leaves_contains = [](const msml_tensor_t** leaves, size_t count, msml_tensor_t* tensor) {
        for (size_t i = 0; i < count; ++i) {
            if (leaves[i] == tensor) return true;
        }
        return false;
    };

    ASSERT_TRUE(leaves_contains(leaves, leaf_nodes, A));
    ASSERT_TRUE(leaves_contains(leaves, leaf_nodes, B));
    ASSERT_TRUE(leaves_contains(leaves, leaf_nodes, C));
    ASSERT_TRUE(leaves_contains(leaves, leaf_nodes, D));
    ASSERT_TRUE(leaves_contains(leaves, leaf_nodes, E));

    size_t n_i = 0;
    const msml_tensor_t** nodes = msml_compute_graph_get_internal_nodes(gra, &n_i);
    ASSERT_EQ(nodes[n_i - 1], result); // Last node must be the evaluation root and result

    msml_ctx_destroy(ctx);
}
