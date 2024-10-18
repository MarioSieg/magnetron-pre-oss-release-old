#pragma once

#include <gtest/gtest.h>
#include <msml.h>

// Helper to compile DAG and execute
inline auto msml_tensor_evaluate_static_graph(msml_tensor_t* root, msml_graph_eval_order_t order = MSML_GRAPH_EVAL_ORDER_FORWARD) -> msml_tensor_t* {
    auto* ctx = msml_tensor_get_ctx(root);
    auto* gra = msml_compute_graph_compile(ctx, root, order, nullptr);
    return msml_compute_graph_execute(gra);
}

template <typename... Args>
[[nodiscard]] inline auto msml_tensor_emit_op_va(msml_ctx_t* ctx, msml_op_t op, Args&&... args) -> msml_tensor_t* {
    std::array<msml_tensor_t*, sizeof...(Args)> tensors {args...};
    return msml_tensor_operator(ctx, op, tensors.data(), tensors.size(), nullptr);
}
