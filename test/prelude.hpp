#pragma once

#include <array>
#include <gtest/gtest.h>
#include <wavelet.h>

// Helper to compile DAG and execute
inline auto wl_tensor_evaluate_static_graph(wl_tensor_t* root, wl_graph_eval_order_t order = WL_GRAPH_EVAL_ORDER_FORWARD) -> wl_tensor_t* {
    auto* ctx = wl_tensor_get_ctx(root);
    auto* gra = wl_compute_graph_compile(ctx, root, order, nullptr);
    return wl_compute_graph_execute(gra);
}

template <typename... Args>
[[nodiscard]] inline auto wl_tensor_emit_op_va(wl_ctx_t* ctx, wl_op_t op, Args&&... args) -> wl_tensor_t* {
    std::array<wl_tensor_t*, sizeof...(Args)> tensors {args...};
    auto* opt = wl_tensor_operator(ctx, op, tensors.data(), tensors.size(), nullptr);
    if (!opt) throw std::runtime_error("wl_tensor_emit_op_va: failed to emit operation");
    return opt;
}
