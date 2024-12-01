#pragma once

#include <wavelet.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>

// Helper to compile DAG and execute
inline auto wl_tensor_evaluate_static_graph(wl_tensor_t* root, wl_graph_eval_order_t order = WL_GRAPH_EVAL_ORDER_FORWARD) -> wl_tensor_t* {
    auto* ctx = wl_tensor_get_ctx(root);
    auto* gra = wl_compute_graph_compile(ctx, root, order, nullptr);
    return wl_compute_graph_execute(gra);
}

template <bool Inplace=false, typename... Args>
[[nodiscard]] inline auto wl_tensor_emit_op_va(wl_ctx_t* ctx, wl_op_t op, Args&&... args) -> wl_tensor_t* {
    std::array<wl_tensor_t*, sizeof...(Args)> tensors {args...};
    auto* opt = wl_tensor_operator(ctx, op, Inplace, tensors.data(), tensors.size(), nullptr);
    if (!opt) throw std::runtime_error("wl_tensor_emit_op_va: failed to emit operation");
    return opt;
}

template <bool Inplace=false, typename... Args>
[[nodiscard]] inline auto wl_tensor_emit_op_va_op_params(wl_ctx_t* ctx, wl_op_t op, wl_op_param_t para, Args&&... args) -> wl_tensor_t* {
    std::array<wl_tensor_t*, sizeof...(Args)> tensors {args...};
    const wl_op_param_t op_params[WL_MAX_OP_PARAMS] {para};
    auto* opt = wl_tensor_operator(ctx, op, Inplace, tensors.data(), tensors.size(), &op_params);
    if (!opt) throw std::runtime_error("wl_tensor_emit_op_va: failed to emit operation");
    return opt;
}

inline auto wl_tensor_buf_f32_to_vec(const wl_tensor_t* tensor, std::vector<float>& out) -> void {
    out.clear();
    out.reserve(wl_tensor_num_elements(tensor));
    std::memcpy(out.data(), wl_tensor_data_as_f32(tensor), wl_tensor_data_size(tensor));
}
