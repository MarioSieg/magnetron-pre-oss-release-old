#pragma once

#include <wavelet_internal.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>

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
