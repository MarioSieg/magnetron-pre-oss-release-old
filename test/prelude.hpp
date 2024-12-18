#pragma once

#include <magnetron_internal.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>

template <bool Inplace=false, typename... Args>
[[nodiscard]] inline auto mag_tensor_emit_op_va(mag_ctx_t* ctx, mag_op_t op, Args&&... args) -> mag_tensor_t* {
    std::array<mag_tensor_t*, sizeof...(Args)> tensors {args...};
    auto* opt = mag_tensor_operator(ctx, op, Inplace, tensors.data(), tensors.size(), nullptr);
    if (!opt) throw std::runtime_error("mag_tensor_emit_op_va: failed to emit operation");
    return opt;
}

template <bool Inplace=false, typename... Args>
[[nodiscard]] inline auto mag_tensor_emit_op_va_op_params(mag_ctx_t* ctx, mag_op_t op, mag_op_param_t para, Args&&... args) -> mag_tensor_t* {
    std::array<mag_tensor_t*, sizeof...(Args)> tensors {args...};
    const mag_op_param_t op_params[MAG_MAX_OP_PARAMS] {para};
    auto* opt = mag_tensor_operator(ctx, op, Inplace, tensors.data(), tensors.size(), &op_params);
    if (!opt) throw std::runtime_error("mag_tensor_emit_op_va: failed to emit operation");
    return opt;
}

inline auto mag_tensor_buf_f32_to_vec(const mag_tensor_t* tensor, std::vector<float>& out) -> void {
    out.clear();
    out.reserve(mag_tensor_numel(tensor));
    std::memcpy(out.data(), mag_tensor_data_ptr(tensor), mag_tensor_data_size(tensor));
}
