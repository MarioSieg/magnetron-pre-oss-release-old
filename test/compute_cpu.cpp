// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"
#include <cmath>

#define impl_test_unary_op(name, op, scalar_op) \
    TEST(compute_cpu, name##_same_shape) { \
        msml_ctx_t* ctx = msml_ctx_create(nullptr); \
        \
        for (int64_t i0=1; i0 <= 9; ++i0) \
        for (int64_t i1=1; i1 <= 9; ++i1) \
        for (int64_t i2=1; i2 <= 9; ++i2) \
        for (int64_t i3=1; i3 <= 9; ++i3) { \
            msml_tensor_t* x = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, i0, i1, i2, i3); \
            msml_tensor_fill_random(x, 0.0f, 1.0f); \
            \
            msml_tensor_t* r = msml_tensor_emit_op_va(ctx, MSML_OP_##op, x); \
            \
            const auto* b_x = msml_tensor_buf_f32(x); \
            const auto* b_r = msml_tensor_buf_f32(r); \
            ASSERT_EQ(msml_tensor_buf_len(x), msml_tensor_buf_len(r)); \
            for (std::int64_t i=0; i < msml_tensor_buf_len(x); ++i) { \
                ASSERT_NEAR(b_r[i], scalar_op(b_x[i]), 0.000001); /* We use a larger absolute error than machine epsilon, because the BLAS uses SIMD for certain functions which have higher accuracy than the scalar lambdas. */ \
            } \
        } \
        \
        msml_ctx_destroy(ctx); \
    } \

impl_test_unary_op(softmax, SOFTMAX, [](float x) -> float {
    return std::exp(x);
})
impl_test_unary_op(softmax_dv, SOFTMAX_DV, [](float x) -> float {
    return std::exp(x);
})

impl_test_unary_op(sigmoid, SIGMOID, [](float x) -> float {
    return 1.0f / (1.0f + std::exp(-x));
})
impl_test_unary_op(sigmoid_dv, SIGMOID_DV, [](float x) -> float {
    return -(std::exp(x) / ((std::exp(x)+1.0f)*(std::exp(x)+1.0f)));
})

impl_test_unary_op(hard_sigmoid, HARD_SIGMOID, [](float x) -> float {
    return std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f));
})
//impl_test_unary_op(hard_sigmoid_dv, HARD_SIGMOID_DV, [](float x) -> float {
//    return -(std::exp(x) / ((std::exp(x)+1.0f)*(std::exp(x)+1.0f)));
//})

impl_test_unary_op(silu, SILU, [](float x) -> float {
    return x / (1.0f + std::exp(-x));
})
//impl_test_unary_op(silu_dv, SILU_DV, [](float x) -> float {
//    return -(std::exp(x) / ((std::exp(x)+1.0f)*(std::exp(x)+1.0f)));
//})

impl_test_unary_op(tanh, TANH, [](float x) -> float {
    return std::tanh(x);
})
impl_test_unary_op(tanh_dv, TANH_DV, [](float x) -> float {
    return 1.0f / (std::cosh(x)*std::cosh(x));
})

impl_test_unary_op(relu, RELU, [](float x) -> float {
    return std::max(x, 0.0f);
})
impl_test_unary_op(relu_dv, RELU_DV, [](float x) -> float {
    return x <= 0.0f ? 0.0f : 1.0f;
})

impl_test_unary_op(gelu, GELU, [](float x) -> float {
    return 0.5f*x*(1.0f + tanhf(0.79788456080286535587989211986876f*x*(1.0f + 0.044715f*x*x)));
})
//impl_test_unary_op(gelu_dv, GELU_DV, [](float x) -> float {
//    return x <= 0.0f ? 0.0f : 1.0f;
//})

#undef impl_test_unary_op

#define impl_test_binary_op(name, op, scalar_op) \
    TEST(compute_cpu, name##_same_shape) { \
        msml_ctx_t* ctx = msml_ctx_create(nullptr); \
        for (int64_t i0=1; i0 <= 9; ++i0) \
        for (int64_t i1=1; i1 <= 9; ++i1) \
        for (int64_t i2=1; i2 <= 9; ++i2) \
        for (int64_t i3=1; i3 <= 9; ++i3) { \
            msml_tensor_t* x = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, i0, i1, i2, i3); \
            msml_tensor_t* y = msml_tensor_emit_op_va(ctx, MSML_OP_CLONE, x); \
            msml_tensor_fill_random(x, 0.0f, 1.0f); \
            msml_tensor_fill_random(y, -5.0f, 5.0f); \
            \
            msml_tensor_t* r = msml_tensor_emit_op_va(ctx, MSML_OP_##op, x, y); \
            \
            const auto* b_x = msml_tensor_buf_f32(x); \
            const auto* b_y = msml_tensor_buf_f32(y); \
            const auto* b_r = msml_tensor_buf_f32(r); \
            ASSERT_EQ(msml_tensor_buf_len(x), msml_tensor_buf_len(y)); \
            ASSERT_EQ(msml_tensor_buf_len(r), msml_tensor_buf_len(y)); \
            for (std::int64_t i=0; i < msml_tensor_buf_len(x); ++i) { \
                ASSERT_FLOAT_EQ(b_r[i], b_x[i] scalar_op b_y[i]); \
            } \
        } \
        \
        msml_ctx_destroy(ctx); \
    } \
     \
    TEST(compute_cpu, name##_scalar_broadcast) { \
        msml_ctx_t* ctx = msml_ctx_create(nullptr); \
        for (int64_t factor=2; factor <= 4; ++factor) \
        for (int64_t i0=1; i0 <= 4; ++i0) \
        for (int64_t i1=1; i1 <= 4; ++i1) \
        for (int64_t i2=1; i2 <= 4; ++i2) \
        for (int64_t i3=1; i3 <= 4; ++i3) { \
            msml_tensor_t* x = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, i0*factor, i1*factor, i2*factor, i3*factor); \
            msml_tensor_t* y = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, i0, i1, i2, i3); \
            msml_tensor_fill_random(x, 0.0f, 1.0f); \
            msml_tensor_fill(y, 2.2f); \
            \
            msml_tensor_t* r = msml_tensor_emit_op_va(ctx, MSML_OP_##op, x, y); \
            \
            const auto* b_x = msml_tensor_buf_f32(x); \
            const auto* b_r = msml_tensor_buf_f32(r); \
            ASSERT_EQ(msml_tensor_buf_len(r), msml_tensor_buf_len(x)); \
            ASSERT_NE(msml_tensor_buf_len(x), msml_tensor_buf_len(y)); \
            for (std::int64_t i=0; i < msml_tensor_buf_len(x); ++i) { \
                ASSERT_FLOAT_EQ(b_r[i], b_x[i] scalar_op 2.2f); \
            } \
        } \
        \
        msml_ctx_destroy(ctx); \
    }

impl_test_binary_op(add_f32, ADD, +)
impl_test_binary_op(sub_f32, SUB, -)
impl_test_binary_op(mul_f32, MUL, *)
impl_test_binary_op(div_f32, DIV, /)

#undef impl_test_binary_op

TEST(compute_cpu, matmul_f32_same_shape_2x2) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    static constexpr float A_values[2][2] = {
        {1.6354027, -1.3607267},
        {1.8556793, 1.1689897}
    };
    static constexpr float B_values[2][2] = {
        {-0.6105532, 0.10695228},
        {-1.0069681, -0.40955952}
    };

    // Manually set known values for A and B
    msml_tensor_t* A = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_copy_buffer_from(A, A_values, sizeof(A_values));

    msml_tensor_t* B = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_copy_buffer_from(B, B_values, sizeof(B_values));

    // Create result tensor R for matrix multiplication
    msml_tensor_t* params[2] = {A, B};
    msml_tensor_t* R = msml_tensor_operator(ctx, MSML_OP_MATMUL, params, 2, nullptr);

    auto* buf = msml_tensor_buf_f32(R);

    static constexpr float expected[2*2] = {
        0.3717081,   0.7322086,
        -2.3101263, -0.28030172
    };

    for (int i = 0; i < 2*2; ++i) {
        ASSERT_FLOAT_EQ(buf[i], expected[i]);
    }

    msml_ctx_destroy(ctx);
}

TEST(compute_cpu, heavy_compute_single_op) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_tensor_t* A = msml_tensor_create_3d(ctx, MSML_DTYPE_F32, 16384, 16384, 3);
    msml_tensor_t* B = msml_tensor_emit_op_va(ctx, MSML_OP_CLONE, A);
    msml_tensor_fill(B, 3.0);
    msml_tensor_t* R = msml_tensor_emit_op_va(ctx, MSML_OP_ADD, A, B);
    ASSERT_NE(R, nullptr);
    msml_ctx_destroy(ctx);
}

TEST(compute_cpu, heavy_compute_single_op_scalar) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_tensor_t* A = msml_tensor_create_1d(ctx, MSML_DTYPE_F32, 1);
    msml_tensor_t* B =  msml_tensor_emit_op_va(ctx, MSML_OP_CLONE, A);
    msml_tensor_fill(B, 3.0);
    msml_tensor_t* R = msml_tensor_emit_op_va(ctx, MSML_OP_ADD, A, B);
    ASSERT_NE(R, nullptr);
    msml_ctx_destroy(ctx);
}
