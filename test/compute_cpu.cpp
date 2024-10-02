// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <gtest/gtest.h>
#include <msml.h>
#include <cmath>

#define impl_test_unary_op(name, op, scalar_op) \
    TEST(compute_cpu, name##_same_shape) { \
        msml_ctx_t* ctx = msml_ctx_create(nullptr); \
        \
        for (int64_t i0=1; i0 <= 14; ++i0) \
        for (int64_t i1=1; i1 <= 14; ++i1) \
        for (int64_t i2=1; i2 <= 14; ++i2) \
        for (int64_t i3=1; i3 <= 14; ++i3) { \
            msml_tensor_t* x = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, i0, i1, i2, i3); \
            msml_tensor_fill_random(x, 0.0f, 1.0f); \
            \
            msml_tensor_t* r = msml_tensor_isomorphic_clone(x); \
            msml_tensor_set_op(r, MSML_OP_##op); \
            msml_tensor_set_arg(r, 0, x); \
            \
            msml_tensor_evaluate(r); \
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
        \
        for (int64_t i0=1; i0 <= 14; ++i0) \
        for (int64_t i1=1; i1 <= 14; ++i1) \
        for (int64_t i2=1; i2 <= 14; ++i2) \
        for (int64_t i3=1; i3 <= 14; ++i3) { \
            msml_tensor_t* x = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, i0, i1, i2, i3); \
            msml_tensor_t* y = msml_tensor_isomorphic_clone(x); \
            msml_tensor_fill_random(x, 0.0f, 1.0f); \
            msml_tensor_fill_random(y, -5.0f, 5.0f); \
            \
            msml_tensor_t* r = msml_tensor_isomorphic_clone(x); \
            msml_tensor_set_op(r, MSML_OP_##op); \
            msml_tensor_set_arg(r, 0, x); \
            msml_tensor_set_arg(r, 1, y); \
            \
            msml_tensor_evaluate(r); \
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
        \
        for (int64_t factor=2; factor <= 8; ++factor) \
        for (int64_t i0=1; i0 <= 4; ++i0) \
        for (int64_t i1=1; i1 <= 4; ++i1) \
        for (int64_t i2=1; i2 <= 4; ++i2) \
        for (int64_t i3=1; i3 <= 4; ++i3) { \
            msml_tensor_t* x = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, i0*factor, i1*factor, i2*factor, i3*factor); \
            msml_tensor_t* y = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, i0, i1, i2, i3); \
            msml_tensor_fill_random(x, 0.0f, 1.0f); \
            msml_tensor_fill(y, 2.2f); \
            \
            msml_tensor_t* r = msml_tensor_isomorphic_clone(x); \
            msml_tensor_set_op(r, MSML_OP_##op); \
            msml_tensor_set_arg(r, 0, x); \
            msml_tensor_set_arg(r, 1, y); \
            \
            msml_tensor_evaluate(r); \
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

TEST(compute_cpu, heavy_compute_single_op) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* R = msml_tensor_create_3d(ctx, MSML_DTYPE_F32, 16384, 16384, 3);
    msml_tensor_t* X = msml_tensor_isomorphic_clone(R);
    msml_tensor_fill(X, 3.0);
    msml_tensor_t* Y = msml_tensor_isomorphic_clone(R);
    msml_tensor_set_op(Y, MSML_OP_MUL);
    msml_tensor_set_arg(Y, 0, R);
    msml_tensor_set_arg(Y, 1, X);
    msml_tensor_evaluate(Y);

    msml_ctx_destroy(ctx);
}
