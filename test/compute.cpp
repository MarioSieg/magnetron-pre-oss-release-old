// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <gtest/gtest.h>
#include <msml.h>

#define impl_test_binary_op(name, op, scalar_op) \
    TEST(compute, name##_same_shape) { \
        msml_ctx_t* ctx = msml_ctx_create(nullptr); \
        \
        for (int64_t i0=1; i0 <= 16; ++i0) \
        for (int64_t i1=1; i1 <= 16; ++i1) \
        for (int64_t i2=1; i2 <= 16; ++i2) \
        for (int64_t i3=1; i3 <= 16; ++i3) { \
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
    TEST(compute, name##_scalar_broadcast) { \
        msml_ctx_t* ctx = msml_ctx_create(nullptr); \
        \
        for (int64_t factor=2; factor <= 8; ++factor) \
        for (int64_t i0=1; i0 <= 5; ++i0) \
        for (int64_t i1=1; i1 <= 5; ++i1) \
        for (int64_t i2=1; i2 <= 5; ++i2) \
        for (int64_t i3=1; i3 <= 5; ++i3) { \
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
            const auto* b_y = msml_tensor_buf_f32(y); \
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
