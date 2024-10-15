// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <gtest/gtest.h>
#include <msml.h>
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
            msml_tensor_t* r = msml_tensor_isomorphic(x); \
            msml_tensor_set_op(r, MSML_OP_##op); \
            msml_tensor_set_arg(r, 0, x); \
            \
            msml_tensor_evaluate(r, MSML_GRAPH_EVAL_ORDER_FORWARD); \
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
        \
        for (int64_t i0=1; i0 <= 9; ++i0) \
        for (int64_t i1=1; i1 <= 9; ++i1) \
        for (int64_t i2=1; i2 <= 9; ++i2) \
        for (int64_t i3=1; i3 <= 9; ++i3) { \
            msml_tensor_t* x = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, i0, i1, i2, i3); \
            msml_tensor_t* y = msml_tensor_isomorphic(x); \
            msml_tensor_fill_random(x, 0.0f, 1.0f); \
            msml_tensor_fill_random(y, -5.0f, 5.0f); \
            \
            msml_tensor_t* r = msml_tensor_isomorphic(x); \
            msml_tensor_set_op(r, MSML_OP_##op); \
            msml_tensor_set_arg(r, 0, x); \
            msml_tensor_set_arg(r, 1, y); \
            \
            msml_tensor_evaluate(r, MSML_GRAPH_EVAL_ORDER_FORWARD); \
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
            msml_tensor_t* r = msml_tensor_isomorphic(x); \
            msml_tensor_set_op(r, MSML_OP_##op); \
            msml_tensor_set_arg(r, 0, x); \
            msml_tensor_set_arg(r, 1, y); \
            \
            msml_tensor_evaluate(r, MSML_GRAPH_EVAL_ORDER_FORWARD); \
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
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };
    static constexpr float B_values[2][2] = {
        {5.0f, 6.0f},
        {7.0f, 8.0f}
    };

    // Manually set known values for A and B
    msml_tensor_t* A = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_copy_buffer_from(A, A_values, sizeof(A_values));

    msml_tensor_t* B = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 2, 2);
    msml_tensor_copy_buffer_from(B, B_values, sizeof(B_values));

    // Create result tensor R for matrix multiplication
    msml_tensor_t* params[2] = {A, B};
    msml_tensor_t* R = msml_tensor_emit_op(MSML_OP_MATMUL, params, 2);
    ASSERT_NE(R, nullptr);

    msml_tensor_evaluate(R, MSML_GRAPH_EVAL_ORDER_FORWARD);

    auto* buf = msml_tensor_buf_f32(R);

    // Manually compute the expected result of Rᵀ = A x Bᵀ
    static constexpr float expected[2][2] = {
        {17.0f,  23.0f},
        {39.0f,  53.0f}
    };

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_FLOAT_EQ(buf[i*2 + j], expected[j][i]); /* We need to transpose the expected matrix Rᵀ = A x Bᵀ as Rᵀ is transposed too */
        }
    }

    msml_ctx_destroy(ctx);
}

TEST(compute_cpu, matmul_f32) {
    static constexpr std::size_t M = 4, N = 16, K = 36;
    static constexpr float A_mtx[M * K] = {
        2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
        10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
        7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
    };
    static constexpr float B_mtx[N * K] = {
        9.0f, 7.0f, 1.0f, 3.0f, 5.0f, 9.0f, 7.0f, 6.0f, 1.0f, 10.0f, 1.0f, 1.0f, 7.0f, 2.0f, 4.0f, 9.0f, 10.0f, 4.0f, 5.0f, 5.0f, 7.0f, 1.0f, 7.0f, 7.0f, 2.0f, 9.0f, 5.0f, 10.0f, 7.0f, 4.0f, 8.0f, 9.0f, 9.0f, 3.0f, 10.0f, 2.0f,
        4.0f, 6.0f, 10.0f, 9.0f, 5.0f, 1.0f, 8.0f, 7.0f, 4.0f, 7.0f, 2.0f, 6.0f, 5.0f, 3.0f, 1.0f, 10.0f, 8.0f, 4.0f, 8.0f, 3.0f, 7.0f, 1.0f, 2.0f, 7.0f, 6.0f, 8.0f, 6.0f, 5.0f, 2.0f, 3.0f, 1.0f, 1.0f, 2.0f, 5.0f, 7.0f, 1.0f,
        8.0f, 2.0f, 8.0f, 8.0f, 8.0f, 8.0f, 4.0f, 4.0f, 6.0f, 10.0f, 10.0f, 9.0f, 2.0f, 9.0f, 3.0f, 7.0f, 7.0f, 1.0f, 4.0f, 9.0f, 1.0f, 2.0f, 3.0f, 6.0f, 1.0f, 10.0f, 5.0f, 8.0f, 9.0f, 4.0f, 6.0f, 2.0f, 3.0f, 1.0f, 2.0f, 7.0f,
        5.0f, 1.0f, 7.0f, 2.0f, 9.0f, 10.0f, 9.0f, 5.0f, 2.0f, 5.0f, 4.0f, 10.0f, 9.0f, 9.0f, 1.0f, 9.0f, 8.0f, 8.0f, 9.0f, 4.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 8.0f, 4.0f, 5.0f, 10.0f, 7.0f, 6.0f, 2.0f, 1.0f, 10.0f, 10.0f, 7.0f,
        9.0f, 4.0f, 5.0f, 9.0f, 5.0f, 10.0f, 10.0f, 3.0f, 6.0f, 6.0f, 4.0f, 4.0f, 4.0f, 8.0f, 5.0f, 4.0f, 9.0f, 1.0f, 9.0f, 9.0f, 1.0f, 7.0f, 9.0f, 2.0f, 10.0f, 9.0f, 10.0f, 8.0f, 3.0f, 3.0f, 9.0f, 3.0f, 9.0f, 10.0f, 1.0f, 8.0f,
        9.0f, 2.0f, 6.0f, 9.0f, 7.0f, 2.0f, 3.0f, 5.0f, 3.0f, 6.0f, 9.0f, 7.0f, 3.0f, 7.0f, 6.0f, 4.0f, 10.0f, 3.0f, 5.0f, 7.0f, 2.0f, 9.0f, 3.0f, 2.0f, 2.0f, 10.0f, 8.0f, 7.0f, 3.0f, 10.0f, 6.0f, 3.0f, 1.0f, 1.0f, 4.0f, 10.0f,
        2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
        10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
        7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
        6.0f, 2.0f, 3.0f, 3.0f, 3.0f, 7.0f, 5.0f, 1.0f, 8.0f, 1.0f, 4.0f, 5.0f, 1.0f, 1.0f, 6.0f, 4.0f, 2.0f, 1.0f, 7.0f, 8.0f, 6.0f, 1.0f, 1.0f, 5.0f, 6.0f, 5.0f, 10.0f, 6.0f, 7.0f, 5.0f, 9.0f, 3.0f, 2.0f, 7.0f, 9.0f, 4.0f,
        2.0f, 5.0f, 9.0f, 5.0f, 10.0f, 3.0f, 1.0f, 8.0f, 1.0f, 7.0f, 1.0f, 8.0f, 1.0f, 6.0f, 7.0f, 8.0f, 4.0f, 9.0f, 5.0f, 10.0f, 3.0f, 7.0f, 6.0f, 8.0f, 8.0f, 5.0f, 6.0f, 8.0f, 10.0f, 9.0f, 4.0f, 1.0f, 3.0f, 3.0f, 4.0f, 7.0f,
        8.0f, 2.0f, 6.0f, 6.0f, 5.0f, 1.0f, 3.0f, 7.0f, 1.0f, 7.0f, 2.0f, 2.0f, 2.0f, 8.0f, 4.0f, 1.0f, 1.0f, 5.0f, 9.0f, 4.0f, 1.0f, 2.0f, 3.0f, 10.0f, 1.0f, 4.0f, 9.0f, 9.0f, 6.0f, 8.0f, 8.0f, 1.0f, 9.0f, 10.0f, 4.0f, 1.0f,
        8.0f, 5.0f, 8.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 1.0f, 9.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 1.0f, 4.0f, 6.0f, 7.0f, 7.0f, 7.0f, 8.0f, 7.0f, 8.0f, 8.0f, 2.0f, 10.0f, 2.0f, 7.0f, 3.0f, 8.0f, 3.0f,
        8.0f, 7.0f, 6.0f, 2.0f, 4.0f, 10.0f, 10.0f, 6.0f, 10.0f, 3.0f, 7.0f, 6.0f, 4.0f, 3.0f, 5.0f, 5.0f, 5.0f, 3.0f, 8.0f, 10.0f, 3.0f, 4.0f, 8.0f, 4.0f, 2.0f, 6.0f, 8.0f, 9.0f, 6.0f, 9.0f, 4.0f, 3.0f, 5.0f, 2.0f, 2.0f, 6.0f,
        10.0f, 6.0f, 2.0f, 1.0f, 7.0f, 5.0f, 6.0f, 4.0f, 1.0f, 9.0f, 10.0f, 2.0f, 4.0f, 5.0f, 8.0f, 5.0f, 7.0f, 4.0f, 7.0f, 6.0f, 3.0f, 9.0f, 2.0f, 1.0f, 4.0f, 2.0f, 6.0f, 6.0f, 3.0f, 3.0f, 2.0f, 8.0f, 5.0f, 9.0f, 3.0f, 4.0f,
    };
    static constexpr float result_mtx[M * N] = {
        1224.0f, 1023.0f, 1158.0f,1259.0f,1359.0f,1194.0f,1535.0f,1247.0f,1185.0f,1029.0f,889.0f,1182.0f,955.0f,1179.0f,1147.0f,1048.0f,
        1216.0f, 1087.0f, 1239.0f,1361.0f,1392.0f,1260.0f,1247.0f,1563.0f,1167.0f,1052.0f,942.0f,1214.0f,1045.0f,1134.0f,1264.0f,1126.0f,
        1125.0f, 966.0f, 1079.0f,1333.0f,1287.0f,1101.0f,1185.0f,1167.0f,1368.0f,990.0f,967.0f,1121.0f,971.0f,1086.0f,1130.0f,980.0f,
        999.0f, 902.0f, 1020.0f,1056.0f,1076.0f,929.0f,1029.0f,1052.0f,990.0f,1108.0f,823.0f,989.0f,759.0f,1041.0f,1003.0f,870.0f
    };

    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* A = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, K, M);
    msml_tensor_copy_buffer_from(A, A_mtx, sizeof(A_mtx));
    msml_tensor_t* B = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, K, N);
    msml_tensor_copy_buffer_from(B, B_mtx, sizeof(B_mtx));
    msml_tensor_t* params[2] = {A, B};
    msml_tensor_t* R = msml_tensor_emit_op(MSML_OP_MATMUL, params, 2);
    ASSERT_NE(R, nullptr);
    msml_tensor_t* RR = msml_tensor_clone(msml_tensor_transpose(R));
    msml_tensor_evaluate(RR, MSML_GRAPH_EVAL_ORDER_FORWARD);
    msml_tensor_print(RR, true);

    auto* buf = msml_tensor_buf_f32(RR);
    for (std::size_t i = 0; i < M * N; ++i) {
        ASSERT_FLOAT_EQ(buf[i], result_mtx[i]);
    }
    msml_ctx_destroy(ctx);
}

TEST(compute_cpu, heavy_compute_single_op) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_tensor_t* A = msml_tensor_create_3d(ctx, MSML_DTYPE_F32, 16384, 16384, 3);
    msml_tensor_t* B = msml_tensor_isomorphic(A);
    msml_tensor_fill(B, 3.0);
    msml_tensor_t* params[2] = {A, B};
    msml_tensor_t* R = msml_tensor_emit_op(MSML_OP_MUL, params, 2);
    ASSERT_NE(R, nullptr);
    msml_tensor_evaluate(R, MSML_GRAPH_EVAL_ORDER_FORWARD);
    msml_ctx_destroy(ctx);
}

TEST(compute_cpu, heavy_compute_single_op_scalar) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_tensor_t* A = msml_tensor_create_1d(ctx, MSML_DTYPE_F32, 1);
    msml_tensor_t* B = msml_tensor_isomorphic(A);
    msml_tensor_fill(B, 3.0);
    msml_tensor_t* params[2] = {A, B};
    msml_tensor_t* R = msml_tensor_emit_op(MSML_OP_ADD, params, 2);
    ASSERT_NE(R, nullptr);
    msml_tensor_evaluate(R, MSML_GRAPH_EVAL_ORDER_FORWARD);
    msml_ctx_destroy(ctx);
}
