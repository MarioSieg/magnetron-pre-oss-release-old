// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"
#include <cmath>

static constexpr std::int64_t k_lim_same_shape = 6;
static constexpr std::int64_t k_lim_broadcast = 3;

#define impl_test_unary_op(name, eps, op, scalar_op) \
    TEST(compute_cpu, name##_same_shape) { \
        wl_ctx_t* ctx = wl_ctx_create(nullptr); \
        \
        for (std::int64_t i0=1; i0 <= k_lim_same_shape; ++i0) \
        for (std::int64_t i1=1; i1 <= k_lim_same_shape; ++i1) \
        for (std::int64_t i2=1; i2 <= k_lim_same_shape; ++i2) \
        for (std::int64_t i3=1; i3 <= k_lim_same_shape; ++i3) \
        for (std::int64_t i4=1; i4 <= k_lim_same_shape; ++i4) \
        for (std::int64_t i5=1; i5 <= k_lim_same_shape; ++i5) { \
            wl_tensor_t* x = wl_tensor_create_6d(ctx, WL_DTYPE_F32, i0, i1, i2, i3, i4, i5); \
            wl_tensor_fill_random(x, 0.0f, 1.0f); \
            \
            wl_tensor_t* r = wl_tensor_emit_op_va(ctx, WL_OP_##op, x); \
            \
            const auto* b_x = wl_tensor_data_as_f32(x); \
            const auto* b_r = wl_tensor_data_as_f32(r); \
            ASSERT_EQ(wl_tensor_num_elements(x), wl_tensor_num_elements(r)); \
            for (std::int64_t i=0; i < wl_tensor_num_elements(x); ++i) { \
                ASSERT_NEAR(b_r[i], scalar_op(b_x[i]), (eps)); /* We use a larger absolute error than machine epsilon, because the BLAS uses SIMD for certain functions which have higher accuracy than the scalar lambdas. */ \
            } \
        } \
        \
        wl_ctx_destroy(ctx); \
    } \

impl_test_unary_op(abs, 1e-6, ABS, [](float x) -> float {
    return std::abs(x);
})

TEST(compute_cpu, neg_same_shape) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    for (std::int64_t i0=1; i0 <= k_lim_same_shape; ++i0)
    for (std::int64_t i1=1; i1 <= k_lim_same_shape; ++i1)
    for (std::int64_t i2=1; i2 <= k_lim_same_shape; ++i2)
    for (std::int64_t i3=1; i3 <= k_lim_same_shape; ++i3)
    for (std::int64_t i4=1; i4 <= k_lim_same_shape; ++i4)
    for (std::int64_t i5=1; i5 <= k_lim_same_shape; ++i5) {
        wl_tensor_t* x = wl_tensor_create_6d(ctx, WL_DTYPE_F32, i0, i1, i2, i3, i4, i5);
        wl_tensor_fill_random(x, 0.0f, 1.0f);
        wl_tensor_t* r = wl_tensor_emit_op_va(ctx, WL_OP_NEG, x);
        const auto* b_x = wl_tensor_data_as_f32(x);
        const auto* b_r = wl_tensor_data_as_f32(r);
        ASSERT_EQ(wl_tensor_num_elements(x), wl_tensor_num_elements(r));
        for (std::int64_t i=0; i < wl_tensor_num_elements(x); ++i) {
            ASSERT_EQ(b_r[i], -b_x[i]);
        }
    }
    wl_ctx_destroy(ctx);
}

impl_test_unary_op(step, 1e-6, STEP, [](float x) -> float {
    return x >= 0.0f ? 1.0f : 0.0f;
})

impl_test_unary_op(softmax, 1e-6, SOFTMAX, [](float x) -> float {
    return std::exp(x);
})
impl_test_unary_op(softmax_dv, 1e-6, SOFTMAX_DV, [](float x) -> float {
    return std::exp(x);
})

impl_test_unary_op(sigmoid, 1e-6, SIGMOID, [](float x) -> float {
    return 1.0f / (1.0f + std::exp(-x));
})
impl_test_unary_op(sigmoid_dv, 1e-6, SIGMOID_DV, [](float x) -> float {
    return x * (1.0f - x);
})

impl_test_unary_op(hard_sigmoid, 1e-6, HARD_SIGMOID, [](float x) -> float {
    return std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f));
})
//impl_test_unary_op(hard_sigmoid_dv, HARD_SIGMOID_DV, [](float x) -> float {
//    return -(std::exp(x) / ((std::exp(x)+1.0f)*(std::exp(x)+1.0f)));
//})

impl_test_unary_op(silu, 1e-6, SILU, [](float x) -> float {
    return x / (1.0f + std::exp(-x));
})
//impl_test_unary_op(silu_dv, SILU_DV, [](float x) -> float {
//    return -(std::exp(x) / ((std::exp(x)+1.0f)*(std::exp(x)+1.0f)));
//})

impl_test_unary_op(tanh, 1e-3, TANH, [](float x) -> float {
    return std::tanh(x);
})
impl_test_unary_op(tanh_dv, 1e-6, TANH_DV, [](float x) -> float {
    return 1.0f / (std::cosh(x)*std::cosh(x));
})

impl_test_unary_op(relu, 1e-6, RELU, [](float x) -> float {
    return std::max(x, 0.0f);
})
impl_test_unary_op(relu_dv, 1e-6, RELU_DV, [](float x) -> float {
    return x <= 0.0f ? 0.0f : 1.0f;
})

impl_test_unary_op(gelu, 1e-3, GELU, [](float x) -> float {
    return 0.5f*x*(1.0f + std::tanh(0.79788456080286535587989211986876f*x*(1.0f + 0.044715f*x*x)));
})
//impl_test_unary_op(gelu_dv, GELU_DV, [](float x) -> float {
//    return x <= 0.0f ? 0.0f : 1.0f;
//})

#undef impl_test_unary_op

#define impl_test_binary_op(name, op, scalar_op) \
    TEST(compute_cpu, name##_same_shape) { \
        wl_ctx_t* ctx = wl_ctx_create(nullptr); \
        for (std::int64_t i0=1; i0 <= k_lim_same_shape; ++i0) \
        for (std::int64_t i1=1; i1 <= k_lim_same_shape; ++i1) \
        for (std::int64_t i2=1; i2 <= k_lim_same_shape; ++i2) \
        for (std::int64_t i3=1; i3 <= k_lim_same_shape; ++i3) \
        for (std::int64_t i4=1; i4 <= k_lim_same_shape; ++i4) \
        for (std::int64_t i5=1; i5 <= k_lim_same_shape; ++i5) { \
            wl_tensor_t* x = wl_tensor_create_6d(ctx, WL_DTYPE_F32, i0, i1, i2, i3, i4, i5); \
            wl_tensor_t* y = wl_tensor_emit_op_va(ctx, WL_OP_CLONE, x); \
            wl_tensor_fill_random(x, 0.0f, 1.0f); \
            wl_tensor_fill_random(y, -5.0f, 5.0f); \
            \
            wl_tensor_t* r = wl_tensor_emit_op_va(ctx, WL_OP_##op, x, y); \
            \
            const auto* b_x = wl_tensor_data_as_f32(x); \
            const auto* b_y = wl_tensor_data_as_f32(y); \
            const auto* b_r = wl_tensor_data_as_f32(r); \
            ASSERT_EQ(wl_tensor_num_elements(x), wl_tensor_num_elements(y)); \
            ASSERT_EQ(wl_tensor_num_elements(r), wl_tensor_num_elements(y)); \
            for (std::int64_t i=0; i < wl_tensor_num_elements(x); ++i) { \
                ASSERT_FLOAT_EQ(b_r[i], b_x[i] scalar_op b_y[i]); \
            } \
        } \
        \
        wl_ctx_destroy(ctx); \
    } \
     \
    TEST(compute_cpu, name##_scalar_broadcast) { \
        wl_ctx_t* ctx = wl_ctx_create(nullptr); \
        for (std::int64_t factor=2; factor <= 4; ++factor) \
        for (std::int64_t i0=1; i0 <= k_lim_broadcast; ++i0) \
        for (std::int64_t i1=1; i1 <= k_lim_broadcast; ++i1) \
        for (std::int64_t i2=1; i2 <= k_lim_broadcast; ++i2) \
        for (std::int64_t i3=1; i3 <= k_lim_broadcast; ++i3) \
        for (std::int64_t i4=1; i4 <= k_lim_broadcast; ++i4) \
        for (std::int64_t i5=1; i5 <= k_lim_broadcast; ++i5) { \
            wl_tensor_t* x = wl_tensor_create_6d(ctx, WL_DTYPE_F32, i0*factor, i1*factor, i2*factor, i3*factor, i4*factor, i5*factor); \
            wl_tensor_t* y = wl_tensor_create_6d(ctx, WL_DTYPE_F32, i0, i1, i2, i3, i4, i5); \
            wl_tensor_fill_random(x, 0.0f, 1.0f); \
            wl_tensor_fill(y, 2.2f); \
            \
            wl_tensor_t* r = wl_tensor_emit_op_va(ctx, WL_OP_##op, x, y); \
            \
            const auto* b_x = wl_tensor_data_as_f32(x); \
            const auto* b_r = wl_tensor_data_as_f32(r); \
            ASSERT_EQ(wl_tensor_num_elements(r), wl_tensor_num_elements(x)); \
            ASSERT_NE(wl_tensor_num_elements(x), wl_tensor_num_elements(y)); \
            for (std::int64_t i=0; i < wl_tensor_num_elements(x); ++i) { \
                ASSERT_FLOAT_EQ(b_r[i], b_x[i] scalar_op 2.2f); \
            } \
        } \
        \
        wl_ctx_destroy(ctx); \
    }

impl_test_binary_op(add_f32, ADD, +)
impl_test_binary_op(sub_f32, SUB, -)
impl_test_binary_op(mul_f32, MUL, *)
impl_test_binary_op(div_f32, DIV, /)

#undef impl_test_binary_op

static void wl__inner_matmul_naive(
    const float* A,
    const float* B,
    float* C,
    const int64_t M,
    const int64_t N,
    const int64_t K
) {
    for (int64_t i = 0; i < M * N; ++i) C[i] = 0.0f;
    for (int64_t i = 0; i < M; ++i) {       // Rows of A and C
        for (int64_t k = 0; k < K; ++k) {   // Columns of A, Rows of B
            float a_ik = A[i * K + k];      // Access A[i][k]
            for (int64_t j = 0; j < N; ++j) { // Columns of B and C
                C[i * N + j] += a_ik * B[k * N + j]; // C[i][j] += A[i][k] * B[k][j]
            }
        }
    }
}

TEST(compute_cpu, matmul_inner_naive) {
    static constexpr float A[6] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    static constexpr float B[2] = {0.5f, -1.0f};
    float C[3];
    wl__inner_matmul_naive(A, B, C, 3, 1, 2);
    ASSERT_FLOAT_EQ(C[0], -1.5f);
    ASSERT_FLOAT_EQ(C[1], -2.5f);
    ASSERT_FLOAT_EQ(C[2], -3.5f);
}

TEST(compute_cpu, matmul_f32_same_shape_2x2) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    static constexpr float A_values[2][2] = {
        {1.6354027, -1.3607267},
        {1.8556793, 1.1689897}
    };
    static constexpr float B_values[2][2] = {
        {-0.6105532, 0.10695228},
        {-1.0069681, -0.40955952}
    };

    // Manually set known values for A and B
    wl_tensor_t* A = wl_tensor_create_2d(ctx, WL_DTYPE_F32, 2, 2);
    wl_tensor_copy_buffer_from(A, A_values, sizeof(A_values));

    wl_tensor_t* B = wl_tensor_create_2d(ctx, WL_DTYPE_F32, 2, 2);
    wl_tensor_copy_buffer_from(B, B_values, sizeof(B_values));

    // Create result tensor R for matrix multiplication
    wl_tensor_t* params[2] = {A, B};
    wl_tensor_t* R = wl_tensor_operator(ctx, WL_OP_MATMUL, params, 2, nullptr);
    wl_tensor_print(R, true, true);
    auto* buf = wl_tensor_data_as_f32(R);

    static constexpr float expected[2*2] = {
        0.3717081,   0.7322086,
        -2.3101263, -0.28030172
    };

    for (int i = 0; i < 2*2; ++i) {
        ASSERT_FLOAT_EQ(buf[i], expected[i]);
    }

    wl_ctx_destroy(ctx);
}

TEST(compute_cpu, matmul_f32_different_shape_2x2) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    static constexpr  float AV[3*2] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    static constexpr float BV[2] = {0.5f, -1.0f};

    // Manually set known values for A and B
    wl_tensor_t* A = wl_tensor_create_2d(ctx, WL_DTYPE_F32, 3, 2);
    wl_tensor_copy_buffer_from(A, AV, sizeof(AV));

    wl_tensor_t* B = wl_tensor_create_1d(ctx, WL_DTYPE_F32, 2);
    wl_tensor_copy_buffer_from(B, BV, sizeof(BV));

    // Create result tensor R for matrix multiplication
    wl_tensor_t* params[2] = {A, B};
    wl_tensor_t* R = wl_tensor_operator(ctx, WL_OP_MATMUL, params, 2, nullptr);
    //ASSERT_EQ(wl_tensor_rank(R), 1);
    ASSERT_EQ(wl_tensor_shape(R)[0], 3);
    const auto* C = wl_tensor_data_as_f32(R);
    //wl__inner_matmul_naive(A, B, C, 3, 1, 2);
    ASSERT_FLOAT_EQ(C[0], -1.5f);
    ASSERT_FLOAT_EQ(C[1], -2.5f);
    ASSERT_FLOAT_EQ(C[2], -3.5f);

    wl_ctx_destroy(ctx);
}

TEST(compute_cpu, arithmetic_mean) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    wl_tensor_t* A = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 4096, 32, 3, 2);
    wl_tensor_fill_random(A, -1.0f, 1.0f);
    wl_tensor_t* R = wl_tensor_emit_op_va(ctx, WL_OP_MEAN, A);
    ASSERT_NE(R, nullptr);
    double a_mean = 0.0;
    for (std::int64_t i=0; i < wl_tensor_num_elements(A); ++i)
        a_mean += static_cast<double>(wl_tensor_data_as_f32(A)[i]);
    a_mean /= static_cast<double>(wl_tensor_num_elements(A));
    double b_mean = 0.0;
    for (std::int64_t i=0; i < wl_tensor_num_elements(R); ++i)
        b_mean += static_cast<double>(wl_tensor_data_as_f32(R)[i]);
    b_mean /= static_cast<double>(wl_tensor_num_elements(R));
    ASSERT_NEAR(static_cast<float>(a_mean), static_cast<float>(b_mean), 1e-6);
    wl_ctx_destroy(ctx);
}

TEST(compute_cpu, heavy_compute_single_op) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    wl_tensor_t* A = wl_tensor_create_3d(ctx, WL_DTYPE_F32, 16384, 16384, 3);
    wl_tensor_t* B = wl_tensor_emit_op_va(ctx, WL_OP_CLONE, A);
    wl_tensor_fill(B, 3.0);
    wl_tensor_t* R = wl_tensor_emit_op_va(ctx, WL_OP_ADD, A, B);
    ASSERT_NE(R, nullptr);
    wl_ctx_destroy(ctx);
}

TEST(compute_cpu, heavy_compute_single_op_scalar) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    wl_tensor_t* A = wl_tensor_create_1d(ctx, WL_DTYPE_F32, 1);
    wl_tensor_t* B =  wl_tensor_emit_op_va(ctx, WL_OP_CLONE, A);
    wl_tensor_fill(B, 3.0);
    wl_tensor_t* R = wl_tensor_emit_op_va(ctx, WL_OP_ADD, A, B);
    ASSERT_NE(R, nullptr);
    wl_ctx_destroy(ctx);
}
