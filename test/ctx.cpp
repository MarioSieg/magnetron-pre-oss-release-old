// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(ctx, create_destroy_cpu) {
    mag_set_set_log_mode(true);
    mag_ctx_t* ctx = mag_ctx_create2(MAG_COMPUTE_DEVICE_TYPE_CPU);
    ASSERT_NE(ctx, nullptr);
    mag_ctx_destroy(ctx);
    mag_set_set_log_mode(false);
}

#ifdef MAG_ENABLE_CUDA
TEST(ctx, create_destroy_cuda) {
    mag_set_set_log_mode(true);
    mag_ctx_t* ctx = mag_ctx_create2(MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA);
    ASSERT_NE(ctx, nullptr);
    mag_ctx_destroy(ctx);
    mag_set_set_log_mode(false);
}
#endif

#ifndef _MSC_VER // MSVC fucks around with linking a __declspex(dllexport) ed function ptr. TODO: fix
TEST(ctx, alloc) {
    int* i = static_cast<int*>((*mag_alloc)(nullptr, sizeof(int)));
    ASSERT_NE(i, nullptr);
    *i = -1;
    ASSERT_EQ(*i, -1);
    (*mag_alloc)(i, 0);
}
#endif

TEST(ctx, alloc_aligned) {
    int* i = static_cast<int*>(mag_alloc_aligned(sizeof(int), 512));
    ASSERT_NE(i, nullptr);
    *i = -1;
    ASSERT_EQ(*i, -1);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(i) % 512, 0);
    mag_free_aligned(i);
}

TEST(atomics_ops, StoreLoadTest) {
    mag_atomic_t val = 0;
    mag_atomic_store(&val, 42, MAG_MO_RELAXED);
    int loaded = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(loaded, 42);
}

TEST(atomics_ops, FetchAddTest) {
    mag_atomic_t val = 10;
    int old = mag_atomic_fetch_add(&val, 5, MAG_MO_RELAXED);
    EXPECT_EQ(old, 10);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 15);
}

TEST(atomics_ops, FetchSubTest) {
    mag_atomic_t val = 20;
    int old = mag_atomic_fetch_sub(&val, 3, MAG_MO_RELAXED);
    EXPECT_EQ(old, 20);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 17);
}

TEST(atomics_ops, FetchAndTest) {
    mag_atomic_t val = 0xF0;
    int old = mag_atomic_fetch_and(&val, 0x0F, MAG_MO_RELAXED);
    EXPECT_EQ(old, 0xF0);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 0x00);
}

TEST(atomics_ops, FetchOrTest) {
    mag_atomic_t val = 0x0F;
    int old = mag_atomic_fetch_or(&val, 0xF0, MAG_MO_RELAXED);
    EXPECT_EQ(old, 0x0F);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 0xFF);
}

TEST(atomics_ops, FetchXorTest) {
    mag_atomic_t val = 0xAA;
    int old = mag_atomic_fetch_xor(&val, 0xFF, MAG_MO_RELAXED);
    EXPECT_EQ(old, 0xAA);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 0x55);
}

TEST(atomics_ops, ExchangeTest) {
    mag_atomic_t val = 100;
    int old = mag_atomic_exchange(&val, 200, MAG_MO_RELAXED);
    EXPECT_EQ(old, 100);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 200);
}

TEST(atomics_ops, CompareExchangeWeakSuccessTest) {
    mag_atomic_t val = 10;
    mag_atomic_t expected = 10;
    mag_atomic_t desired = 20;
    bool success = mag_atomic_compare_exchange_weak(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_TRUE(success);
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 20);
    EXPECT_EQ(expected, 10);
}

TEST(atomics_ops, CompareExchangeWeakFailTest) {
    mag_atomic_t val = 10;
    mag_atomic_t expected = 5;
    mag_atomic_t desired = 20;
    bool success = mag_atomic_compare_exchange_weak(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_FALSE(success);
    EXPECT_EQ(expected, 10);
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 10);
}

TEST(atomics_ops, CompareExchangeStrongSuccessTest) {
    mag_atomic_t val = 30;
    mag_atomic_t expected = 30;
    mag_atomic_t desired = 40;
    bool success = mag_atomic_compare_exchange_strong(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_TRUE(success);
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 40);
    EXPECT_EQ(expected, 30);
}

TEST(atomics_ops, CompareExchangeStrongFailTest) {
    mag_atomic_t val = 30;
    mag_atomic_t expected = 25;
    mag_atomic_t desired = 40;
    bool success = mag_atomic_compare_exchange_strong(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_FALSE(success);
    EXPECT_EQ(expected, 30); // updated to current value of val on failure
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 30);
}
