// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(ctx, create_destroy) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    ASSERT_NE(ctx, nullptr);
    wl_ctx_destroy(ctx);
}

#ifndef _MSC_VER // MSVC fucks around with linking a __declspex(dllexport) ed function ptr. TODO: fix
TEST(ctx, alloc) {
    int* i = static_cast<int*>((*wl_alloc)(nullptr, sizeof(int)));
    ASSERT_NE(i, nullptr);
    *i = -1;
    ASSERT_EQ(*i, -1);
    (*wl_alloc)(i, 0);
}
#endif

TEST(ctx, alloc_aligned) {
    int* i = static_cast<int*>(wl_alloc_aligned(sizeof(int), 512));
    ASSERT_NE(i, nullptr);
    *i = -1;
    ASSERT_EQ(*i, -1);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(i) % 512, 0);
    wl_free_aligned(i);
}

TEST(atomics_ops, StoreLoadTest) {
    wl_atomic_t val = 0;
    wl_atomic_store(&val, 42, WL_MO_RELAXED);
    int loaded = wl_atomic_load(&val, WL_MO_RELAXED);
    EXPECT_EQ(loaded, 42);
}

TEST(atomics_ops, FetchAddTest) {
    wl_atomic_t val = 10;
    int old = wl_atomic_fetch_add(&val, 5, WL_MO_RELAXED);
    EXPECT_EQ(old, 10);
    int new_val = wl_atomic_load(&val, WL_MO_RELAXED);
    EXPECT_EQ(new_val, 15);
}

TEST(atomics_ops, FetchSubTest) {
    wl_atomic_t val = 20;
    int old = wl_atomic_fetch_sub(&val, 3, WL_MO_RELAXED);
    EXPECT_EQ(old, 20);
    int new_val = wl_atomic_load(&val, WL_MO_RELAXED);
    EXPECT_EQ(new_val, 17);
}

TEST(atomics_ops, FetchAndTest) {
    wl_atomic_t val = 0xF0;
    int old = wl_atomic_fetch_and(&val, 0x0F, WL_MO_RELAXED);
    EXPECT_EQ(old, 0xF0);
    int new_val = wl_atomic_load(&val, WL_MO_RELAXED);
    EXPECT_EQ(new_val, 0x00);
}

TEST(atomics_ops, FetchOrTest) {
    wl_atomic_t val = 0x0F;
    int old = wl_atomic_fetch_or(&val, 0xF0, WL_MO_RELAXED);
    EXPECT_EQ(old, 0x0F);
    int new_val = wl_atomic_load(&val, WL_MO_RELAXED);
    EXPECT_EQ(new_val, 0xFF);
}

TEST(atomics_ops, FetchXorTest) {
    wl_atomic_t val = 0xAA;
    int old = wl_atomic_fetch_xor(&val, 0xFF, WL_MO_RELAXED);
    EXPECT_EQ(old, 0xAA);
    int new_val = wl_atomic_load(&val, WL_MO_RELAXED);
    EXPECT_EQ(new_val, 0x55);
}

TEST(atomics_ops, ExchangeTest) {
    wl_atomic_t val = 100;
    int old = wl_atomic_exchange(&val, 200, WL_MO_RELAXED);
    EXPECT_EQ(old, 100);
    int new_val = wl_atomic_load(&val, WL_MO_RELAXED);
    EXPECT_EQ(new_val, 200);
}

TEST(atomics_ops, CompareExchangeWeakSuccessTest) {
    wl_atomic_t val = 10;
    wl_atomic_t expected = 10;
    wl_atomic_t desired = 20;
    bool success = wl_atomic_compare_exchange_weak(&val, &expected, &desired, WL_MO_RELAXED, WL_MO_RELAXED);
    EXPECT_TRUE(success);
    EXPECT_EQ(wl_atomic_load(&val, WL_MO_RELAXED), 20);
    EXPECT_EQ(expected, 10);
}

TEST(atomics_ops, CompareExchangeWeakFailTest) {
    wl_atomic_t val = 10;
    wl_atomic_t expected = 5;
    wl_atomic_t desired = 20;
    bool success = wl_atomic_compare_exchange_weak(&val, &expected, &desired, WL_MO_RELAXED, WL_MO_RELAXED);
    EXPECT_FALSE(success);
    EXPECT_EQ(expected, 10);
    EXPECT_EQ(wl_atomic_load(&val, WL_MO_RELAXED), 10);
}

TEST(atomics_ops, CompareExchangeStrongSuccessTest) {
    wl_atomic_t val = 30;
    wl_atomic_t expected = 30;
    wl_atomic_t desired = 40;
    bool success = wl_atomic_compare_exchange_strong(&val, &expected, &desired, WL_MO_RELAXED, WL_MO_RELAXED);
    EXPECT_TRUE(success);
    EXPECT_EQ(wl_atomic_load(&val, WL_MO_RELAXED), 40);
    EXPECT_EQ(expected, 30);
}

TEST(atomics_ops, CompareExchangeStrongFailTest) {
    wl_atomic_t val = 30;
    wl_atomic_t expected = 25;
    wl_atomic_t desired = 40;
    bool success = wl_atomic_compare_exchange_strong(&val, &expected, &desired, WL_MO_RELAXED, WL_MO_RELAXED);
    EXPECT_FALSE(success);
    EXPECT_EQ(expected, 30); // updated to current value of val on failure
    EXPECT_EQ(wl_atomic_load(&val, WL_MO_RELAXED), 30);
}
