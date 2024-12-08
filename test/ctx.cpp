// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(ctx, create_destroy) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    ASSERT_NE(ctx, nullptr);
    wl_ctx_destroy(ctx);
}

#ifndef _MSC_VER // MSVC fucks around with linking a __declspex(dllexport) ed function ptr. TODO: fix
TEST(ctx, alloc) {
    int* i = static_cast<int*>((*wl__alloc)(nullptr, sizeof(int)));
    ASSERT_NE(i, nullptr);
    *i = -1;
    ASSERT_EQ(*i, -1);
    (*wl__alloc)(i, 0);
}
#endif

TEST(ctx, alloc_aligned) {
    int* i = static_cast<int*>(wl__alloc_aligned(sizeof(int), 512));
    ASSERT_NE(i, nullptr);
    *i = -1;
    ASSERT_EQ(*i, -1);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(i) % 512, 0);
    wl__free_aligned(i);
}

TEST(atomics_ops, StoreLoadTest) {
    wl__atomic_t val = 0;
    neo_atomic_store(&val, 42, WL__MO_RELAXED);
    int loaded = neo_atomic_load(&val, WL__MO_RELAXED);
    EXPECT_EQ(loaded, 42);
}

TEST(atomics_ops, FetchAddTest) {
    wl__atomic_t val = 10;
    int old = neo_atomic_fetch_add(&val, 5, WL__MO_RELAXED);
    EXPECT_EQ(old, 10);
    int new_val = neo_atomic_load(&val, WL__MO_RELAXED);
    EXPECT_EQ(new_val, 15);
}

TEST(atomics_ops, FetchSubTest) {
    wl__atomic_t val = 20;
    int old = neo_atomic_fetch_sub(&val, 3, WL__MO_RELAXED);
    EXPECT_EQ(old, 20);
    int new_val = neo_atomic_load(&val, WL__MO_RELAXED);
    EXPECT_EQ(new_val, 17);
}

TEST(atomics_ops, FetchAndTest) {
    wl__atomic_t val = 0xF0;
    int old = neo_atomic_fetch_and(&val, 0x0F, WL__MO_RELAXED);
    EXPECT_EQ(old, 0xF0);
    int new_val = neo_atomic_load(&val, WL__MO_RELAXED);
    EXPECT_EQ(new_val, 0x00);
}

TEST(atomics_ops, FetchOrTest) {
    wl__atomic_t val = 0x0F;
    int old = neo_atomic_fetch_or(&val, 0xF0, WL__MO_RELAXED);
    EXPECT_EQ(old, 0x0F);
    int new_val = neo_atomic_load(&val, WL__MO_RELAXED);
    EXPECT_EQ(new_val, 0xFF);
}

TEST(atomics_ops, FetchXorTest) {
    wl__atomic_t val = 0xAA;
    int old = neo_atomic_fetch_xor(&val, 0xFF, WL__MO_RELAXED);
    EXPECT_EQ(old, 0xAA);
    int new_val = neo_atomic_load(&val, WL__MO_RELAXED);
    EXPECT_EQ(new_val, 0x55);
}

TEST(atomics_ops, ExchangeTest) {
    wl__atomic_t val = 100;
    int old = neo_atomic_exchange(&val, 200, WL__MO_RELAXED);
    EXPECT_EQ(old, 100);
    int new_val = neo_atomic_load(&val, WL__MO_RELAXED);
    EXPECT_EQ(new_val, 200);
}

TEST(atomics_ops, CompareExchangeWeakSuccessTest) {
    wl__atomic_t val = 10;
    int expected = 10;
    int desired = 20;
    bool success = neo_atomic_compare_exchange_weak(&val, &expected, &desired, WL__MO_RELAXED, WL__MO_RELAXED);
    EXPECT_TRUE(success);
    EXPECT_EQ(neo_atomic_load(&val, WL__MO_RELAXED), 20);
    EXPECT_EQ(expected, 10);
}

TEST(atomics_ops, CompareExchangeWeakFailTest) {
    wl__atomic_t val = 10;
    int expected = 5;
    int desired = 20;
    bool success = neo_atomic_compare_exchange_weak(&val, &expected, &desired, WL__MO_RELAXED, WL__MO_RELAXED);
    EXPECT_FALSE(success);
    EXPECT_EQ(expected, 10);
    EXPECT_EQ(neo_atomic_load(&val, WL__MO_RELAXED), 10);
}

TEST(atomics_ops, CompareExchangeStrongSuccessTest) {
    wl__atomic_t val = 30;
    int expected = 30;
    int desired = 40;
    bool success = neo_atomic_compare_exchange_strong(&val, &expected, &desired, WL__MO_RELAXED, WL__MO_RELAXED);
    EXPECT_TRUE(success);
    EXPECT_EQ(neo_atomic_load(&val, WL__MO_RELAXED), 40);
    EXPECT_EQ(expected, 30);
}

TEST(atomics_ops, CompareExchangeStrongFailTest) {
    wl__atomic_t val = 30;
    int expected = 25;
    int desired = 40;
    bool success = neo_atomic_compare_exchange_strong(&val, &expected, &desired, WL__MO_RELAXED, WL__MO_RELAXED);
    EXPECT_FALSE(success);
    EXPECT_EQ(expected, 30); // updated to current value of val on failure
    EXPECT_EQ(neo_atomic_load(&val, WL__MO_RELAXED), 30);
}
