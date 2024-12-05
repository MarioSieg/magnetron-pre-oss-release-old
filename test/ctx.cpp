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
