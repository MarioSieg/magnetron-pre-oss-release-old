// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <gtest/gtest.h>
#include <msml.c>

TEST(core, crc32) {
    ASSERT_EQ(msml__crc32c("Hello, World!", std::strlen("Hello, World!")), 1297420392);
    uint8_t y = 0x3f;
    ASSERT_EQ(msml__crc32c(& y, sizeof(y)), 1015883460);
    ASSERT_EQ(msml__crc32c(nullptr, 0), 0);
    ASSERT_EQ(msml__crc32c("AB", std::strlen("AB")), 3180610794);
    ASSERT_EQ(msml__crc32c(
            "Ich liebe Berliner Kebap, der ist einfach ultra schmackofatz, gerade um 4 Uhr Morgens nach einer langen Clubnacht.",
            std::strlen(
                    "Ich liebe Berliner Kebap, der ist einfach ultra schmackofatz, gerade um 4 Uhr Morgens nach einer langen Clubnacht.")), 60440201);
    std::vector<std::uint8_t> huge {};
    huge.resize(0xffff);
    for (std::size_t i = 0; i < huge.size(); ++i) {
        huge[i] = i % 0xff;
    }
    ASSERT_EQ(msml__crc32c(huge.data(), huge.size()), 2008503331);
}