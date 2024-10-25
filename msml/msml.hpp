/*
** (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
** Modern C++ 20 wrapper for the MSML C99 API.
*/

#pragma once

#include "msml.h"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

namespace msml {
    constexpr std::size_t default_chunk_size {MSML_DEFAULT_CHUNK_SIZE};
    constexpr std::size_t default_chunk_cap {MSML_DEFAULT_CHUNK_CAP};
    constexpr std::size_t max_dims {MSML_MAX_DIMS};
    constexpr std::size_t max_tensor_name_len {MSML_MAX_TENSOR_NAME_LEN};
    constexpr std::size_t max_input_tensors {MSML_MAX_INPUT_TENSORS};
    constexpr std::size_t max_op_params {MSML_MAX_OP_PARAMS};

    [[nodiscard]] constexpr auto version_major(std::uint16_t v) noexcept -> std::uint8_t { return msml_version_major(v); }
    [[nodiscard]] constexpr auto version_minor(std::uint16_t v) noexcept -> std::uint8_t { return msml_version_minor(v); }
    constexpr std::uint16_t version {MSML_VERSION};

    enum class exec_mode : std::underlying_type_t<msml_exec_mode_t> {
        eager = MSML_EXEC_MODE_EAGER, /* Execute operations immediately. (Dynamic computation graph, like PyTorch). */
        deferred = MSML_EXEC_MODE_DEFERRED /* Build computation graph and execute later. (Static computation graph, like TensorFlow 1.0). */
    };

    enum class prng_algorithm : std::underlying_type_t<msml_prng_algorithm_t> {
        mersenne_twister = MSML_PRNG_MERSENNE_TWISTER, /* Mersenne Twister PRNG */
        pcg = MSML_PRNG_PCG, /* Permuted Congruential Generator PRNG */
    };

    using ctx_info = msml_ctx_info_t;

    class ctx final {
    public:
        ctx() noexcept : m_ctx{msml_ctx_create(nullptr)} {}
        explicit ctx(const ctx_info& info) noexcept : m_ctx{msml_ctx_create(&info)} {}
        explicit ctx(std::size_t pool_size) noexcept : m_ctx{msml_ctx_create2(pool_size)} {}
        ctx(const ctx&) = delete;
        ctx(ctx&& other) noexcept : m_ctx{other.m_ctx} { other.m_ctx = nullptr; }
        ~ctx() {
            if (m_ctx) msml_ctx_destroy(m_ctx);
            m_ctx = nullptr;
        }
        auto operator=(const ctx&) -> ctx& = delete;
        auto operator=(ctx&& other) noexcept -> ctx& {
            if (this != &other) {
                if (m_ctx) msml_ctx_destroy(m_ctx);
                m_ctx = other.m_ctx;
                other.m_ctx = nullptr;
            }
            return *this;
        }
        auto operator*() const noexcept -> msml_ctx_t* { return m_ctx; }
        [[nodiscard]] auto pool_alloc(std::size_t size) noexcept -> void* { return msml_ctx_pool_alloc(m_ctx, size); }
        [[nodiscard]] auto pool_alloc(std::size_t size, std::size_t align) noexcept -> void* { return msml_ctx_pool_alloc_aligned(m_ctx, size, align); }

        template <typename T, typename... Args> requires
            std::is_trivially_destructible_v<T> && std::is_constructible_v<T, Args...>
        [[nodiscard]] auto pool_alloc_obj(Args&&... args) noexcept(std::is_nothrow_constructible_v<T>) -> T* {
            T* obj;
            if constexpr (alignof(T) <= alignof(std::max_align_t) && !(alignof(T) & (alignof(T)-1))) {
                obj = static_cast<T*>(pool_alloc(sizeof(T)));
            } else {
                obj = static_cast<T*>(pool_alloc(sizeof(T), alignof(T)));
            }
            return std::launder<T>(new(obj) T {std::forward<Args>(args)...});
        }

        [[nodiscard]] auto total_pool_memory() const noexcept -> std::size_t { return msml_ctx_total_allocated_pool_memory(m_ctx); }
        [[nodiscard]] auto exec_mode() const noexcept -> exec_mode { return static_cast<enum exec_mode>(msml_ctx_get_exec_mode(m_ctx)); }
        auto exec_mode(enum exec_mode mode) noexcept -> void { msml_ctx_set_exec_mode(m_ctx, static_cast<msml_exec_mode_t>(mode)); }
        [[nodiscard]] auto prng_algorithm() const noexcept -> prng_algorithm { return static_cast<enum prng_algorithm>(msml_ctx_get_prng_algorithm(m_ctx)); }
        auto prng_algorithm(enum prng_algorithm algo, std::uint64_t seed) noexcept -> void { msml_ctx_set_prng_algorithm(m_ctx, static_cast<msml_prng_algorithm_t>(algo), seed); }
        [[nodiscard]] auto os_name() const -> std::string { return msml_ctx_get_os_name(m_ctx); }
        [[nodiscard]] auto cpu_name() const -> std::string { return msml_ctx_get_cpu_name(m_ctx); }
        [[nodiscard]] auto virtual_cpu_cores() const noexcept -> std::uint32_t { return msml_ctx_get_cpu_virtual_cores(m_ctx); }
        [[nodiscard]] auto physical_cpu_cores() const noexcept -> std::uint32_t { return msml_ctx_get_cpu_physical_cores(m_ctx); }
        [[nodiscard]] auto cpu_sockets() const noexcept -> std::uint32_t { return msml_ctx_get_cpu_sockets(m_ctx); }
        [[nodiscard]] auto physical_mem_total() const noexcept -> std::size_t { return msml_ctx_get_physical_memory_total(m_ctx); }
        [[nodiscard]] auto physical_mem_free() const noexcept -> std::size_t { return msml_ctx_get_physical_memory_free(m_ctx); }
        [[nodiscard]] auto is_numa_system() const noexcept -> bool { return msml_ctx_is_numa_system(m_ctx); }

    private:
        msml_ctx_t* m_ctx {};
    };

    enum class dtype : std::underlying_type_t<msml_dtype_t> {
        f32 = MSML_DTYPE_F32
    };
    using dtype_info = msml_dtype_info_t;
    [[nodiscard]] inline auto dtype_info_of(msml_dtype_t type) noexcept -> const dtype_info& { return * msml_dtype_info_of(type); }

    enum class desired_color_channels : std::underlying_type_t<msml_color_channels_t> {
        automatic = MSML_COLOR_CHANNELS_AUTO, /* Automatically determine the number of color channels. */
        grayscale = MSML_COLOR_CHANNELS_GRAY, /* Convert to grayscale. */
        grayscale_alpha = MSML_COLOR_CHANNELS_GRAY_A, /* Convert to grayscale with alpha channel. */
        rgb = MSML_COLOR_CHANNELS_RGB, /* Convert to RGB. */
        rgba = MSML_COLOR_CHANNELS_RGBA /* Convert to RGBA. */
    };

    struct op final {
        #define _(enumerator, mnemonic, argcount) enumerator
            enum $ : std::underlying_type_t<msml_op_t> {
                msml_op_def(_, MSML_SEP)
                count_ = MSML_OP__COUNT
            };
        #undef _
        $ value;
        constexpr op($ value) noexcept : value{value} {}
        constexpr operator $() const noexcept { return value; }
        [[nodiscard]] inline auto name() const noexcept -> std::string_view { return msml_op_get_name(static_cast<msml_op_t>(value)); }
        [[nodiscard]] inline auto mnemonic() const noexcept -> std::string_view { return msml_op_get_mnemonic(static_cast<msml_op_t>(value)); }
        [[nodiscard]] inline auto argcount() const noexcept -> std::uint8_t { return msml_op_get_argcount(static_cast<msml_op_t>(value)); }
        [[nodiscard]] inline auto is_unary() const noexcept -> bool { return 1 == argcount(); }
        [[nodiscard]] inline auto is_binary() const noexcept -> bool { return 2 == argcount(); }
    };

    enum class param_type : std::underlying_type_t<msml_op_param_type_t> {
        float_param = MSML_OP_PARAM_FLOAT,
        int_param = MSML_OP_PARAM_INT,
    };

    struct op_param final {
    public:
        constexpr op_param() noexcept = default;
        constexpr op_param(std::uint64_t x) noexcept : m_param{msml_op_param_int(x)} {}
        [[nodiscard]] auto is_int() const noexcept -> bool { return msml_op_param_is_int(m_param); }
        [[nodiscard]] auto unpack_int() const noexcept -> std::uint64_t { return msml_op_param_unpack_int(m_param); }

    private:
        msml_op_param_t m_param {};
    };
    static_assert(sizeof(op_param) == sizeof(msml_op_param_t));

    enum class graph_eval_order : std::underlying_type_t<msml_graph_eval_order_t> {
        forward = MSML_GRAPH_EVAL_ORDER_FORWARD,
        reverse = MSML_GRAPH_EVAL_ORDER_REVERSE
    };
}
