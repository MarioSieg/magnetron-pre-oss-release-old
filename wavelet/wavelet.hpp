/*
** (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
** Modern C++ 20 wrapper for the WAVELET C99 API.
*/

#pragma once

#include "wavelet.h"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <span>
#include <type_traits>

namespace wavelet {
    extern "C" [[noreturn]] auto wl__panic(const char* msg, ...) -> void;
    #define wl_cpp_assert(expr, msg, ...) \
        if (!(expr)) [[unlikely]] { \
            ::wavelet::wl__panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
        }
    #define wl_cpp_assert2(expr) wl_cpp_assert(expr, "")

    constexpr std::size_t default_chunk_size {WL_DEFAULT_CHUNK_SIZE};
    constexpr std::size_t default_chunk_cap {WL_DEFAULT_CHUNK_CAP};
    constexpr std::size_t max_dims {WL_MAX_DIMS};
    constexpr std::size_t max_tensor_name_len {WL_MAX_TENSOR_NAME_LEN};
    constexpr std::size_t max_input_tensors {WL_MAX_INPUT_TENSORS};
    constexpr std::size_t max_op_params {WL_MAX_OP_PARAMS};

    [[nodiscard]] constexpr auto version_major(std::uint16_t v) noexcept -> std::uint8_t { return wl_version_major(v); }
    [[nodiscard]] constexpr auto version_minor(std::uint16_t v) noexcept -> std::uint8_t { return wl_version_minor(v); }
    constexpr std::uint16_t version {WL_VERSION};

    enum class exec_mode : std::underlying_type_t<wl_exec_mode_t> {
        eager = WL_EXEC_MODE_EAGER, /* Execute operations immediately. (Dynamic computation graph, like PyTorch). */
        deferred = WL_EXEC_MODE_DEFERRED /* Build computation graph and execute later. (Static computation graph, like TensorFlow 1.0). */
    };

    enum class prng_algorithm : std::underlying_type_t<wl_prng_algorithm_t> {
        mersenne_twister = WL_PRNG_MERSENNE_TWISTER, /* Mersenne Twister PRNG */
        pcg = WL_PRNG_PCG, /* Permuted Congruential Generator PRNG */
    };

    using ctx_info = wl_ctx_info_t;

    class ctx final {
    public:
        ctx() noexcept : m_ctx{wl_ctx_create(nullptr)} {}
        explicit ctx(const ctx_info& info) noexcept : m_ctx{wl_ctx_create(&info)} {}
        explicit ctx(std::size_t pool_size) noexcept : m_ctx{wl_ctx_create2(pool_size)} {}
        ctx(const ctx&) = delete;
        ctx(ctx&& other) noexcept : m_ctx{other.m_ctx} { other.m_ctx = nullptr; }
        ~ctx() {
            if (m_ctx) wl_ctx_destroy(m_ctx);
            m_ctx = nullptr;
        }
        auto operator=(const ctx&) -> ctx& = delete;
        auto operator=(ctx&& other) noexcept -> ctx& {
            if (this != &other) {
                if (m_ctx) wl_ctx_destroy(m_ctx);
                m_ctx = other.m_ctx;
                other.m_ctx = nullptr;
            }
            return *this;
        }
        auto operator*() const noexcept -> wl_ctx_t* { return m_ctx; }
        [[nodiscard]] auto pool_alloc(std::size_t size) noexcept -> void* { return wl_ctx_pool_alloc(m_ctx, size); }
        [[nodiscard]] auto pool_alloc(std::size_t size, std::size_t align) noexcept -> void* { return wl_ctx_pool_alloc_aligned(m_ctx, size, align); }

        template <typename T, typename... Args> requires
            std::is_trivially_destructible_v<T> && std::is_constructible_v<T, Args...>
        [[nodiscard]] auto pool_alloc_obj(Args&&... args) noexcept(std::is_nothrow_constructible_v<T>) -> T* {
            T* obj;
            if constexpr (alignof(T) <= alignof(std::max_align_t) && !(alignof(T) & alignof(T)-1)) obj = static_cast<T*>(pool_alloc(sizeof(T)));
            else obj = static_cast<T*>(pool_alloc(sizeof(T), alignof(T)));
            return std::launder<T>(new(obj) T {std::forward<Args>(args)...});
        }

        [[nodiscard]] auto total_pool_memory() const noexcept -> std::size_t { return wl_ctx_total_allocated_pool_memory(m_ctx); }
        [[nodiscard]] auto exec_mode() const noexcept -> exec_mode { return static_cast<enum exec_mode>(wl_ctx_get_exec_mode(m_ctx)); }
        auto exec_mode(enum exec_mode mode) noexcept -> void { wl_ctx_set_exec_mode(m_ctx, static_cast<wl_exec_mode_t>(mode)); }
        [[nodiscard]] auto prng_algorithm() const noexcept -> prng_algorithm { return static_cast<enum prng_algorithm>(wl_ctx_get_prng_algorithm(m_ctx)); }
        auto prng_algorithm(enum prng_algorithm algo, std::uint64_t seed) noexcept -> void { wl_ctx_set_prng_algorithm(m_ctx, static_cast<wl_prng_algorithm_t>(algo), seed); }
        [[nodiscard]] auto os_name() const -> std::string { return wl_ctx_get_os_name(m_ctx); }
        [[nodiscard]] auto cpu_name() const -> std::string { return wl_ctx_get_cpu_name(m_ctx); }
        [[nodiscard]] auto virtual_cpu_cores() const noexcept -> std::uint32_t { return wl_ctx_get_cpu_virtual_cores(m_ctx); }
        [[nodiscard]] auto physical_cpu_cores() const noexcept -> std::uint32_t { return wl_ctx_get_cpu_physical_cores(m_ctx); }
        [[nodiscard]] auto cpu_sockets() const noexcept -> std::uint32_t { return wl_ctx_get_cpu_sockets(m_ctx); }
        [[nodiscard]] auto physical_mem_total() const noexcept -> std::size_t { return wl_ctx_get_physical_memory_total(m_ctx); }
        [[nodiscard]] auto physical_mem_free() const noexcept -> std::size_t { return wl_ctx_get_physical_memory_free(m_ctx); }
        [[nodiscard]] auto is_numa_system() const noexcept -> bool { return wl_ctx_is_numa_system(m_ctx); }
        [[nodiscard]] auto total_tensors_created() const noexcept -> std::size_t { return wl_ctx_get_total_tensors_created(m_ctx); }
        [[nodiscard]] auto total_tensors_allocated() const noexcept -> std::size_t { return wl_ctx_get_total_tensors_allocated(m_ctx); }
        auto start_profiling() const -> void { wl_ctx_profile_start_recording(m_ctx); };
        auto stop_profiling(const std::string& export_csv_file = "") const -> void { wl_ctx_profile_stop_recording(m_ctx, export_csv_file.empty() ? nullptr : export_csv_file.c_str()); };

    private:
        wl_ctx_t* m_ctx {};
    };

    enum class dtype : std::underlying_type_t<wl_dtype_t> {
        f32 = WL_DTYPE_F32
    };
    using dtype_info = wl_dtype_info_t;
    [[nodiscard]] inline auto dtype_info_of(wl_dtype_t type) noexcept -> const dtype_info& { return * wl_dtype_info_of(type); }

    enum class desired_color_channels : std::underlying_type_t<wl_color_channels_t> {
        automatic = WL_COLOR_CHANNELS_AUTO, /* Automatically determine the number of color channels. */
        grayscale = WL_COLOR_CHANNELS_GRAY, /* Convert to grayscale. */
        grayscale_alpha = WL_COLOR_CHANNELS_GRAY_A, /* Convert to grayscale with alpha channel. */
        rgb = WL_COLOR_CHANNELS_RGB, /* Convert to RGB. */
        rgba = WL_COLOR_CHANNELS_RGBA /* Convert to RGBA. */
    };

    struct op final {
        #define _(enumerator, mnemonic, argcount, paramcount, inplace) enumerator
            enum $ : std::underlying_type_t<wl_op_t> {
                wl_op_def(_, WL_SEP)
                count_ = WL_OP__COUNT
            };
        #undef _
        $ opc;
        constexpr op($ opc) noexcept : opc{opc} {}
        constexpr operator $() const noexcept { return opc; }
        [[nodiscard]] inline auto name() const noexcept -> std::string_view { return wl_op_get_name(static_cast<wl_op_t>(opc)); }
        [[nodiscard]] inline auto mnemonic() const noexcept -> std::string_view { return wl_op_get_mnemonic(static_cast<wl_op_t>(opc)); }
        [[nodiscard]] inline auto paramcount() const noexcept -> std::uint8_t { return wl_op_get_paramcount(static_cast<wl_op_t>(opc)); }
        [[nodiscard]] inline auto argcount() const noexcept -> std::uint8_t { return wl_op_get_argcount(static_cast<wl_op_t>(opc)); }
        [[nodiscard]] inline auto supports_inplace() const noexcept -> bool { return wl_op_get_argcount(static_cast<wl_op_t>(opc)); }
        [[nodiscard]] inline auto is_unary() const noexcept -> bool { return 1 == argcount(); }
        [[nodiscard]] inline auto is_binary() const noexcept -> bool { return 2 == argcount(); }
    };

    enum class param_type : std::underlying_type_t<wl_op_param_type_t> {
        int_param = WL_OP_PARAM_INT,
        float_param = WL_OP_PARAM_FLOAT
    };

    struct op_param final {
    public:
        constexpr op_param() noexcept = default;
        explicit op_param(std::uint64_t x) noexcept : m_param{wl_op_param_int(x)} {}
        explicit op_param(float x) noexcept : m_param{wl_op_param_float(x)} {}
        [[nodiscard]] auto is_int() const noexcept -> bool { return wl_op_param_is_int(m_param); }
        [[nodiscard]] auto unpack_int() const noexcept -> std::uint64_t { return wl_op_param_unpack_int(m_param); }
        [[nodiscard]] auto is_float() const noexcept -> bool { return wl_op_param_is_float(m_param); }
        [[nodiscard]] auto unpack_float() const noexcept -> float{ return wl_op_param_unpack_float(m_param); }

    private:
        wl_op_param_t m_param {};
    };
    static_assert(sizeof(op_param) == sizeof(wl_op_param_t));
    static_assert(alignof(op_param) == alignof(wl_op_param_t));

    enum class graph_eval_order : std::underlying_type_t<wl_graph_eval_order_t> {
        forward = WL_GRAPH_EVAL_ORDER_FORWARD,
        reverse = WL_GRAPH_EVAL_ORDER_REVERSE
    };

    namespace detail {
        template <typename>
        struct dtype_mapper {};

        template <> struct dtype_mapper<float> { static constexpr auto type = dtype::f32; };
    }

    class tensor final {
    public:
        constexpr explicit tensor(wl_tensor_t* t) noexcept : m_t{t} {}

        [[nodiscard]] static auto create(ctx& ctx, dtype type, std::span<const std::int64_t> shape) -> tensor {
            switch (shape.size()) {
                default: wl_cpp_assert2("Invalid tensor shape");
                case 1: return tensor{wl_tensor_create_1d(*ctx, static_cast<wl_dtype_t>(type), shape[0])};
                case 2: return tensor{wl_tensor_create_2d(*ctx, static_cast<wl_dtype_t>(type), shape[0], shape[1])};
                case 3: return tensor{wl_tensor_create_3d(*ctx, static_cast<wl_dtype_t>(type), shape[0], shape[1], shape[2])};
                case 4: return tensor{wl_tensor_create_4d(*ctx, static_cast<wl_dtype_t>(type), shape[0], shape[1], shape[2], shape[3])};
                case 5: return tensor{wl_tensor_create_5d(*ctx, static_cast<wl_dtype_t>(type), shape[0], shape[1], shape[2], shape[3], shape[4])};
                case 6: return tensor{wl_tensor_create_6d(*ctx, static_cast<wl_dtype_t>(type), shape[0], shape[1], shape[2], shape[3], shape[4], shape[5])};
            }
        }

        [[nodiscard]] static auto operation(ctx& ctx, op op, bool inplace, std::span<tensor*> inputs, std::span<const op_param, max_op_params> params) -> tensor { // operator is a reserved keyword in C++
            wl_op_param_t vparams[max_op_params];
            std::copy(params.begin(), params.end(), reinterpret_cast<op_param*>(vparams));
            return tensor{wl_tensor_operator(
                *ctx,
                static_cast<wl_op_t>(op.opc),
                inplace,
                reinterpret_cast<wl_tensor_t**>(inputs.data()),
                static_cast<std::uint32_t>(inputs.size()),
                &vparams
            )};
        }

        auto dtype() const noexcept -> dtype { return static_cast<enum dtype>(wl_tensor_dtype(m_t)); }

        template <typename T>
        auto copy_buffer_from(std::span<const T> buffer) noexcept -> void {
            wl_cpp_assert(detail::dtype_mapper<T>::type == dtype(), "Invalid buffer type");
            wl_tensor_copy_buffer_from(m_t, buffer.data(), buffer.size()*sizeof(T));
        }
        auto fill(const float x) noexcept -> void { wl_tensor_fill(m_t, x); }
        auto fill_random(float min, float max) noexcept -> void { wl_tensor_fill_random_uniform(m_t, min, max); }


    private:
        wl_tensor_t* m_t {};
    };
    static_assert(sizeof(wl_tensor_t*) == sizeof(tensor));
    static_assert(alignof(wl_tensor_t*) == alignof(tensor));
}
