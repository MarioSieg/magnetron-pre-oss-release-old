/*
** (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
** MSML - Single file STB-style machine learning library in C99 with Python bindings.
** For license see LICENSE file.
*/

#define MSML_EXPORT_DLL
#include "msml.h"

#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <float.h>

#ifdef _MSC_VER
#   include <intrin.h>
#endif

msml_static_assert(sizeof(0u) == 4);
msml_static_assert(sizeof(0ull) == 8);

#ifdef MSML_ENABLE_IMAGE_SUPPORT
#   define STBI_MALLOC(sz) msml_allocator(NULL, (sz))
#   define STBI_FREE(ptr) msml_allocator((ptr), 0)
#   define STBI_REALLOC(ptr, sz) msml_allocator((ptr), (sz))
#   define STBIR_MALLOC(sz, usr) msml_allocator(NULL, (sz))
#   define STBIR_FREE(ptr, usr) msml_allocator((ptr), 0)
#   define STBIR_REALLOC(ptr, sz, usr) msml_allocator((ptr), (sz))
#   define STBIW_MALLOC(sz) msml_allocator(NULL, (sz))
#   define STBIW_FREE(ptr) msml_allocator((ptr), 0)
#   define STBIW_REALLOC(ptr, sz) msml_allocator((ptr), (sz))
#   define STB_IMAGE_IMPLEMENTATION
#   include <stb_image.h>
#   define STB_IMAGE_RESIZE_IMPLEMENTATION
#   include <stb_image_resize2.h>
#   define STB_IMAGE_WRITE_IMPLEMENTATION
#   include <stb_image_write.h>
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#	define MSML_NORET __attribute__((noreturn))
#	define MSML_ALIGN(x) __attribute__((aligned(x)))
#	define MSML_AINLINE inline __attribute__((always_inline))
#	define MSML_NOINLINE __attribute__((noinline))
#   define MSML_HOTPROC __attribute__((hot))
#   define MSML_COLDPROC __attribute__((cold))
#   define MSML_PACKED __attribute__((packed))
#   define MSML_FALLTHROUGH __attribute__((fallthrough))
#   define MSML_UNUSED __attribute__((unused))
#	define msml_likely(x) __builtin_expect(!!(x), 1)
#	define msml_unlikely(x) __builtin_expect(!!(x), 0)
#   define msml_ffs(x) ((uint32_t)__builtin_ctz(x))
#   define msml_fls(x) ((uint32_t)(__builtin_clz(x)^31))
#   define msml_ffs64(x) ((uint32_t)__builtin_ctzll(x))
#   define msml_fls64(x) ((uint32_t)(__builtin_clzll(x)^63))
#else
    unsigned char _BitScanForward64(unsigned long*, unsigned __int64);
    unsigned char _BitScanReverse64(unsigned long*, unsigned __int64);
#   pragma intrinsic(_BitScanForward64)
#   pragma intrinsic(_BitScanReverse64)
#	define MSML_NORET __declspec(noreturn)
#	define MSML_ALIGN(x) __declspec(align(x))
#	define MSML_AINLINE inline __forceinline
#	define MSML_NOINLINE __declspec(noinline)
#   define MSML_HOTPROC
#   define MSML_COLDPROC
#   define MSML_PACKED __declspec(align(1))
#   define MSML_FALLTHROUGH
#   define MSML_UNUSED
#	define msml_likely(x) (x)
#	define msml_unlikely(x) (x)
    static __forceinline uint32_t msml_ffs(const uint32_t x) {
        unsigned long r; _BitScanForward(&r, x); return (uint32_t)r;
    }
    static __forceinline uint32_t msml_fls(const uint32_t x) {
        unsigned long r; _BitScanReverse(&r, x); return (uint32_t)r;
    }
    static __forceinline uint32_t msml_ffs64(const uint64_t x) {
      unsigned long r; _BitScanForward64(&r, x); return (uint32_t)r;
    }
    static __forceinline uint32_t msml_fls64(const uint64_t x) {
      unsigned long r; _BitScanReverse64(&r, x); return (uint32_t)r;
    }
#endif

#define msml_max(x, y) (((x) > (y)) ? (x) : (y))
#define msml_min(x, y) (((x) < (y)) ? (x) : (y))
#define MSML_CCRED "\x1b[31m"
#define MSML_CCGREEN "\x1b[32m"
#define MSML_CCYELLOW "\x1b[33m"
#define MSML_CCBLUE "\x1b[34m"
#define MSML_CCMAGENTA "\x1b[35m"
#define MSML_CCCYAN "\x1b[36m"
#define MSML_CCRESET "\x1b[0m"
#define MSML_STRINGIZE(x) MSML_STRINGIZE2(x)
#define MSML_STRINGIZE2(x) #x
#ifdef _MSC_VER
#   define MSML_SRC_NAME __FILE__ ":" MSML_STRINGIZE(__LINE__)
#else
#   define MSML_SRC_NAME __FILE_NAME__ ":" MSML_STRINGIZE(__LINE__)
#endif
#define msml_log_info(msg, ...) fprintf(stdout,  "[MSML] " MSML_SRC_NAME " " msg "\n", ## __VA_ARGS__)
#define msml_log_warn(msg, ...) fprintf(stderr,  "[MSML] " MSML_SRC_NAME " " MSML_CCYELLOW msg MSML_CCRESET "\n", ## __VA_ARGS__)

struct msml_ctx_t {
    void* (*alloc_fn)(void* blk, size_t size);
    size_t chunk_size;
    size_t chunk_len;
    size_t chunk_cap;
    uint8_t** chunks;
    uint8_t* delta;
    bool warmup_chunks;
    size_t alloc_acc;
    size_t mapped_total;
    size_t alloc_total;
    union {
        struct {
            uint64_t state;
            uint64_t inc;
        } pcg;
        struct {
            uint32_t remaining;
            uint32_t next;
            uint32_t state[624];
        } mersenne;
    } prng_state;
    msml_prng_algorithm_t prng_algorithm;
    uintptr_t host_thread_id;
    void* user_data;
};

struct msml_tensor_t {
    msml_ctx_t* ctx;
    int64_t rank;
    int64_t shape[MSML_MAX_DIMS];
    int64_t strides[MSML_MAX_DIMS];
    msml_dtype_t dtype;
    void* buf;
    int64_t buf_size;
    msml_op_t op;
    msml_tensor_t* args[MSML_MAX_ARG_TENSORS];
    msml_tensor_t* slice;
    size_t slice_offset;
    char name[MSML_MAX_TENSOR_NAME_LEN];
    void* user_data;
};

static MSML_NORET void msml_panic(const char* msg, ...) {
    fprintf(stderr, "%s", MSML_CCRED);
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
    fprintf(stderr, "%s", MSML_CCRESET);
    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

#define msml_assert(expr, msg, ...) \
    if (msml_unlikely(!(expr))) { \
        msml_panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
    }
#define msml_assert2(expr) msml_assert(expr, "")

void* msml_default_allocator_impl(void* blk, size_t size) {
    if (!size) {
        free(blk);
        return NULL;
    } else if(!blk) {
        blk = malloc(size);
        msml_assert(blk, "Failed to allocate %.03fKiB memory", (double)size/(double)(1<<10));
        return blk;
    } else {
        void* block = realloc(blk, size);
        msml_assert(blk, "Failed to reallocate %.03fKiB memory", (double)size/(double)(1<<10));
        return block;
    }
}

static void msml__humanize_memory_size(size_t n, double* out, const char** unit) {
    if (n < (1<<10)) {
        *out = (double)n;
        *unit = "B";
    } else if (n < (1<<20)) {
        *out = (double)n/(double)(1<<10);
        *unit = "KiB";
    } else if (n < (1<<30)) {
        *out = (double)n/(double)(1<<20);
        *unit = "MiB";
    } else {
        *out = (double)n/(double)(1<<30);
        *unit = "GiB";
    }
}

#ifdef _WIN32
#include <wchar.h>
extern __declspec(dllimport) int __stdcall MultiByteToWideChar(
    unsigned int cp,
    unsigned long flags,
    const char* str,
    int cbmb,
    wchar_t* widestr,
    int cchwide
);
extern __declspec(dllimport) int __stdcall WideCharToMultiByte(
    unsigned int cp,
    unsigned long flags,
    const wchar_t* widestr,
    int cchwide,
    char* str,
    int cbmb,
    const char* defchar,
    int* used_default
);
#endif

static FILE* msml__fopen(const char* file, const char* mode) {
    msml_assert(file && *file && mode && *mode, "Invalid file name or mode");
    FILE* f = NULL;
    #ifdef _WIN32
        wchar_t w_mode[64];
        wchar_t w_file[1024];
        if (MultiByteToWideChar(65001 /* UTF8 */, 0, file, -1, w_file, sizeof(w_file)/sizeof(*w_file)) == 0) return NULL;
        if (MultiByteToWideChar(65001 /* UTF8 */, 0, mode, -1, w_mode, sizeof(w_mode)/sizeof(*w_mode)) == 0) return NULL;
        #if defined(_MSC_VER) && _MSC_VER >= 1400
           if (_wfopen_s(&f, w_file, w_mode) != 0)
               return NULL;
        #else
           f = _wfopen(w_file, w_mode);
        #endif
    #elif defined(_MSC_VER) && _MSC_VER >= 1400
        if (fopen_s(&f, filename, mode) != 0) return NULL;
    #else
        f = fopen(file, mode);
    #endif
    return f;
}

static inline uintptr_t msml__thread_id(void) {
    uintptr_t tid;
    #if defined(_MSC_VER) && defined(_M_X64)
        tid = __readgsqword(48);
    #elif defined(_MSC_VER) && defined(_M_IX86)
        tid = __readfsdword(24);
    #elif defined(_MSC_VER) && defined(_M_ARM64)
        tid = __getReg(18);
    #elif defined(__i386__)
        __asm__ __volatile__("movl %%gs:0, %0" : "=r" (tid));  /* x86-32 WIN32 uses %GS */
    #elif defined(__MACH__) && defined(__x86_64__)
        __asm__ __volatile__("movq %%gs:0, %0" : "=r" (tid));  /* x86.64 OSX uses %GS */
    #elif defined(__x86_64__)
        __asm__ __volatile__("movq %%fs:0, %0" : "=r" (tid));  /* x86-64 Linux and BSD uses %FS */
    #elif defined(__arm__)
        __asm__ __volatile__("mrc p15, 0, %0, c13, c0, 3\nbic %0, %0, #3" : "=r" (tid));
    #elif defined(__aarch64__) && defined(__APPLE__)
        __asm__ __volatile__("mrs %0, tpidrro_el0" : "=r" (tid));
    #elif defined(__aarch64__)
        __asm__ __volatile__("mrs %0, tpidr_el0" : "=r" (tid));
    #elif defined(__powerpc64__)
    #   ifdef __clang__
            tid = (uintptr_t)__builtin_thread_pointer();
    #   else
            register uintptr_t tp __asm__ ("r13");
            __asm__ __volatile__("" : "=r" (tp));
            tid = tp;
    #   endif
    #elif defined(__powerpc__)
    #   ifdef __clang__
                tid = (uintptr_t)__builtin_thread_pointer();
    #   else
                register uintptr_t tp __asm__ ("r2");
                __asm__ __volatile__("" : "=r" (tp));
                tid = tp;
    #   endif
    #elif defined(__s390__) && defined(__GNUC__)
        tid = (uintptr_t)__builtin_thread_pointer();
    #elif defined(__riscv)
    #   ifdef __clang__
            tid = (uintptr_t)__builtin_thread_pointer();
    #   else
            __asm__ ("mv %0, tp" : "=r" (tid));
    #   endif
    #else
    #   error "Unsupported MSML platform"
    #endif
    return tid;
}

static void MSML_AINLINE msml__bswap32(uint32_t* p_x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
    (void)p_x;
#if defined(__AARCH64EB__) || __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    uint32_t x = *p_x;
    x = (x & 0xff000000) >> 24 |
    (x & 0xff0000) >> 8 |
    (x & 0xff00) << 8 |
    (x & 0xff) << 24;
    *p_x = x;
#endif
}

static void MSML_AINLINE msml__bswap64(uint64_t* p_x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
    (void)p_x;
    #if defined(__AARCH64EB__) || __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        uint64_t x = *p_x;
        x = (x & 0xff00000000000000) >> 56 |
        (x & 0xff000000000000) >> 40 |
        (x & 0xff0000000000) >> 24 |
        (x & 0xff00000000) >> 8 |
        (x & 0xff000000) << 8 |
        (x & 0xff0000) << 24 |
        (x & 0xff00) << 40 |
        (x & 0xff) << 56;
        *p_x = x;
    #endif
}

static uint32_t msml__crc32(const void* buf, size_t size) { /* Compute CRC32 checksum. */
    const uint8_t* buffer = buf;
    static const uint32_t crc_lut[256] = {
        0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
        0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988, 0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
        0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
        0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
        0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172, 0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
        0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
        0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
        0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924, 0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
        0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
        0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
        0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e, 0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
        0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
        0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
        0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0, 0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
        0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
        0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
        0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a, 0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
        0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
        0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
        0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc, 0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
        0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
        0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
        0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236, 0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
        0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
        0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
        0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38, 0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
        0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
        0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
        0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2, 0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
        0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
        0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693, 0x54de5729, 0x23d967bf,
        0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94, 0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
    };
    uint32_t crc = ~0u;
    for (size_t i=0; i < size; ++i)
        crc = (crc >> 8) ^ crc_lut[buffer[i] ^ (crc & 0xff)];
    return ~crc;
}

static bool MSML_AINLINE msml__imull64_ov(int64_t a, int64_t b, int64_t* out) { /* Performs c = a*b with overflow checking. Returns true on overflow, else false. */
#ifdef _MSC_VER
    msml_panic("NYI"); // TODO - maybe MSVC intrinsic available?
#else
#if __SIZEOF_LONG_LONG__ == 8
    return __builtin_smulll_overflow(a, b, out);
#else
    return __builtin_smull_overflow(a, b, out);
#endif
#endif
}

/* Generate n uniform random floats within [min, max]. */
static void msml__prng_generate_n(msml_ctx_t* ctx, float* out_gen, int64_t out_n, float min, float max) {
    float rescale_uniform = max - min;
    switch (ctx->prng_algorithm) {
        case MSML_PRNG_MERSENNE_TWISTER: {
            uint32_t* rem = &ctx->prng_state.mersenne.remaining;
            uint32_t* next = &ctx->prng_state.mersenne.next;
            uint32_t* state = ctx->prng_state.mersenne.state;
            for (int64_t ii=0; ii < out_n; ++ii) {
                if (--*rem <= 0) {
                    *rem = 624;
                    *next = 0;
                    uint32_t y, i;
                    for (i = 0; i < 624-397; ++i) {
                        y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                        state[i] = state[i+397] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                    }
                    for (; i < 624-1; ++i) {
                        y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                        state[i] = state[i + (397-624)] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                    }
                    y = (state[624-1] & 0x80000000u) | (*state & 0x7fffffffu);
                    state[624-1] = state[397-1] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                }
                uint32_t y = state[(*next)++];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680;
                y ^= (y << 15) & 0xefc60000;
                y ^= y >> 18;
                out_gen[ii] = min + rescale_uniform * (1.f/(float)(1<<23)*((float)(y>>9) + 0.5f));
            }
        } break;
        case MSML_PRNG_PCG: {
            uint64_t* state = &ctx->prng_state.pcg.state;
            uint64_t* inc = &ctx->prng_state.pcg.inc;
            for (int64_t ii=0; ii < out_n; ++ii) {
                uint64_t prev = *state;
                *state = prev*6364136223846793005ull + *inc;
                uint32_t mixed = ((prev>>18u) ^ prev) >> 27u;
                uint32_t rot = prev >> 59u;
                uint32_t y = (mixed>>rot) | (mixed << ((-rot)&31));
                out_gen[ii] = min + rescale_uniform * (1.f/(float)(1<<23)*((float)(y>>9) + 0.5f));
            }
        } break;
        default:
            msml_panic("Unknown PRNG algorithm: %d", ctx->prng_algorithm);
    }
}

static void msml__prng_init(msml_ctx_t* ctx, uint64_t seed) {
    seed = seed ? seed : 0x853c49e6748fea9bull ^ (uintptr_t)ctx ^ (uintptr_t)&ctx; /* Default seed. */
    switch (ctx->prng_algorithm) {
        case MSML_PRNG_MERSENNE_TWISTER: {
            uint32_t* state = ctx->prng_state.mersenne.state;
            *state = (uint32_t)seed;
            for (size_t i=1; i < 624; ++i)
                state[i] = ((state[i-1] ^ (state[i-1] >> 30))*1812433253 + i) & ~0u;
            ctx->prng_state.mersenne.next = 0;
            ctx->prng_state.mersenne.remaining = 1;
        } break;
        case MSML_PRNG_PCG: {
            ctx->prng_state.pcg.state = seed ^ 0x853c49e6748fea9bull;
            ctx->prng_state.pcg.inc = 0xda3e39cb94b95bdbull;
        } break;
        default:
            msml_panic("Unknown PRNG algorithm: %d", ctx->prng_algorithm);
    }
}

static void msml__ctx_push_chunk(msml_ctx_t* ctx) {
    uint8_t* chunk = (*ctx->alloc_fn)(NULL, ctx->chunk_size);
    if (ctx->warmup_chunks) memset(chunk, 0, ctx->chunk_size);
    ctx->mapped_total += ctx->chunk_size;
    ctx->delta = chunk + ctx->chunk_size;
    if (ctx->chunk_len == ctx->chunk_cap)
        ctx->chunks = (*ctx->alloc_fn)(ctx->chunks, (ctx->chunk_cap<<=1) * sizeof(*ctx->chunks));
    ctx->chunks[ctx->chunk_len++] = chunk;
}

msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info) {
    printf("Creating MSML context...\n");

    msml_ctx_info_t ctx_info = {0};
    if (info) ctx_info = *info;
    ctx_info.alloc_fn = ctx_info.alloc_fn ? ctx_info.alloc_fn : &msml_allocator;

    msml_ctx_t* ctx = (*ctx_info.alloc_fn)(NULL, sizeof(*ctx));
    memset(ctx, 0, sizeof(*ctx));
    ctx->alloc_fn = ctx_info.alloc_fn;
    ctx->user_data = ctx_info.user_data;
    ctx->chunk_size = ctx_info.pool_chunk_size ? msml_max(ctx_info.pool_chunk_size, 8) : MSML_DEFAULT_CHUNK_SIZE;
    ctx->chunk_cap = ctx_info.pool_chunks_cap ? msml_max(ctx_info.pool_chunks_cap, 1) : MSML_DEFAULT_CHUNK_CAP;
    ctx->warmup_chunks = ctx_info.warmup_chunks;

    ctx->chunks = (uint8_t**)(*ctx->alloc_fn)(NULL, ctx->chunk_cap * sizeof(*ctx->chunks));
    msml__ctx_push_chunk(ctx);
    uint64_t host_tid = msml__thread_id();
    ctx->prng_algorithm = ctx_info.prng_algorithm;
    msml__prng_init(ctx, ctx_info.prng_seed^host_tid^(uintptr_t)ctx^(uintptr_t)&ctx_info);
    ctx->host_thread_id = host_tid;
    printf("MSML context created.\n");
    return ctx;
}

msml_ctx_t* msml_ctx_create2(size_t pool_chunk_size) {
    msml_ctx_info_t info = {0};
    info.pool_chunk_size = pool_chunk_size;
    return msml_ctx_create(&info);
}

void* msml_ctx_pool_alloc(msml_ctx_t* ctx, size_t size) {
    msml_assert(size > 0 && size < (size_t)PTRDIFF_MAX, "Allocation size must be within (0, %zu), but is: %zu", PTRDIFF_MAX, size);
    if (ctx->delta - ctx->chunks[ctx->chunk_len-1] < (ptrdiff_t)size) {
        if (ctx->chunk_size < size) { /* Increase the chunk size if it's too small to accommodate the requested length */
            size_t lim = (size_t)PTRDIFF_MAX >> 1;
            do ctx->chunk_size <<= 1;
            while (ctx->chunk_size < size && (ctx->chunk_size <= lim));
        }
        msml__ctx_push_chunk(ctx);
        msml_log_info("Allocated pool chunk: %.03f MiB", (double)ctx->chunk_size/(double)(1<<20));
    }
    ctx->delta -= size;
    ++ctx->alloc_acc;
    ctx->alloc_total += size;
    return ctx->delta;
}

void* msml_ctx_pool_alloc_aligned(msml_ctx_t* ctx, size_t size, size_t align) {
    msml_assert(align && !(align & (align - 1)), "Alignment must be power of 2: %zu", align); /* Alignment must be a power of 2 */
    return (void*)(((uintptr_t)msml_ctx_pool_alloc(ctx, size + align-1) + align-1) & ~(align-1));
}

size_t msml_ctx_total_memory(const msml_ctx_t* ctx) {
    size_t mem = sizeof(*ctx);
    mem += sizeof(*ctx->chunks) * ctx->chunk_cap;
    mem += ctx->alloc_total;
    return mem;
}

msml_prng_algorithm_t msml_ctx_get_prng_algorithm(const msml_ctx_t* ctx) {
    return ctx->prng_algorithm;
}

void msml_ctx_set_prng_algorithm(msml_ctx_t* ctx, msml_prng_algorithm_t algorithm, uint64_t seed) {
    ctx->prng_algorithm = algorithm;
    msml__prng_init(ctx, seed);
}

void msml_ctx_destroy(msml_ctx_t* ctx) {
    size_t mem_total = msml_ctx_total_memory(ctx);
    size_t mem_mapped = ctx->mapped_total;
    void* (*alloc)(void* blk, size_t size) = ctx->alloc_fn;
    for (size_t i=0; i < ctx->chunk_len; ++i) /* Free individual chunks */
        (*alloc)(ctx->chunks[i], 0);
    (*alloc)(ctx->chunks, 0);
    memset(ctx, (uintptr_t)ctx & 0xff, sizeof(*ctx));
    (*alloc)(ctx, 0);
    ctx = NULL;
    double alloc_total, mapped_total;
    const char* alloc_unit, *mapped_unit;
    msml__humanize_memory_size(mem_total, &alloc_total, &alloc_unit);
    msml__humanize_memory_size(mem_mapped, &mapped_total, &mapped_unit);
    printf("Allocated in pool: %.03f %s, Mapped memory: %.03f %s\n", alloc_total, alloc_unit, mapped_total, mapped_unit);
    printf("MSML context destroyed.\n");
}

#define msml__load_local_storage_group(xk, prefix, var) \
    const int64_t prefix##0 = (xk)->var[0]; \
    const int64_t prefix##1 = (xk)->var[1]; \
    const int64_t prefix##2 = (xk)->var[2]; \
    const int64_t prefix##3 = (xk)->var[3]; \
    (void)prefix##0; \
    (void)prefix##1; \
    (void)prefix##2; \
    (void)prefix##3;

#define msml__dot4_unrolled_var_arr(arr, x0, x1, x2, x3) \
    ( \
        arr[0]*(x0) \
        + arr[1]*(x1) \
        + arr[2]*(x2) \
        + arr[3]*(x3) \
    )

#define msml__resolve_physical_ptr(tensor, d0, d1, d2, d3) \
    (((uint8_t*)(tensor)->buf) + msml__dot4_unrolled_var_arr((tensor)->strides, d0, d1, d2, d3))

const msml_dtype_info_t* msml_get_dtype_info(msml_dtype_t type) {
    static const msml_dtype_info_t infos[MSML_DTYPE_COUNT_] = {
        [MSML_DTYPE_F32] = {
            sizeof(float),
            "f32"
        },
    };
    return infos + type;
}

const char* msml_op_get_name(msml_op_t op) {
    #define _(enumerator, mnemonic, argcount) #enumerator
        static const char* const names[MSML_OP__COUNT] = {
            msml_op_def(_, MSML_SEP)
        };
    #undef _
    return names[op];
}

const char* msml_op_get_mnemonic(msml_op_t op) {
    #define _(enumerator, mnemonic, argcount) mnemonic
        static const char* const mnemonics[MSML_OP__COUNT] = {
            msml_op_def(_, MSML_SEP)
        };
    #undef _
    return mnemonics[op];
}

uint8_t msml_op_get_argcount(msml_op_t op) {
    #define _(enumerator, mnemonic, argcount) ((argcount)&0xff)
        static const uint8_t arg_counts[MSML_OP__COUNT] = {
            msml_op_def(_, MSML_SEP)
        };
    #undef _
    return arg_counts[op];
}

/* Rescale factors to push the exponent of a number towards zero. */
#define rescale_exponents(P, N) \
  P(308), P(289), P(270), P(250), P(231), P(212), P(193), P(173), P(154), \
  P(135), P(115), P(96), P(77), P(58), P(38), P(0), P(0), P(0), N(39), N(58), \
  N(77), N(96), N(116), N(135), N(154), N(174), N(193), N(212), N(231), \
  N(251), N(270), N(289)
#define one_e_p(X) 1e+0 ## X
#define one_e_n(X) 1e-0 ## X
static const int16_t msml__rescale_e[] = { rescale_exponents(-, +) };
static const double msml__rescale_n[] = { rescale_exponents(one_e_p, one_e_n) };
#undef one_e_n
#undef one_e_p

/*
** For p in range -70 through 57, this table encodes pairs (m, e) such that
** 4*2^p <= (uint8_t)m*10^e, and is the smallest value for which this holds.
*/
static const int8_t msml__four_ulp_m_e[] = {
    34, -21, 68, -21, 14, -20, 28, -20, 55, -20, 2, -19, 3, -19, 5, -19, 9, -19,
    -82, -18, 35, -18, 7, -17, -117, -17, 28, -17, 56, -17, 112, -16, -33, -16,
    45, -16, 89, -16, -78, -15, 36, -15, 72, -15, -113, -14, 29, -14, 57, -14,
    114, -13, -28, -13, 46, -13, 91, -12, -74, -12, 37, -12, 73, -12, 15, -11, 3,
    -11, 59, -11, 2, -10, 3, -10, 5, -10, 1, -9, -69, -9, 38, -9, 75, -9, 15, -7,
    3, -7, 6, -7, 12, -6, -17, -7, 48, -7, 96, -7, -65, -6, 39, -6, 77, -6, -103,
    -5, 31, -5, 62, -5, 123, -4, -11, -4, 49, -4, 98, -4, -60, -3, 4, -2, 79, -3,
    16, -2, 32, -2, 63, -2, 2, -1, 25, 0, 5, 1, 1, 2, 2, 2, 4, 2, 8, 2, 16, 2,
    32, 2, 64, 2, -128, 2, 26, 2, 52, 2, 103, 3, -51, 3, 41, 4, 82, 4, -92, 4,
    33, 4, 66, 4, -124, 5, 27, 5, 53, 5, 105, 6, 21, 6, 42, 6, 84, 6, 17, 7, 34,
    7, 68, 7, 2, 8, 3, 8, 6, 8, 108, 9, -41, 9, 43, 10, 86, 9, -84, 10, 35, 10,
    69, 10, -118, 11, 28, 11, 55, 12, 11, 13, 22, 13, 44, 13, 88, 13, -80, 13,
    36, 13, 71, 13, -115, 14, 29, 14, 57, 14, 113, 15, -30, 15, 46, 15, 91, 15,
    19, 16, 37, 16, 73, 16, 2, 17, 3, 17, 6, 17
};

/* min(2^32-1, 10^e-1) for e in range 0 through 10 */
static const uint32_t msml__ndigits_dec_threshold[] = {
    0, 9U, 99U, 999U, 9999U, 99999U, 999999U,
    9999999U, 99999999U, 999999999U, 0xffffffffU
};

/* Compute the number of digits in the decimal representation of x. */
static size_t msml__ndigits_dec(uint32_t x) {
    size_t t = ((msml_fls(x | 1) * 77) >> 8) + 1; /* 2^8/77 is roughly log2(10) */
    return t + (x > msml__ndigits_dec_threshold[t]);
}

#define wint_r(x, sh, sc) { uint32_t d = (x*(((1<<sh)+sc-1)/sc))>>sh; x -= d*sc; *p++ = (char)('0'+d); }
static char* msml__wuint9(char* p, uint32_t u) {
    uint32_t v = u / 10000, w;
    u -= v * 10000;
    w = v / 10000;
    v -= w * 10000;
    *p++ = (char)('0'+w);
    wint_r(v, 23, 1000)
    wint_r(v, 12, 100)
    wint_r(v, 10, 10)
    *p++ = (char)('0'+v);
    wint_r(u, 23, 1000)
    wint_r(u, 12, 100)
    wint_r(u, 10, 10)
    *p++ = (char)('0'+u);
    return p;
}
#undef wint_r

#define wint_r(x, sh, sc) { uint32_t d = (x*(((1<<sh)+sc-1)/sc))>>sh; x -= d*sc; *p++ = (char)('0'+d); }
static char* msml__wint(char* p, int32_t k) {
    uint32_t u = (uint32_t)k;
    if (k < 0) { u = ~u+1u; *p++ = '-'; }
    if (u < 10000) {
        if (u < 10) goto dig1;
        if (u < 100) goto dig2;
        if (u < 1000) goto dig3;
    } else {
        uint32_t v = u / 10000; u -= v * 10000;
        if (v < 10000) {
            if (v < 10) goto dig5;
            if (v < 100) goto dig6;
            if (v < 1000) goto dig7;
        } else {
            uint32_t w = v / 10000; v -= w * 10000;
            if (w >= 10) wint_r(w, 10, 10)
                         *p++ = (char)('0'+w);
        }
        wint_r(v, 23, 1000)
        dig7: wint_r(v, 12, 100)
        dig6: wint_r(v, 10, 10)
        dig5: *p++ = (char)('0'+v);
    }
    wint_r(u, 23, 1000)
    dig3: wint_r(u, 12, 100)
    dig2: wint_r(u, 10, 10)
    dig1: *p++ = (char)('0'+u);
    return p;
}
#undef wint_r

/* -- Extended precision arithmetic --------------------------------------- */

/*
** The "nd" format is a fixed-precision decimal representation for numbers. It
** consists of up to 64 uint32_t values, with each uint32_t storing a value
** in the range [0, 1e9). A number in "nd" format consists of three variables:
**
**  uint32_t nd[64];
**  uint32_t ndlo;
**  uint32_t ndhi;
**
** The integral part of the number is stored in nd[0 ... ndhi], the value of
** which is sum{i in [0, ndhi] | nd[i] * 10^(9*i)}. If the fractional part of
** the number is zero, ndlo is zero. Otherwise, the fractional part is stored
** in nd[ndlo ... 63], the value of which is taken to be
** sum{i in [ndlo, 63] | nd[i] * 10^(9*(i-64))}.
**
** If the array part had 128 elements rather than 64, then every double would
** have an exact representation in "nd" format. With 64 elements, all integral
** doubles have an exact representation, and all non-integral doubles have
** enough digits to make both %.99e and %.99f do the right thing.
*/
#define ND_MUL2K_MAX_SHIFT 29
#define ND_MUL2K_DIV1E9(val) ((uint32_t)((val) / 1000000000))

/* Multiply nd by 2^k and add carry_in (ndlo is assumed to be zero). */
static uint32_t nd_mul2k(uint32_t* nd, uint32_t ndhi, uint32_t k, uint32_t carry_in, msml_format_flags sf) {
    uint32_t i, ndlo = 0, start = 1;
    /* Performance hacks. */
    if (k > ND_MUL2K_MAX_SHIFT*2 && MSML_FMT_FP(sf) != MSML_FMT_FP(MSML_FMT_T_FP_F)) {
        start = ndhi - (MSML_FMT_PREC(sf) + 17) / 8;
    }
    /* Real logic. */
    while (k >= ND_MUL2K_MAX_SHIFT) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << ND_MUL2K_MAX_SHIFT) | carry_in;
            carry_in = ND_MUL2K_DIV1E9(val);
            nd[i] = (uint32_t)val - carry_in * 1000000000;
        }
        if (carry_in) {
            nd[++ndhi] = carry_in; carry_in = 0;
            if (start++ == ndlo) ++ndlo;
        }
        k -= ND_MUL2K_MAX_SHIFT;
    }
    if (k) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << k) | carry_in;
            carry_in = ND_MUL2K_DIV1E9(val);
            nd[i] = (uint32_t)val - carry_in * 1000000000;
        }
        if (carry_in) nd[++ndhi] = carry_in;
    }
    return ndhi;
}

/* Divide nd by 2^k (ndlo is assumed to be zero). */
static uint32_t nd_div2k(uint32_t* nd, uint32_t ndhi, uint32_t k, msml_format_flags sf) {
    uint32_t ndlo = 0, stop1 = ~0, stop2 = ~0;
    /* Performance hacks. */
    if (!ndhi) {
        if (!nd[0]) {
            return 0;
        } else {
            uint32_t s = msml_ffs(nd[0]);
            if (s >= k) { nd[0] >>= k; return 0; }
            nd[0] >>= s; k -= s;
        }
    }
    if (k > 18) {
        if (MSML_FMT_FP(sf) == MSML_FMT_FP(MSML_FMT_T_FP_F)) {
            stop1 = 63 - (int32_t)MSML_FMT_PREC(sf) / 9;
        } else {
            int32_t floorlog2 = ndhi * 29 + msml_fls(nd[ndhi]) - k;
            int32_t floorlog10 = (int32_t)(floorlog2 * 0.30102999566398114);
            stop1 = 62 + (floorlog10 - (int32_t)MSML_FMT_PREC(sf)) / 9;
            stop2 = 61 + ndhi - (int32_t)MSML_FMT_PREC(sf) / 8;
        }
    }
    /* Real logic. */
    while (k >= 9) {
        uint32_t i = ndhi, carry = 0;
        for (;;) {
            uint32_t val = nd[i];
            nd[i] = (val >> 9) + carry;
            carry = (val & 0x1ff) * 1953125;
            if (i == ndlo) break;
            i = (i - 1) & 0x3f;
        }
        if (ndlo != stop1 && ndlo != stop2) {
            if (carry) { ndlo = (ndlo - 1) & 0x3f; nd[ndlo] = carry; }
            if (!nd[ndhi]) { ndhi = (ndhi - 1) & 0x3f; stop2--; }
        } else if (!nd[ndhi]) {
            if (ndhi != ndlo) { ndhi = (ndhi - 1) & 0x3f; stop2--; }
            else return ndlo;
        }
        k -= 9;
    }
    if (k) {
        uint32_t mask = (1U << k) - 1, mul = 1000000000 >> k, i = ndhi, carry = 0;
        for (;;) {
            uint32_t val = nd[i];
            nd[i] = (val >> k) + carry;
            carry = (val & mask) * mul;
            if (i == ndlo) break;
            i = (i - 1) & 0x3f;
        }
        if (carry) { ndlo = (ndlo - 1) & 0x3f; nd[ndlo] = carry; }
    }
    return ndlo;
}

/* Add m*10^e to nd (assumes ndlo <= e/9 <= ndhi and 0 <= m <= 9). */
static uint32_t nd_add_m10e(uint32_t* nd, uint32_t ndhi, uint8_t m, int32_t e) {
    uint32_t i, carry;
    if (e >= 0) {
        i = (uint32_t)e/9;
        carry = m * (msml__ndigits_dec_threshold[e - (int32_t)i*9] + 1);
    } else {
        int32_t f = (e-8)/9;
        i = (uint32_t)(64 + f);
        carry = m * (msml__ndigits_dec_threshold[e - f*9] + 1);
    }
    for (;;) {
        uint32_t val = nd[i] + carry;
        if (msml_unlikely(val >= 1000000000)) {
            val -= 1000000000;
            nd[i] = val;
            if (msml_unlikely(i == ndhi)) {
                ndhi = (ndhi + 1) & 0x3f;
                nd[ndhi] = 1;
                break;
            }
            carry = 1;
            i = (i + 1) & 0x3f;
        } else {
            nd[i] = val;
            break;
        }
    }
    return ndhi;
}

static bool nd_similar(uint32_t* nd, uint32_t ndhi, uint32_t* ref, size_t hilen, size_t prec) {
    char nd9[9], ref9[9];
    if (hilen <= prec) {
        if (msml_unlikely(nd[ndhi] != *ref)) return 0;
        prec -= hilen; ref--; ndhi = (ndhi - 1) & 0x3f;
        if (prec >= 9) {
            if (msml_unlikely(nd[ndhi] != *ref)) return 0;
            prec -= 9; ref--; ndhi = (ndhi - 1) & 0x3f;
        }
    } else {
        prec -= hilen - 9;
    }
    msml_assert(prec < 9, "bad precision %d", prec);
    msml__wuint9(nd9, nd[ndhi]);
    msml__wuint9(ref9, *ref);
    return !memcmp(nd9, ref9, prec) && (nd9[prec] < '5') == (ref9[prec] < '5');
}

/* Format f64 according to format flags. */
static char* msml__fmt_f64(msml_format_flags sf, double n, char* p) {
    size_t width = MSML_FMT_WIDTH(sf), prec = MSML_FMT_PREC(sf), len;
    union {
        uint64_t u64;
        double n;
        struct { /* TODO: make endian aware */
            uint32_t lo, hi;
        } u32;
    } t = {.n = n};
    if (msml_unlikely((t.u32.hi << 1) >= 0xffe00000)) {
        /* Handle non-finite values uniformly for %a, %e, %f, %g. */
        int prefix = 0, ch = (sf & MSML_FMT_F_UPPER) ? 0x202020 : 0;
        if (((t.u32.hi & 0x000fffff) | t.u32.lo) != 0) {
            ch ^= ('n' << 16) | ('a' << 8) | 'n';
            if ((sf & MSML_FMT_F_SPACE)) prefix = ' ';
        } else {
            ch ^= ('i' << 16) | ('n' << 8) | 'f';
            if ((t.u32.hi & 0x80000000)) prefix = '-';
            else if ((sf & MSML_FMT_F_PLUS)) prefix = '+';
            else if ((sf & MSML_FMT_F_SPACE)) prefix = ' ';
        }
        len = 3 + (prefix != 0);
        if (!(sf & MSML_FMT_F_LEFT)) while (width-- > len) *p++ = ' ';
        if (prefix) *p++ = prefix;
        *p++ = (char)(ch >> 16); *p++ = (char)(ch >> 8); *p++ = (char)ch;
    } else if (MSML_FMT_FP(sf) == MSML_FMT_FP(MSML_FMT_T_FP_A)) {
        /* %a */
        const char* hexdig = (sf & MSML_FMT_F_UPPER) ? "0123456789ABCDEFPX" : "0123456789abcdefpx";
        int32_t e = (t.u32.hi >> 20) & 0x7ff;
        char prefix = 0, eprefix = '+';
        if (t.u32.hi & 0x80000000) prefix = '-';
        else if ((sf & MSML_FMT_F_PLUS)) prefix = '+';
        else if ((sf & MSML_FMT_F_SPACE)) prefix = ' ';
        t.u32.hi &= 0xfffff;
        if (e) {
            t.u32.hi |= 0x100000;
            e -= 1023;
        } else if (t.u32.lo | t.u32.hi) {
            /* Non-zero denormal - normalise it. */
            uint32_t shift = t.u32.hi ? 20-msml_fls(t.u32.hi) : 52-msml_fls(t.u32.lo);
            e = -1022 - shift;
            t.u64 <<= shift;
        }
        /* abs(n) == t.u64 * 2^(e - 52) */
        /* If n != 0, bit 52 of t.u64 is set, and is the highest set bit. */
        if ((int32_t)prec < 0) {
            /* Default precision: use smallest precision giving exact result. */
            prec = t.u32.lo ? 13-msml_ffs(t.u32.lo)/4 : 5-msml_ffs(t.u32.hi|0x100000)/4;
        } else if (prec < 13) {
            /* Precision is sufficiently low as to maybe require rounding. */
            t.u64 += (((uint64_t)1) << (51 - prec*4));
        }
        if (e < 0) {
            eprefix = '-';
            e = -e;
        }
        len = 5 + msml__ndigits_dec((uint32_t)e) + prec + (prefix != 0)
              + ((prec | (sf & MSML_FMT_F_ALT)) != 0);
        if (!(sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO))) {
            while (width-- > len) *p++ = ' ';
        }
        if (prefix) *p++ = prefix;
        *p++ = '0';
        *p++ = hexdig[17]; /* x or X */
        if ((sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO)) == MSML_FMT_F_ZERO) {
            while (width-- > len) *p++ = '0';
        }
        *p++ = '0' + (t.u32.hi >> 20); /* Usually '1', sometimes '0' or '2'. */
        if ((prec | (sf & MSML_FMT_F_ALT))) {
            /* Emit fractional part. */
            char* q = p + 1 + prec;
            *p = '.';
            if (prec < 13) t.u64 >>= (52 - prec*4);
            else while (prec > 13) p[prec--] = '0';
            while (prec) { p[prec--] = hexdig[t.u64 & 15]; t.u64 >>= 4; }
            p = q;
        }
        *p++ = hexdig[16]; /* p or P */
        *p++ = eprefix; /* + or - */
        p = msml__wint(p, e);
    } else {
        /* %e or %f or %g - begin by converting n to "nd" format. */
        uint32_t nd[64];
        uint32_t ndhi = 0, ndlo, i;
        int32_t e = (int32_t)(t.u32.hi >> 20) & 0x7ff, ndebias = 0;
        char prefix = 0, *q;
        if (t.u32.hi & 0x80000000) prefix = '-';
        else if ((sf & MSML_FMT_F_PLUS)) prefix = '+';
        else if ((sf & MSML_FMT_F_SPACE)) prefix = ' ';
        prec += ((int32_t)prec >> 31) & 7; /* Default precision is 6. */
        if (MSML_FMT_FP(sf) == MSML_FMT_FP(MSML_FMT_T_FP_G)) {
            /* %g - decrement precision if non-zero (to make it like %e). */
            prec--;
            prec ^= (uint32_t)((int32_t)prec >> 31);
        }
        if ((sf & MSML_FMT_T_FP_E) && prec < 14 && n != 0) {
            /* Precision is sufficiently low that rescaling will probably work. */
            if ((ndebias = msml__rescale_e[e >> 6])) {
                t.n = n * msml__rescale_n[e >> 6];
                if (msml_unlikely(!e)) t.n *= 1e10, ndebias -= 10;
                t.u64 -= 2; /* Convert 2ulp below (later we convert 2ulp above). */
                nd[0] = 0x100000 | (t.u32.hi & 0xfffff);
                e = ((int32_t)(t.u32.hi >> 20) & 0x7ff) - 1075 - (ND_MUL2K_MAX_SHIFT < 29);
                goto load_t_lo; rescale_failed:
                t.n = n;
                e = (int32_t)(t.u32.hi >> 20) & 0x7ff;
                ndebias = 0;
                ndhi = 0;
            }
        }
        nd[0] = t.u32.hi & 0xfffff;
        if (e == 0) e++; else nd[0] |= 0x100000;
        e -= 1043;
        if (t.u32.lo) {
            e -= 32 + (ND_MUL2K_MAX_SHIFT < 29); load_t_lo:
            #if ND_MUL2K_MAX_SHIFT >= 29
                nd[0] = (nd[0] << 3) | (t.u32.lo >> 29);
                ndhi = nd_mul2k(nd, ndhi, 29, t.u32.lo & 0x1fffffff, sf);
            #elif ND_MUL2K_MAX_SHIFT >= 11
                ndhi = nd_mul2k(nd, ndhi, 11, t.u32.lo >> 21, sf);
                ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo >> 10) & 0x7ff, sf);
                ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo <<  1) & 0x7ff, sf);
            #else
            #   error "ND_MUL2K_MAX_SHIFT not big enough"
            #endif
        }
        if (e >= 0) {
            ndhi = nd_mul2k(nd, ndhi, (uint32_t)e, 0, sf);
            ndlo = 0;
        } else {
            ndlo = nd_div2k(nd, ndhi, (uint32_t)-e, sf);
            if (ndhi && !nd[ndhi]) ndhi--;
        }
        /* |n| == nd * 10^ndebias (for slightly loose interpretation of ==) */
        if ((sf & MSML_FMT_T_FP_E)) {
            /* %e or %g - assume %e and start by calculating nd's exponent (nde). */
            char eprefix = '+';
            int32_t nde = -1;
            size_t hilen;
            if (ndlo && !nd[ndhi]) {
                ndhi = 64; do {} while (!nd[--ndhi]);
                nde -= 64 * 9;
            }
            hilen = msml__ndigits_dec(nd[ndhi]);
            nde += (int32_t)(ndhi * 9 + hilen);
            if (ndebias) {
                /*
                ** Rescaling was performed, but this introduced some error, and might
                ** have pushed us across a rounding boundary. We check whether this
                ** error affected the result by introducing even more error (2ulp in
                ** either direction), and seeing whether a rounding boundary was
                ** crossed. Having already converted the -2ulp case, we save off its
                ** most significant digits, convert the +2ulp case, and compare them.
                */
                int32_t eidx = e + 70 + (ND_MUL2K_MAX_SHIFT < 29)
                               + (t.u32.lo >= 0xfffffffe && !(~t.u32.hi << 12));
                const int8_t *m_e = msml__four_ulp_m_e + eidx * 2;
                msml_assert(0 <= eidx && eidx < 128, "bad eidx %d", eidx);
                nd[33] = nd[ndhi];
                nd[32] = nd[(ndhi - 1) & 0x3f];
                nd[31] = nd[(ndhi - 2) & 0x3f];
                nd_add_m10e(nd, ndhi, (uint8_t)*m_e, m_e[1]);
                if (msml_unlikely(!nd_similar(nd, ndhi, nd + 33, hilen, prec + 1))) {
                    goto rescale_failed;
                }
            }
            if ((int32_t)(prec - nde) < (0x3f & -(int32_t)ndlo) * 9) {
                /* Precision is sufficiently low as to maybe require rounding. */
                ndhi = nd_add_m10e(nd, ndhi, 5, (int32_t)nde - prec - 1);
                nde += (hilen != msml__ndigits_dec(nd[ndhi]));
            }
            nde += ndebias;
            if ((sf & MSML_FMT_T_FP_F)) {
                /* %g */
                if ((int32_t)prec >= nde && nde >= -4) {
                    if (nde < 0) ndhi = 0;
                    prec -= nde;
                    goto g_format_like_f;
                } else if (!(sf & MSML_FMT_F_ALT) && prec && width > 5) {
                    /* Decrease precision in order to strip trailing zeroes. */
                    char tail[9];
                    uint32_t maxprec = hilen - 1 + ((ndhi - ndlo) & 0x3f) * 9;
                    if (prec >= maxprec) prec = maxprec;
                    else ndlo = (ndhi - (((int32_t)(prec - hilen) + 9) / 9)) & 0x3f;
                    i = prec - hilen - (((ndhi - ndlo) & 0x3f) * 9) + 10;
                    msml__wuint9(tail, nd[ndlo]);
                    while (prec && tail[--i] == '0') {
                        prec--;
                        if (!i) {
                            if (ndlo == ndhi) { prec = 0; break; }
                            ndlo = (ndlo + 1) & 0x3f;
                            msml__wuint9(tail, nd[ndlo]);
                            i = 9;
                        }
                    }
                }
            }
            if (nde < 0) {
                /* Make nde non-negative. */
                eprefix = '-';
                nde = -nde;
            }
            len = 3 + prec + (prefix != 0) + msml__ndigits_dec((uint32_t)nde) + (nde < 10)
                  + ((prec | (sf & MSML_FMT_F_ALT)) != 0);
            if (!(sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO))) {
                while (width-- > len) *p++ = ' ';
            }
            if (prefix) *p++ = prefix;
            if ((sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO)) == MSML_FMT_F_ZERO) {
                while (width-- > len) *p++ = '0';
            }
            q = msml__wint(p + 1, nd[ndhi]);
            p[0] = p[1]; /* Put leading digit in the correct place. */
            if ((prec | (sf & MSML_FMT_F_ALT))) {
                /* Emit fractional part. */
                p[1] = '.'; p += 2;
                prec -= (size_t)(q - p); p = q; /* Account for digits already emitted. */
                /* Then emit chunks of 9 digits (this may emit 8 digits too many). */
                for (i = ndhi; (int32_t)prec > 0 && i != ndlo; prec -= 9) {
                    i = (i - 1) & 0x3f;
                    p = msml__wuint9(p, nd[i]);
                }
                if ((sf & MSML_FMT_T_FP_F) && !(sf & MSML_FMT_F_ALT)) {
                    /* %g (and not %#g) - strip trailing zeroes. */
                    p += (int32_t)prec & ((int32_t)prec >> 31);
                    while (p[-1] == '0') p--;
                    if (p[-1] == '.') p--;
                } else {
                    /* %e (or %#g) - emit trailing zeroes. */
                    while ((int32_t)prec > 0) { *p++ = '0'; prec--; }
                    p += (int32_t)prec;
                }
            } else {
                p++;
            }
            *p++ = (sf & MSML_FMT_F_UPPER) ? 'E' : 'e';
            *p++ = eprefix; /* + or - */
            if (nde < 10) *p++ = '0'; /* Always at least two digits of exponent. */
            p = msml__wint(p, nde);
        } else {
            /* %f (or, shortly, %g in %f style) */
            if (prec < (size_t)(0x3f & -(int32_t)ndlo) * 9) {
                /* Precision is sufficiently low as to maybe require rounding. */
                ndhi = nd_add_m10e(nd, ndhi, 5, 0 - prec - 1);
            }
            g_format_like_f:
            if ((sf & MSML_FMT_T_FP_E) && !(sf & MSML_FMT_F_ALT) && prec && width) {
                /* Decrease precision in order to strip trailing zeroes. */
                if (ndlo) {
                    /* nd has a fractional part; we need to look at its digits. */
                    char tail[9];
                    uint32_t maxprec = (64 - ndlo) * 9;
                    if (prec >= maxprec) prec = maxprec;
                    else ndlo = 64 - (prec + 8) / 9;
                    i = prec - ((63 - ndlo) * 9);
                    msml__wuint9(tail, nd[ndlo]);
                    while (prec && tail[--i] == '0') {
                        prec--;
                        if (!i) {
                            if (ndlo == 63) { prec = 0; break; }
                            msml__wuint9(tail, nd[++ndlo]);
                            i = 9;
                        }
                    }
                } else {
                    /* nd has no fractional part, so precision goes straight to zero. */
                    prec = 0;
                }
            }
            len = ndhi * 9 + msml__ndigits_dec(nd[ndhi]) + prec + (prefix != 0)
                  + ((prec | (sf & MSML_FMT_F_ALT)) != 0);
            if (!(sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO))) {
                while (width-- > len) *p++ = ' ';
            }
            if (prefix) *p++ = prefix;
            if ((sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO)) == MSML_FMT_F_ZERO) {
                while (width-- > len) *p++ = '0';
            }
            /* Emit integer part. */
            p = msml__wint(p, nd[ndhi]);
            i = ndhi;
            while (i) p = msml__wuint9(p, nd[--i]);
            if ((prec | (sf & MSML_FMT_F_ALT))) {
                /* Emit fractional part. */
                *p++ = '.';
                /* Emit chunks of 9 digits (this may emit 8 digits too many). */
                while ((int32_t)prec > 0 && i != ndlo) {
                    i = (i - 1) & 0x3f;
                    p = msml__wuint9(p, nd[i]);
                    prec -= 9;
                }
                if ((sf & MSML_FMT_T_FP_E) && !(sf & MSML_FMT_F_ALT)) {
                    /* %g (and not %#g) - strip trailing zeroes. */
                    p += (int32_t)prec & ((int32_t)prec >> 31);
                    while (p[-1] == '0') p--;
                    if (p[-1] == '.') p--;
                } else {
                    /* %f (or %#g) - emit trailing zeroes. */
                    while ((int32_t)prec > 0) { *p++ = '0'; prec--; }
                    p += (int32_t)prec;
                }
            }
        }
    }
    if ((sf & MSML_FMT_F_LEFT)) while (width-- > len) *p++ = ' ';
    return p;
}

msml_ctx_t* msml_tensor_get_ctx(const msml_tensor_t* tensor) {
    return tensor->ctx;
}

msml_tensor_t* msml_tensor_create(msml_ctx_t* ctx, msml_dtype_t type, const int64_t* dims, int64_t rank, msml_tensor_t* slice, size_t slice_offset) {
    msml_assert(dims != NULL && rank > -1 && rank <= MSML_MAX_DIMS, "Rank must be within (0, %d]", MSML_MAX_DIMS);
    if (slice && slice->slice) { /* Accumulate relative slice offset. */
        slice_offset += slice->slice_offset;
        slice = slice->slice;
    }
    int64_t scalar_size = msml_get_dtype_info(type)->size;
    int64_t buf_size = scalar_size;
    for (int64_t i=0; i < rank; ++i) {
        msml_assert(dims[i] > 0, "Dimension must be > 0: %lld", dims[i]);
        msml_assert(!msml__imull64_ov(msml_max(1, dims[i]), buf_size, &buf_size), "Overflow in buffer size. Max: INT64_MAX. Reduce dimensions.");
    }
    msml_assert2(!slice || !buf_size || buf_size + slice_offset <= slice->buf_size); /* Slice must be within sliced tensor data range. */
    msml_tensor_t* tensor = msml_ctx_pool_alloc(ctx, sizeof(*tensor) + (slice ? 0 : buf_size)); /* Allocate memory for tensor struct and data */
    memset(tensor, 0, sizeof(*tensor));
    tensor->ctx = ctx;
    tensor->rank = rank;
    tensor->dtype = type;
    tensor->buf_size = buf_size;
    for (int64_t i=0; i < MSML_MAX_DIMS; ++i)
        tensor->shape[i] = i < rank ? msml_max(1, dims[i]) : 1;
    *tensor->strides = scalar_size;
    for (int i=1; i < MSML_MAX_DIMS; ++i) {
        msml_assert(!msml__imull64_ov(tensor->strides[i-1], tensor->shape[i - 1], tensor->strides + i), "Overflow in stride calculation. Max: INT64_MAX. Reduce dimensions.");
    }
    tensor->buf = slice ? (uint8_t*)slice->buf+slice_offset : (uint8_t*)(tensor+1); /* Set buffer pointer to the end of the tensor struct, where data follows */
    return tensor;
}

msml_tensor_t* msml_tensor_create_1d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1}, 1, NULL, 0);
}

msml_tensor_t* msml_tensor_create_2d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2}, 2, NULL, 0);
}

msml_tensor_t* msml_tensor_create_3d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2, d3}, 3, NULL, 0);
}

msml_tensor_t* msml_tensor_create_4d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2, d3, d4}, 4, NULL, 0);
}

msml_tensor_t* msml_tensor_get_arg(const msml_tensor_t* tensor, size_t slot) {
    msml_assert(slot < MSML_MAX_ARG_TENSORS, "Slot must be within [0, %d)", MSML_MAX_ARG_TENSORS);
    return tensor->args[slot];
}

void msml_tensor_set_arg(msml_tensor_t* tensor, size_t slot, msml_tensor_t* arg) {
    msml_assert(slot < MSML_MAX_ARG_TENSORS, "Slot must be within [0, %d)", MSML_MAX_ARG_TENSORS);
    msml_assert(tensor->args[slot] == NULL, "Argument at slot #%zu already set", slot);
    tensor->args[slot] = arg;
}

msml_op_t msml_tensor_get_op(const msml_tensor_t* tensor) {
    return tensor->op;
}

void msml_tensor_set_op(msml_tensor_t* tensor, msml_op_t op) {
    tensor->op = op;
}

msml_tensor_t* msml_tensor_isomorphic_clone(msml_tensor_t* tensor) {
    msml_tensor_t* isomorph = msml_tensor_create(tensor->ctx, tensor->dtype, tensor->shape, tensor->rank, NULL, 0);
    return isomorph;
}

msml_tensor_t* msml_tensor_deep_clone(msml_tensor_t* tensor) {
    msml_tensor_t* clone = msml_tensor_isomorphic_clone(tensor);
    memcpy(clone->buf, tensor->buf, tensor->buf_size);
    return clone;
}

void msml_tensor_copy_buffer_from(msml_tensor_t* tensor, const void* data, size_t size) {
    msml_assert(size == (size_t)tensor->buf_size, "Buffer size mismatch: %zu != %lld", size, tensor->buf_size);
    memcpy(tensor->buf, data, size);
}

void msml_tensor_fill(msml_tensor_t* tensor, float x) {
    if (x == 0.0f) {
        memset(tensor->buf, 0, tensor->buf_size);
        return;
    }
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = (float*)tensor->buf;
            for (int64_t i=0; i < n; ++i) buf[i] = x;
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
}

void msml_tensor_fill_random(msml_tensor_t* tensor, float min, float max) {
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = (float*)tensor->buf;
            msml__prng_generate_n(tensor->ctx, buf, n, min, max);
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
}

size_t msml_tensor_get_memory_usage(const msml_tensor_t* tensor) {
    return sizeof(*tensor) + tensor->buf_size;
}

void msml_tensor_print(const msml_tensor_t* tensor, bool with_data) {
    msml_assert(tensor->dtype == MSML_DTYPE_F32, "Tensor must be F32");
    double buf_size_cvt = 0.0;
    const char* buf_size_unit = NULL;
    msml__humanize_memory_size(msml_tensor_get_memory_usage(tensor), &buf_size_cvt, &buf_size_unit);
    printf("Tensor '%s', DType: %s, Rank: %zu, Shape: [%zu, %zu, %zu, %zu], Strides: [%zu, %zu, %zu, %zu], Mem: %.03f %s \n",
       tensor->name,
       msml_get_dtype_info(tensor->dtype)->name,
       (size_t)tensor->rank,
       (size_t)tensor->shape[0],
       (size_t)tensor->shape[1],
       (size_t)tensor->shape[2],
       (size_t)tensor->shape[3],
       (size_t)tensor->strides[0],
       (size_t)tensor->strides[1],
       (size_t)tensor->strides[2],
       (size_t)tensor->strides[3],
       buf_size_cvt,
       buf_size_unit
    );
    if (with_data) {
        printf("[\n");
        const float* buf = (const float*)tensor->buf;
        for (int64_t i3=0; i3 < tensor->shape[2]; ++i3) { // TODO: d4
            printf("[\n");
            for (int64_t i2=0; i2 < tensor->shape[1]; ++i2) {
                putchar('\t');
                for (int64_t i1=0; i1 < tensor->shape[0]; ++i1) {
                    // TODO: dtype check
                    float x = buf[i3 * tensor->shape[1] * tensor->shape[0] + i2 * tensor->shape[0] + i1];
                    char fmt_buf[128];
                    *msml__fmt_f64(MSML_FMT_G14, x, fmt_buf) = '\0';
                    printf("%s ", fmt_buf);
                }
                putchar('\n');
            }
            printf("]\n");
        }
        printf("]\n");
    }
}

void msml_tensor_set_name(msml_tensor_t* tensor, const char* name) {
    strncpy(tensor->name, name, MSML_MAX_TENSOR_NAME_LEN);
    tensor->name[MSML_MAX_TENSOR_NAME_LEN-1] = '\0';
}

const char* msml_tensor_get_name(const msml_tensor_t* tensor) {
    return tensor->name;
}

int64_t msml_tensor_rank(const msml_tensor_t* tensor) {
    return tensor->rank;
}

const int64_t* msml_tensor_shape(const msml_tensor_t* tensor) {
    return tensor->shape;
}

const int64_t* msml_tensor_strides(const msml_tensor_t* tensor) {
    return tensor->strides;
}

msml_dtype_t msml_tensor_dtype(const msml_tensor_t* tensor) {
    return tensor->dtype;
}

void* msml_tensor_buf(const msml_tensor_t* tensor) {
    return tensor->buf;
}

float* msml_tensor_buf_f32(const msml_tensor_t* tensor) {
    msml_assert(tensor->dtype == MSML_DTYPE_F32, "Tensor data type must be F32, not %s", msml_get_dtype_info(tensor->dtype)->name);
    return (float*)tensor->buf;
}

int64_t msml_tensor_buf_size(const msml_tensor_t* tensor) {
    return tensor->buf_size;
}

int64_t msml_tensor_buf_len(const msml_tensor_t* tensor) {
    return tensor->buf_size / msml_get_dtype_info(tensor->dtype)->size;
}

int64_t msml_tensor_num_rows(const msml_tensor_t* tensor) {
    int64_t rows=tensor->shape[1];
    for (int64_t i=2; i < MSML_MAX_DIMS; ++i)
        rows *= tensor->shape[i];
    return rows;
}

int64_t msml_tensor_num_cols(const msml_tensor_t* tensor) {
    return tensor->shape[0];
}

bool msml_tensor_is_scalar(const msml_tensor_t* tensor) {
    for (int i=0; i < MSML_MAX_DIMS; ++i)
        if (tensor->shape[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_vector(const msml_tensor_t* tensor) {
    for (int i=1; i < MSML_MAX_DIMS; ++i)
        if (tensor->shape[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_matrix(const msml_tensor_t* tensor) {
    for (int i=2; i < MSML_MAX_DIMS; ++i)
        if (tensor->shape[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_higher_order_3d(const msml_tensor_t* tensor) {
    for (int i=3; i < MSML_MAX_DIMS; ++i)
        if (tensor->shape[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_shape_eq(const msml_tensor_t* a, const msml_tensor_t* b) {
    return memcmp(a->shape, b->shape, sizeof(a->shape)) == 0;
}

bool msml_tensor_are_strides_eq(const msml_tensor_t* a, const msml_tensor_t* b) {
    return memcmp(a->strides, b->strides, sizeof(a->strides)) == 0;
}

bool msml_tensor_can_broadcast(const msml_tensor_t* a, const msml_tensor_t* b) {
    for (int i=0; i < MSML_MAX_DIMS; ++i)
        if ((b->shape[i] % a->shape[i]) != 0)
            return false;
    return true;
}

void msml_tensor_virtual_to_physical_index(const msml_tensor_t* tensor, int64_t v_idx, int64_t(*p_idx)[MSML_MAX_DIMS]) {
    msml__load_local_storage_group(tensor, d, shape);
    (*p_idx)[3] = v_idx / (d2*d1*d0);
    (*p_idx)[2] = (v_idx - (*p_idx)[3]*d2*d1*d0) / (d1*d0);
    (*p_idx)[1] = (v_idx - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0) / d0;
    (*p_idx)[0] =  v_idx - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0 - (*p_idx)[1]*d0;
}

int64_t msml_tensor_physical_to_virtual_index(const msml_tensor_t* tensor, const int64_t (*p_idx)[MSML_MAX_DIMS]) {
    int64_t v_idx = 0;
    for (int i=0; i < MSML_MAX_DIMS; ++i)
        v_idx += (*p_idx)[i] * tensor->strides[i];
    return v_idx;
}

bool msml_tensor_is_contiguous(const msml_tensor_t* tensor) {
    return *tensor->strides == msml_get_dtype_info(tensor->dtype)->size;
}

float msml_tensor_get_scalar_physical_index(const msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    const uint8_t* dst = msml__resolve_physical_ptr(tensor, d0, d1, d2, d3);
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: return *(float*)dst;
        default: msml_panic("Unsupported data type: %s", msml_get_dtype_info(tensor->dtype)->name);
    }
}

void msml_tensor_set_scalar_physical_index(msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3, float x) {
    uint8_t* dst = msml__resolve_physical_ptr(tensor, d0, d1, d2, d3);
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: *(float*)dst = x; break;
        default: msml_panic("Unsupported data type: %s", msml_get_dtype_info(tensor->dtype)->name);
    }
}

float msml_tensor_get_scalar_virtual_index(const msml_tensor_t* tensor, int64_t v_idx) {
    if (!msml_tensor_is_contiguous(tensor)) {
        int64_t physical_idx[MSML_MAX_DIMS];
        msml_tensor_virtual_to_physical_index(tensor, v_idx, &physical_idx);
        return msml_tensor_get_scalar_physical_index(tensor, physical_idx[0], physical_idx[1], physical_idx[2], physical_idx[3]);
    }
    switch (tensor->dtype) {
        case MSML_DTYPE_F32:
            return ((const float*)tensor->buf)[v_idx];
        default:
            msml_panic("Unsupported data type: %s", msml_get_dtype_info(tensor->dtype)->name);
    }
}

void msml_tensor_set_scalar_virtual_index(msml_tensor_t* tensor, int64_t v_idx, float x) {
    if (!msml_tensor_is_contiguous(tensor)) {
        int64_t physical_idx[MSML_MAX_DIMS];
        msml_tensor_virtual_to_physical_index(tensor, v_idx, &physical_idx);
        msml_tensor_set_scalar_physical_index(tensor, physical_idx[0], physical_idx[1], physical_idx[2], physical_idx[3], x);
        return;
    }
    switch (tensor->dtype) {
        case MSML_DTYPE_F32:
            ((float*)tensor->buf)[v_idx] = x;
            break;
        default:
            msml_panic("Unsupported data type: %s", msml_get_dtype_info(tensor->dtype)->name);
    }
}

bool msml_tensor_eq(const msml_tensor_t* a, const msml_tensor_t* b) {
    if (a->dtype != b->dtype) return false;
    if (a->rank != b->rank) return false;
    if (memcmp(a->shape, b->shape, sizeof(a->shape)) != 0) return false;
    if (a->buf_size != b->buf_size) return false;
    int64_t n = msml_tensor_buf_len(a);
    switch (a->dtype) {
        case MSML_DTYPE_F32: {
            const float* buf_a = (const float*)a->buf;
            const float* buf_b = (const float*)b->buf;
            for (int64_t i = 0; i < n; ++i) {
                if (buf_a[i] != buf_b[i]) {
                    return false;
                }
            }
        } break;
        default:
            msml_panic("Unsupported data type: %s", msml_get_dtype_info(a->dtype)->name);
    }
    return true;
}

bool msml_tensor_isclose(const msml_tensor_t* a, const msml_tensor_t* b, float eps, double* percent_eq) {
    if (a->dtype != b->dtype) return false;
    if (a->rank != b->rank) return false;
    if (memcmp(a->shape, b->shape, sizeof(a->shape)) != 0) return false;
    if (a->buf_size != b->buf_size) return false;
    eps = eps < 0.0f ? FLT_EPSILON : eps;
    int64_t n = msml_tensor_buf_len(a);
    int64_t n_eq = 0;
    switch (a->dtype) {
        case MSML_DTYPE_F32: {
            const float* buf_a = (const float*)a->buf;
            const float* buf_b = (const float*)b->buf;
            for (int64_t i = 0; i < n; ++i)  /* |x - y| <= ε     ∀ x, y ∈ A, B */
                if (fabsf(buf_a[i] - buf_b[i]) <= eps) ++n_eq;
        } break;
        default:
            msml_panic("Unsupported data type: %s", msml_get_dtype_info(a->dtype)->name);
    }
    if (percent_eq) *percent_eq = (double)n_eq / (double)n * 100.0;
    return n_eq == n;
}

/* CPU BLAS impl */
#define MSML__GELU_COEFF 0.044715f

static void MSML_HOTPROC msml__vadd_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] + y[i];
    }
}

static void MSML_HOTPROC msml__vsub_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] - y[i];
    }
}

static void MSML_HOTPROC msml__vmul_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] * y[i];
    }
}

static void MSML_HOTPROC msml__vdiv_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] / y[i];
    }
}

static float MSML_UNUSED MSML_HOTPROC msml__vdot_f32(
    const int64_t n,
    const float* const x,
    const float* const y
) {
    double r = 0.0;
    for (int64_t i=0; i < n; ++i) {
        r += x[i] + y[i];
    }
    return (float)r;
}

static void MSML_HOTPROC msml__vsoftmax_f32( /* softmax : ℝ -> (0, ∞), x |-> e^x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = expf(x[i]); /* e^x */
    }
}

static void MSML_HOTPROC msml__vsoftmax_dv_f32( /* softmax' = softmax : ℝ -> (0, ∞), x |-> e^x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    return msml__vsoftmax_f32(n, o, x);
}

static void MSML_HOTPROC msml__vsigmoid_f32( /* σ : ℝ -> (0, 1), x |-> 1/(1 + e^(-x)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

static void MSML_HOTPROC msml__vsigmoid_dv_f32( /* σ' : ℝ -> (0, 1), x |-> -(e^x / ((e^x + 1)^2)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        const float e_x = expf(x[i]);
        const float e_x1 = e_x + 1.0f;
        o[i] = -(e_x / (e_x1*e_x1));
    }
}

static void MSML_HOTPROC msml__vsilu_f32( /* silu : ℝ -> x |-> x/(1 + e^(-x)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static void MSML_HOTPROC msml__vsilu_dv_f32( /* silu' : ℝ -> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        msml_panic("NYI!");
    }
}

static void MSML_HOTPROC msml__vtanh_f32( /* tanh : ℝ -> (-1, 1), x |-> tanh x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = tanhf(x[i]);
    }
}

static void MSML_HOTPROC msml__vtanh_dv_f32( /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        const float cx = coshf(x[i]);
        o[i] = 1.0f / (cx*cx);
    }
}

static void MSML_HOTPROC msml__vrelu_f32( /* relu : ℝ -> ℝ^+, x |-> max {x, 0} */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = msml_max(x[i], 0.0f);
    }
}

static void MSML_HOTPROC msml__vrelu_dv_f32( /* relu' : ℝ -> ℝ^+, x |-> { 0 if x < 0, UB if x = 0, else 1 */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] <= 0.0f ? 0.0f : 1.0f; /* relu' is mathematically undefined for x = 0, but we return 0 in this case. */
    }
}

static void MSML_HOTPROC msml__vgelu_f32( /* gelu : ℝ -> ℝ, x |-> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = 0.5f*x[i]*(1.0f + tanhf(0.79788456080286535587989211986876f*x[i]*(1.0f + MSML__GELU_COEFF*x[i]*x[i])));
    }
}

static void MSML_HOTPROC msml__vgelu_dv_f32( /* gelu' : ℝ -> ℝ, x |-> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        msml_panic("NYI"); /* TODO */
    }
}

typedef struct msml__blas_ctx {
    int64_t n_threads;
    int64_t thread_idx;
} msml__blas_ctx;

#define msml__blas_impl_unary_op(name, T, vec_op) \
    static void MSML_HOTPROC msml__blas_##name( \
        const msml__blas_ctx* const blas_ctx, \
        msml_tensor_t* const r, \
        const msml_tensor_t* const x \
    ) { \
        msml_assert2(msml_tensor_is_shape_eq(x, r)); \
        uint8_t* const b_r = (uint8_t*)r->buf; \
        const uint8_t* const b_x = (const uint8_t*)x->buf; \
        msml__load_local_storage_group(r, r_s, strides) \
        msml__load_local_storage_group(x, x_s, strides) \
        const int64_t rc = msml_tensor_num_rows(x); \
        const int64_t cc = msml_tensor_num_cols(x); \
        for (int64_t ri=0; ri < rc; ++ri) { \
            vec_op(cc, (T*)(b_r + ri*r_s1), (const T*)(b_x + ri*x_s1)); \
        } \
    }

msml__blas_impl_unary_op(softmax_f32, float, msml__vsoftmax_f32)
msml__blas_impl_unary_op(softmax_dv_f32, float, msml__vsoftmax_dv_f32)
msml__blas_impl_unary_op(sigmoid_f32, float, msml__vsigmoid_f32)
msml__blas_impl_unary_op(sigmoid_dv_f32, float, msml__vsigmoid_dv_f32)
msml__blas_impl_unary_op(silu_f32, float, msml__vsilu_f32)
msml__blas_impl_unary_op(silu_dv_f32, float, msml__vsilu_dv_f32)
msml__blas_impl_unary_op(tanh_f32, float, msml__vtanh_f32)
msml__blas_impl_unary_op(tanh_dv_f32, float, msml__vtanh_dv_f32)
msml__blas_impl_unary_op(relu_f32, float, msml__vrelu_f32)
msml__blas_impl_unary_op(relu_dv_f32, float, msml__vrelu_dv_f32)
msml__blas_impl_unary_op(gelu_f32, float, msml__vgelu_f32)
msml__blas_impl_unary_op(gelu_dv_f32, float, msml__vgelu_dv_f32)

#undef msml__blas_impl_unary_op

#define msml__blas_impl_binary_op(name, T, vec_op, scalar_op) \
    static void MSML_HOTPROC msml__blas_##name( \
        const msml__blas_ctx* const blas_ctx, \
        msml_tensor_t* const r, \
        const msml_tensor_t* const x, \
        const msml_tensor_t* const y \
    ) { \
        msml_assert2(msml_tensor_can_broadcast(y, x)); \
        msml_assert2(msml_tensor_is_shape_eq(x, r)); \
        uint8_t* const b_r = (uint8_t*)r->buf; \
        const uint8_t* const b_x = (const uint8_t*)x->buf; \
        const uint8_t* const b_y = (const uint8_t*)y->buf; \
        msml__load_local_storage_group(r, r_d, shape) \
        msml__load_local_storage_group(r, r_s, strides) \
        msml__load_local_storage_group(x, x_d, shape) \
        msml__load_local_storage_group(x, x_s, strides) \
        msml__load_local_storage_group(y, y_d, shape) \
        msml__load_local_storage_group(y, y_s, strides) \
        msml_assert2(r_s0 == sizeof(T)); \
        msml_assert2(x_s0 == sizeof(T)); \
        const int64_t rc = msml_tensor_num_rows(x);  \
        const int64_t ti = blas_ctx->thread_idx;  \
        const int64_t tc = blas_ctx->n_threads;  \
        const int64_t rpt = (rc + tc - 1)/tc;  \
        const int64_t rs = rpt * ti;  \
        const int64_t re = msml_min(rs+rpt, rc); \
        if (y_s0 == sizeof(T)) { \
            for (int64_t ri=rs; ri < re; ++ri) { \
                const int64_t x_i3 = ri / (x_d2*x_d1); \
                const int64_t x_i2 = (ri - x_i3*x_d2*x_d1)/x_d1; \
                const int64_t x_i1 = ri - x_i3*x_d2*x_d1 - x_i2*x_d1; \
                const int64_t y_i3 = x_i3 % y_d3; \
                const int64_t y_i2 = x_i2 % y_d2; \
                const int64_t y_i1 = x_i1 % y_d1; \
                T* const p_r = (T*)(b_r + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1); \
                const T* const p_x = (const T*)(b_x + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1); \
                const T* const p_y = (const T*)(b_y + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1); \
                const int64_t pa = x_d0 / y_d0; \
                for (int64_t i=0; i < pa; ++i) { \
                    vec_op(y_d0, p_r + i*y_d0, p_x + i*y_d0, p_y); \
                } \
            } \
        } else { \
            for (int64_t ri=rs; ri < re; ++ri) { \
                const int64_t x_i3 = ri / (x_d2*x_d1); \
                const int64_t x_i2 = (ri - x_i3*x_d2*x_d1)/x_d1; \
                const int64_t x_i1 = ri - x_i3*x_d2*x_d1 - x_i2*x_d1; \
                const int64_t y_i3 = x_i3 % y_d3; \
                const int64_t y_i2 = x_i2 % y_d2; \
                const int64_t y_i1 = x_i1 % y_d1; \
                T* const p_r = (T*)(b_r + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1); \
                const T* const p_x = (const T*)(b_x + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1); \
                for (int64_t i=0; i < r_d0; ++i) { \
                    p_r[i] = p_x[i] scalar_op *(const T*)(b_y + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1 + i%y_d0*y_s0); \
                } \
            } \
        } \
    }

msml__blas_impl_binary_op(add_f32, float, msml__vadd_f32, +)
msml__blas_impl_binary_op(sub_f32, float, msml__vsub_f32, -)
msml__blas_impl_binary_op(mul_f32, float, msml__vmul_f32, *)
msml__blas_impl_binary_op(div_f32, float, msml__vdiv_f32, /)

#undef msml__blas_impl_binary_op

static void msml__blas_matmul_f32(
    const msml__blas_ctx* const blas_ctx,
    msml_tensor_t* const r,
    const msml_tensor_t* const x,
    const msml_tensor_t* const y
) {
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    const uint8_t* const b_y = (const uint8_t*)y->buf;
    msml__load_local_storage_group(r, r_d, shape)
    msml__load_local_storage_group(r, r_s, strides)
    msml__load_local_storage_group(x, x_d, shape)
    msml__load_local_storage_group(x, x_s, strides)
    msml__load_local_storage_group(y, y_d, shape)
    msml__load_local_storage_group(y, y_s, strides)
    for (int64_t i3=0; i3 < r_d3; ++i3) {
        for (int64_t i2=0; i2 < r_d2; ++i2) {
            for (int64_t i1=0; i1 < r_d1; ++i1) {
                for (int64_t i0=0; i0 < r_d1; ++i0) {
                    double sum = 0.0;
                    for (int64_t k=0; k < x_d0; ++k) {
                        const float* const p_x = (const float*)(b_x + k*x_s0 + i0*x_s1 + i2*x_s2 + i3*x_s3);
                        const float* const p_y = (const float*)(b_y + i1*y_s0 + k*y_s1 + i2*y_s2 + i3*y_s3);
                        sum += (double)(*p_x**p_y);
                    }
                    float* const p_r = (float*)(b_r + i1*r_s0 + i0*r_s1 + i2*r_s2 + i3*r_s3);
                    *p_r = (float)sum;
                }
            }
        }
    }
}

static void MSML_HOTPROC msml__compute_dag_eval(const msml__blas_ctx* const blas_ctx, msml_tensor_t* const node) {
    if (!node || node->op == MSML_OP_NOP) return;
    msml_tensor_t** const args = node->args;
    const int n_args = (int)msml_op_get_argcount(node->op);
    for (int i=0; i < n_args; ++i) { /* Eval parents */
        msml_assert2(args[i] != 0);
        msml__compute_dag_eval(blas_ctx, args[i]); /* Eval parent node recursive */
    }
    /* TODO: computed goto / dispatch table */
    switch (node->op) {
        case MSML_OP_NOP: default: return;
        case MSML_OP_SOFTMAX: msml__blas_softmax_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_SOFTMAX_DV: msml__blas_softmax_dv_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_SIGMOID: msml__blas_sigmoid_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_SIGMOID_DV: msml__blas_sigmoid_dv_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_SILU: msml__blas_silu_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_SILU_DV: msml__blas_silu_dv_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_TANH: msml__blas_tanh_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_TANH_DV: msml__blas_tanh_dv_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_RELU: msml__blas_relu_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_RELU_DV: msml__blas_relu_dv_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_GELU: msml__blas_gelu_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_GELU_DV: msml__blas_gelu_dv_f32(blas_ctx, node, args[0]); return;
        case MSML_OP_ADD: msml__blas_add_f32(blas_ctx, node, args[0], args[1]); return;
        case MSML_OP_SUB: msml__blas_sub_f32(blas_ctx, node, args[0], args[1]); return;
        case MSML_OP_MUL: msml__blas_mul_f32(blas_ctx, node, args[0], args[1]); return;
        case MSML_OP_DIV: msml__blas_div_f32(blas_ctx, node, args[0], args[1]); return;
        case MSML_OP_MATMUL: msml__blas_matmul_f32(blas_ctx, node, args[0], args[1]); return;
    }
}

void MSML_HOTPROC msml_tensor_evaluate(msml_tensor_t* tensor) {
    msml__blas_ctx blas_ctx = {
        .n_threads = 1,
        .thread_idx = 0
    };
    msml__compute_dag_eval(& blas_ctx, tensor);
}

#define msml__save_fwrite(f, file_name, data, size) \
    do { \
        size_t written = fwrite((data), 1, (size), (f)); \
        msml_assert(written == (size), "Failed to write data to file: %s, Written: %zu, Should be: %zu", (file_name), written, (size)); \
    } while (0);

/*
** MSML tensor storage format
** +--------------------------+
** |         Header           | typeof(msml__storage_header)
** +--------------------------+
** +--------------------------+
** |          Tensor 0        | typeof(msml__storage_tensor_header) + DATA
** +--------------------------+
** +--------------------------+
** |          Tensor 1        | typeof(msml__storage_tensor_header) + DATA
** +--------------------------+
**              ...
** +--------------------------+
** |          Tensor N        | typeof(msml__storage_tensor_header) + DATA, N = header.stored_tensors
** +--------------------------+
**
** WORKFLOW
** 1. Write file header (msml__storage_header) DONE IN msml__storage_header_serialize
** 2. Write all tensors (msml__storage_tensor_header + tensor data) DONE IN msml__serialize_tensor_to_buffer WITH msml__storage_tensor_serialize
** 3. Compute checksum of whole buffer. Except fields in msml__storage_header BEFORE msml__storage_header_checksum_offset. DONE IN msml__storage_header_fixup_checksum.
**    So checksum is computed from BUFFER_START+msml__storage_header_checksum_offset ... BUFFER_END (BUFFER_START contains the header).
** 4. Write back full checksum into msml__storage_header.checksum DONE IN msml__storage_header_fixup_checksum.
** 5. Use your serialized tensor buffer (dump to file, send to network etc.)
*/

#define MSML__HEADER_KEY 0x68539076fd713daeull /* Key to XOR-encrypt some header fields. */

#define msml__adjust_header_field_u8(x) (*(x)^=(MSML__HEADER_KEY&0xff))
#define msml__adjust_header_field_u32(x) (*(x)^=(MSML__HEADER_KEY&~0u), msml__bswap32(x))
#define msml__adjust_header_field_u64(x) (*(x)^=MSML__HEADER_KEY, msml__bswap64(x))

typedef struct msml__storage_header { /* Order & size matter, do NOT reorder!. */
    uint32_t magic;
    uint32_t checksum;
    uint32_t msml_version;
    uint32_t storage_version;
    uint32_t stored_tensors;
} msml__storage_header;
msml_static_assert(sizeof(msml__storage_header)%4 == 0);
msml_static_assert(sizeof(msml__storage_header) == 4*5);
msml_static_assert(offsetof(msml__storage_header, magic) == 0);
msml_static_assert(offsetof(msml__storage_header, checksum) == 4);
msml_static_assert(offsetof(msml__storage_header, msml_version) == 4+4);
#define msml__storage_header_checksum_offset offsetof(msml__storage_header, msml_version) /* Magic + checksum itself are excluded from integrity test. */

static void msml__storage_header_serialize(
    uint8_t** pp,
    const uint8_t* end,
    size_t tensor_count
) {
    /* Prepare header */
    uint32_t msml_magic = 0;
    msml_static_assert(sizeof(msml_magic) == sizeof("MSML")-1);
    memcpy(&msml_magic, "MSML", sizeof("MSML")-1);
    msml__storage_header header = {
        .magic = msml_magic,
        .checksum = 0, /* Written later with msml__storage_header_fixup_checksum. */
        .msml_version = MSML_VERSION,
        .storage_version = MSML_STORAGE_VERSION,
        .stored_tensors = (uint32_t)tensor_count
    };

    /* Fixup header endianess */
    msml__adjust_header_field_u32(&header.magic);
    /* msml__adjust_header_field_u32(&header.checksum); ! Checksum is set later with msml__storage_header_fixup_checksum */
    msml__adjust_header_field_u32(&header.msml_version);
    msml__adjust_header_field_u32(&header.storage_version);
    msml__adjust_header_field_u32(&header.stored_tensors);

    /* Write header */
    msml_assert2(*pp+sizeof(header) < end);
    memcpy(*pp, &header, sizeof(header));
    *pp += sizeof(header);
}

static void msml__storage_header_fixup_checksum(uint8_t* base, size_t total_size) {
    size_t moff = msml__storage_header_checksum_offset;
    uint32_t checksum = msml__crc32(base+moff, total_size); /* Compute checksum from [buf_base + offset, end] */
    msml__adjust_header_field_u32(&checksum);
    memcpy(base+offsetof(msml__storage_header, checksum), &checksum, sizeof(checksum)); /* Overwrite computed checksum in buffer at correct pos. */
}

typedef struct msml__storage_tensor_header { /* Order & size matter, do NOT reorder! */
    int64_t dims[MSML_MAX_DIMS];
    uint8_t rank;
    uint8_t dtype; /* Type: msml_dtype_t */
    uint8_t pad__[2];
    uint64_t reserved__;
    char name[MSML_MAX_TENSOR_NAME_LEN];
} msml__storage_tensor_header;
msml_static_assert(sizeof(msml__storage_tensor_header)%4 == 0);

static void msml__storage_tensor_serialize_single(
    uint8_t** pp,
    const uint8_t* end,
    const msml_tensor_t* tensor
) {
    uint8_t* p = *pp;

    /* Prepare header */
    msml__storage_tensor_header header = {
        .dims = {0},
        .rank = (uint8_t)tensor->rank,
        .dtype = (uint8_t)tensor->dtype,
        .name = {0}
    };
    memcpy(header.dims, tensor->shape, sizeof(header.dims));
    memcpy(header.name, tensor->name, sizeof(header.name));
    /* Fixup header endianness */
    for (int j=0; j < MSML_MAX_DIMS; ++j)
        msml__adjust_header_field_u64((uint64_t*)header.dims+j);
    msml__adjust_header_field_u8(&header.rank);
    msml__adjust_header_field_u8(&header.dtype);
    for (size_t i=0; i < MSML_MAX_TENSOR_NAME_LEN; ++i)
        msml__adjust_header_field_u8(header.name+i);

    /* Write header */
    msml_assert2(p+sizeof(header) < end);
    memcpy(p, &header, sizeof(header));
    p += sizeof(header);

    /* Write data */
    int64_t n = msml_tensor_buf_len(tensor);
    msml_assert2(p+n*msml_get_dtype_info(tensor->dtype)->size <= end);
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            const float* t_p = (const float*)tensor->buf;
            const float* t_end = t_p + n;
            for (; t_p < t_end; ++t_p) {
                uint32_t u32;
                memcpy(&u32, t_p, sizeof(u32));
                msml__bswap32(&u32);
                memcpy(p, &u32, sizeof(u32));
                p += sizeof(u32);
            }
            break;
        }
        default:
            msml_panic("Unsupported data type: %s", msml_get_dtype_info(tensor->dtype)->name);
    }

    *pp = p;
}

static void msml__serialize_tensor_to_buffer(
    msml_ctx_t* ctx,
    const msml_tensor_t** tensors,
    const size_t tensor_count,
    uint8_t** out_buf,
    size_t* out_buf_size
) {
    msml_assert2(tensor_count != 0);
    size_t total_size = 0;
    total_size += sizeof(msml__storage_header);
    total_size += sizeof(msml__storage_tensor_header) * tensor_count;
    for (size_t i = 0; i < tensor_count; ++i) {
        total_size += tensors[i]->buf_size;
    }

    uint8_t* const base = (*ctx->alloc_fn)(NULL, total_size);
    uint8_t* const end = base + total_size;
    uint8_t* p = base; /* Needle */

    msml__storage_header_serialize(&p, end, tensor_count); /* Write header first. */

    /* Write all tensors */
    for (size_t i=0; i < tensor_count; ++i) {
        const msml_tensor_t* tensor = tensors[i];
        msml__storage_tensor_serialize_single(& p, end, tensor);
    }

    /* Write checksum back into buffer header after it has been computed. */
    msml__storage_header_fixup_checksum(base, total_size);

    *out_buf = base;
    *out_buf_size = total_size;
}

void msml_tensor_save(const msml_tensor_t* tensor, const char* file_name) {
    FILE* f = msml__fopen(file_name, "wb");
    msml_assert(f, "Failed to open MSML file for writing: %s", file_name);
    msml_ctx_t* ctx = tensor->ctx;
    const msml_tensor_t** tensors = &tensor;
    const size_t len = 1;
    uint8_t* buf = NULL;
    size_t buf_size = 0;
    msml__serialize_tensor_to_buffer(tensor->ctx, tensors, len, &buf, &buf_size);
    msml__save_fwrite(f, file_name, buf, buf_size);
    (*ctx->alloc_fn)(buf, 0); /* Free serialized buffer */
    fclose(f);
    msml_log_info("Saved tensor to: '%s', %.03f MiB written.", file_name, (double)buf_size/(double)(1<<20));
}

msml_tensor_t* msml_tensor_load(msml_ctx_t* ctx, const char* file_name) {
    FILE* f = msml__fopen(file_name, "rb");
    msml_assert(f, "Failed to open MSML file for writing: %s", file_name);
    // TODO
    fclose(f);
    return NULL;
}

#undef msml__save_fwrite

msml_tensor_t* msml_tensor_create_from_image(msml_ctx_t* ctx, const char* file_path, msml_desired_color_channels_t in_desired_channels, uint32_t resize_width, uint32_t resize_height) {
#ifdef MSML_ENABLE_IMAGE_SUPPORT
    int width, height, channels, desired_channels;
    switch (in_desired_channels) {
        default: desired_channels = STBI_default; break;
        case MSML_COLOR_CHANNELS_GRAY: desired_channels = STBI_grey; break;
        case MSML_COLOR_CHANNELS_GRAY_A: desired_channels = STBI_grey_alpha; break;
        case MSML_COLOR_CHANNELS_RGB: desired_channels = STBI_rgb; break;
        case MSML_COLOR_CHANNELS_RGBA: desired_channels = STBI_rgb_alpha; break;
    }
    uint8_t* image_data = stbi_load(file_path, &width, &height, &channels, desired_channels);
    if (!image_data || width == 0 || height == 0 || channels == 0) {
        msml_panic("Failed to load image from file: %s\n", file_path);
    }
    if (resize_width && resize_height) { /* Resize image if requested */
        uint8_t* resized_data = stbir_resize_uint8_srgb(
            image_data,
            width,
            height,
            0,
            NULL,
            (int)resize_width,
            (int)resize_height,
            0,
            (stbir_pixel_layout)desired_channels
        );
        if (resized_data) { /* Replace original image data with resized data */
            stbi_image_free(image_data);
            image_data = resized_data;
            width = (int)resize_width;
            height = (int)resize_height;
        }
    }
    msml_tensor_t* tensor = msml_tensor_create_3d(ctx, MSML_DTYPE_F32, width, height, channels);
    float* dst = (float*)tensor->buf;
    int64_t n = width*height*channels;
    msml_assert(n == msml_tensor_buf_len(tensor), "Buffer size mismatch: %zu != %lld", n, msml_tensor_buf_len(tensor));
    for (int64_t i=0; i < n; ++i)
        dst[i] = (float)image_data[i] / 255.0f; /* Normalize pixel values to [0, 1] */
    stbi_image_free(image_data);
    msml_log_info("Loaded tensor from image: %s, width: %d, height: %d, channels: %d", file_path, width, height, channels);
    return tensor;
#else
    msml_panic("Image support is disabled. MSML must be compiled with MSML_ENABLE_IMAGE_SUPPORT defined.");
#endif
}

void msml_tensor_save_to_image(const msml_tensor_t* tensor, const char* file_path) {
#ifdef MSML_ENABLE_IMAGE_SUPPORT
    const int64_t* dims = msml_tensor_shape(tensor);
    int64_t rank = msml_tensor_rank(tensor);
    msml_assert(rank == 3, "Tensor rank must be 3, but is: %" PRIi64, (size_t)rank);
    int64_t width = dims[0];
    int64_t height = dims[1];
    int64_t channels = dims[2];
    msml_assert(channels == 1 || channels == 3 || channels == 4, "Invalid number of channels: %" PRIi64, channels);
    size_t n = width*height*channels;
    msml_assert(n == msml_tensor_buf_len(tensor), "Buffer size mismatch: %zu != %lld", n, msml_tensor_buf_len(tensor));
    uint8_t* image_data = (*tensor->ctx->alloc_fn)(NULL, n); /* Allocate memory for image data */
    for (size_t i=0; i < n; ++i) { /* Clamp and denormalize pixel values to [0, 255] */
        image_data[i] = (uint8_t)(255.0f * msml_min(msml_max(((const float*)tensor->buf)[i], 0.0f), 1.0f));
    }
    int result = stbi_write_jpg(file_path, (int)width, (int)height, (int)channels, image_data, 100);
    msml_assert(result, "Failed to save tensor to image: %s", file_path);
    (*tensor->ctx->alloc_fn)(image_data, 0); /* Free image data */
    msml_log_info("Saved tensor to image: %s, width: %d, height: %d, channels: %d", file_path, (int)width, (int)height, (int)channels);
#else
    msml_panic("Image support is disabled. MSML must be compiled with MSML_ENABLE_IMAGE_SUPPORT defined.");
#endif
}
