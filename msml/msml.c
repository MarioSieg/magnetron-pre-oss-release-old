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
    int64_t dims[MSML_MAX_DIMS];
    int64_t strides[MSML_MAX_DIMS];
    msml_dtype_t dtype;
    union {
        uint8_t* u8;
        float* f32;
    } buf;
    int64_t buf_size;
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

static void MSML_AINLINE MSML_UNUSED msml__bswap32(uint32_t* p_x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
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

static void MSML_AINLINE MSML_UNUSED msml__bswap64(uint64_t* p_x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
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
        0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
        0x0eDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91,
        0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
        0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
        0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
        0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
        0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
        0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
        0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
        0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01,
        0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
        0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
        0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB,
        0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
        0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
        0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD,
        0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683,
        0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
        0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7,
        0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5,
        0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
        0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79,
        0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236, 0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F,
        0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
        0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713,
        0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21,
        0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
        0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
        0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB,
        0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
        0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD70693, 0x54DE5729, 0x23D967BF,
        0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94, 0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
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
                        y = (state[i] & 0x80000000u) | (state[i + 1] & 0x7fffffffu);
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
                out_gen[ii] = min + rescale_uniform * ((float)(y & ((1u<<24)-1)) * (1.0f / (float)(1u<<24)));
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
                out_gen[ii] = min + rescale_uniform * ((float)(y & ((1u<<24)-1)) * (1.0f / (float)(1u<<24)));
            }
        } break;
        default:
            msml_panic("Unknown PRNG algorithm: %d", ctx->prng_algorithm);
    }
}

static void msml__prng_init(msml_ctx_t* ctx, uint64_t seed) {
    seed = seed ? seed : 0x853c49e6748fea9bull; /* Default seed. */
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
    ctx->mapped_total += ctx->chunk_size;
    ctx->delta = chunk + ctx->chunk_size;
    if (ctx->chunk_len == ctx->chunk_cap)
        ctx->chunks = (*ctx->alloc_fn)(ctx->chunks, (ctx->chunk_cap<<=1) * sizeof(*ctx->chunks));
    ctx->chunks[ctx->chunk_len++] = chunk;
}

msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info) {
    printf("Creating MSML context...\n");
    msml_ctx_info_t ctx_info;
    memset(&ctx_info, 0, sizeof(ctx_info));
    if (info) ctx_info = *info;
    ctx_info.alloc_fn = ctx_info.alloc_fn ? ctx_info.alloc_fn : &msml_allocator;
    msml_ctx_t* ctx = (*ctx_info.alloc_fn)(NULL, sizeof(*ctx));
    memset(ctx, 0, sizeof(*ctx));
    ctx->alloc_fn = ctx_info.alloc_fn;
    ctx->user_data = ctx_info.user_data;
    ctx->chunk_size = ctx_info.pool_chunk_size ? msml_max(ctx_info.pool_chunk_size, 8) : MSML_DEFAULT_CHUNK_SIZE;
    ctx->chunk_cap = ctx_info.pool_chunks_cap ? msml_max(ctx_info.pool_chunks_cap, 1) : MSML_DEFAULT_CHUNK_CAP;
    ctx->chunks = (uint8_t**)(*ctx->alloc_fn)(NULL, ctx->chunk_cap * sizeof(*ctx->chunks));
    msml__ctx_push_chunk(ctx);
    uint64_t host_tid = msml__thread_id();
    ctx->prng_algorithm = ctx_info.prng_algorithm;
    msml__prng_init(ctx, ctx_info.prng_seed^host_tid^(uintptr_t)ctx^(uintptr_t)&ctx_info);
    ctx->host_thread_id = host_tid;
    printf("MSML context created.\n");
    return ctx;
}

void* msml_ctx_pool_alloc(msml_ctx_t* ctx, size_t size) {
    msml_assert(size > 0 && size < (size_t)PTRDIFF_MAX, "Allocation size must be within (0, %zu), but is: %zu", PTRDIFF_MAX, size);
    if (ctx->delta - ctx->chunks[ctx->chunk_len-1] < (ptrdiff_t)size) {
        if (ctx->chunk_size < size) { /* Increase the chunk size if it's too small to accommodate the requested length */
            const size_t lim = (size_t)PTRDIFF_MAX>>1;
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
    mem += ctx->mapped_total;
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
    for (size_t i=0; i < ctx->chunk_len; ++i) /* Free individual chunks */
        (*ctx->alloc_fn)(ctx->chunks[i], 0);
    (*ctx->alloc_fn)(ctx->chunks, 0);
    (*ctx->alloc_fn)(ctx, 0);
    double mem_size; const char* mem_unit;
    msml__humanize_memory_size(mem_total, &mem_size, &mem_unit);
    printf("Total memory allocated: %.03f %s\n", mem_size, mem_unit);
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
    ((tensor)->buf.u8 + msml__dot4_unrolled_var_arr((tensor)->strides, d0, d1, d2, d3))

const msml_dtype_info_t msml_dtype_info[MSML_DTYPE_COUNT_] = {
    [MSML_DTYPE_F32] = {
        sizeof(float),
        "f32"
    },
};

msml_ctx_t* msml_tensor_get_ctx(const msml_tensor_t* tensor) {
    return tensor->ctx;
}

msml_tensor_t* msml_tensor_create(msml_ctx_t* ctx, msml_dtype_t type, const int64_t* dims, int64_t rank) {
    msml_assert(dims != NULL && rank > -1 && rank <= MSML_MAX_DIMS, "Rank must be within (0, %d]", MSML_MAX_DIMS);
    int64_t scalar_size = (int64_t)msml_dtype_info[type].size;
    int64_t buf_size = scalar_size;
    for (int64_t i=0; i < rank; ++i) {
        msml_assert(dims[i] > 0, "Dimension must be > 0: %lld", dims[i]);
        msml_assert(!msml__imull64_ov(msml_max(1, dims[i]), buf_size, &buf_size), "Overflow in buffer size. Max: INT64_MAX. Reduce dimensions.");
    }
    msml_tensor_t* tensor = msml_ctx_pool_alloc(ctx, sizeof(*tensor) + buf_size); /* Allocate memory for tensor struct and data */
    memset(tensor, 0, sizeof(*tensor));
    tensor->ctx = ctx;
    tensor->rank = rank;
    tensor->dtype = type;
    tensor->buf_size = buf_size;
    for (int64_t i=0; i < MSML_MAX_DIMS; ++i)
        tensor->dims[i] = i < rank ? msml_max(1, dims[i]) : 1;
    *tensor->strides = scalar_size;
    for (int i=1; i < MSML_MAX_DIMS; ++i) {
        msml_assert(!msml__imull64_ov(tensor->strides[i-1], tensor->dims[i-1], tensor->strides+i), "Overflow in stride calculation. Max: INT64_MAX. Reduce dimensions.");
    }
    tensor->buf.u8 = (uint8_t*)(tensor + 1); /* Set buffer pointer to the end of the tensor struct, where data follows */
    return tensor;
}

msml_tensor_t* msml_tensor_create_1d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1}, 1);
}

msml_tensor_t* msml_tensor_create_2d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2}, 2);
}

msml_tensor_t* msml_tensor_create_3d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2, d3}, 3);
}

msml_tensor_t* msml_tensor_create_4d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2, d3, d4}, 4);
}

msml_tensor_t* msml_tensor_isomorphic_clone(msml_tensor_t* tensor) {
    msml_tensor_t* isomorph = msml_tensor_create(tensor->ctx, tensor->dtype, tensor->dims, tensor->rank);
    return isomorph;
}

msml_tensor_t* msml_tensor_deep_clone(msml_tensor_t* tensor) {
    msml_tensor_t* clone = msml_tensor_isomorphic_clone(tensor);
    memcpy(clone->buf.u8, tensor->buf.u8, tensor->buf_size);
    return clone;
}

void msml_tensor_copy_buffer_from(msml_tensor_t* tensor, const void* data, size_t size) {
    msml_assert(size == (size_t)tensor->buf_size, "Buffer size mismatch: %zu != %lld", size, tensor->buf_size);
    memcpy(tensor->buf.u8, data, size);
}

void msml_tensor_fill_zero(msml_tensor_t* tensor) {
    memset(tensor->buf.u8, 0, tensor->buf_size);
}

void msml_tensor_fill_one(msml_tensor_t* tensor) {
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = tensor->buf.f32;
            for (int64_t i=0; i < n; ++i) buf[i] = 1.0f;
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
}

void msml_tensor_fill(msml_tensor_t* tensor, float x) {
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = tensor->buf.f32;
            for (int64_t i=0; i < n; ++i) buf[i] = x;
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
}

void msml_tensor_fill_random(msml_tensor_t* tensor, float min, float max) {
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = tensor->buf.f32;
            msml__prng_generate_n(tensor->ctx, buf, n, min, max);
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
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
        int32_t e = (t.u32.hi >> 20) & 0x7ff, ndebias = 0;
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

void msml_tensor_print(const msml_tensor_t* tensor, bool with_data) {
    double buf_size_cvt = 0.0;
    const char* buf_size_unit = NULL;
    msml__humanize_memory_size(tensor->buf_size, &buf_size_cvt, &buf_size_unit);
    printf("Tensor '%s', DType: %s, Rank: %zu, Dims: [%zu, %zu, %zu, %zu], Strides: [%zu, %zu, %zu, %zu], Size: %.01f %s \n",
       tensor->name,
       msml_dtype_info[tensor->dtype].name,
       (size_t)tensor->rank,
       (size_t)tensor->dims[0],
       (size_t)tensor->dims[1],
       (size_t)tensor->dims[2],
       (size_t)tensor->dims[3],
       (size_t)tensor->strides[0],
       (size_t)tensor->strides[1],
       (size_t)tensor->strides[2],
       (size_t)tensor->strides[3],
       buf_size_cvt,
       buf_size_unit
    );
    if (with_data) {
        printf("[\n");
        const float* buf = tensor->buf.f32;
        for (int64_t i3=0; i3 < tensor->dims[2]; ++i3) { // TODO: d4
            for (int64_t i2=0; i2 < tensor->dims[1]; ++i2) {
                putchar('\t');
                for (int64_t i1=0; i1 < tensor->dims[0]; ++i1) {
                    // TODO: dtype check
                    float x = buf[i3*tensor->dims[1]*tensor->dims[0] + i2*tensor->dims[0] + i1];
                    char fmt_buf[128];
                    msml__fmt_f64(MSML_FMT_G14, x, fmt_buf);
                    printf("%s ", fmt_buf);
                }
                putchar('\n');
            }
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

const int64_t* msml_tensor_dims(const msml_tensor_t* tensor) {
    return tensor->dims;
}

const int64_t* msml_tensor_strides(const msml_tensor_t* tensor) {
    return tensor->strides;
}

msml_dtype_t msml_tensor_dtype(const msml_tensor_t* tensor) {
    return tensor->dtype;
}

void* msml_tensor_buf(const msml_tensor_t* tensor) {
    return tensor->buf.u8;
}

int64_t msml_tensor_buf_size(const msml_tensor_t* tensor) {
    return tensor->buf_size;
}

int64_t msml_tensor_buf_len(const msml_tensor_t* tensor) {
    return tensor->buf_size / (int64_t)msml_dtype_info[tensor->dtype].size;
}

int64_t msml_tensor_num_rows(const msml_tensor_t* tensor) {
    int64_t rows=tensor->dims[1];
    for (int64_t i=2; i < MSML_MAX_DIMS; ++i)
        rows *= tensor->dims[i];
    return rows;
}

int64_t msml_tensor_num_cols(const msml_tensor_t* tensor) {
    return tensor->dims[0];
}

bool msml_tensor_is_scalar(const msml_tensor_t* tensor) {
    for (int i=0; i < MSML_MAX_DIMS; ++i)
        if (tensor->dims[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_vector(const msml_tensor_t* tensor) {
    for (int i=1; i < MSML_MAX_DIMS; ++i)
        if (tensor->dims[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_matrix(const msml_tensor_t* tensor) {
    for (int i=2; i < MSML_MAX_DIMS; ++i)
        if (tensor->dims[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_higher_order_3d(const msml_tensor_t* tensor) {
    for (int i=3; i < MSML_MAX_DIMS; ++i)
        if (tensor->dims[i] != 1)
            return false;
    return true;
}

void msml_tensor_virtual_to_physical_index(const msml_tensor_t* tensor, int64_t v_idx, int64_t(*p_idx)[MSML_MAX_DIMS]) {
    msml__load_local_storage_group(tensor, d, dims);
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
    return *tensor->strides == (int64_t)msml_dtype_info[tensor->dtype].size;
}

float msml_tensor_get_scalar_physical_index(const msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    const uint8_t* dst = msml__resolve_physical_ptr(tensor, d0, d1, d2, d3);
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: return *(float*)dst;
        default: msml_panic("Unsupported data type: %s", msml_dtype_info[tensor->dtype].name);
    }
}

void msml_tensor_set_scalar_physical_index(msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3, float x) {
    uint8_t* dst = msml__resolve_physical_ptr(tensor, d0, d1, d2, d3);
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: *(float*)dst = x; break;
        default: msml_panic("Unsupported data type: %s", msml_dtype_info[tensor->dtype].name);
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
            return tensor->buf.f32[v_idx];
        default:
            msml_panic("Unsupported data type: %s", msml_dtype_info[tensor->dtype].name);
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
            tensor->buf.f32[v_idx] = x;
            break;
        default:
            msml_panic("Unsupported data type: %s", msml_dtype_info[tensor->dtype].name);
    }
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
    memcpy(header.dims, tensor->dims, sizeof(header.dims));
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
    msml_assert2(p+n*msml_dtype_info[tensor->dtype].size <= end);
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            const float* t_p = tensor->buf.f32;
            const float* t_end = tensor->buf.f32 + n;
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
            msml_panic("Unsupported data type: %s", msml_dtype_info[tensor->dtype].name);
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
    float* dst = tensor->buf.f32;
    size_t n = width*height*channels;
    msml_assert(n == msml_tensor_buf_len(tensor), "Buffer size mismatch: %zu != %lld", n, msml_tensor_buf_len(tensor));
    for (size_t i=0; i < n; ++i)
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
    const int64_t* dims = msml_tensor_dims(tensor);
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
        float f32_u8 = msml_min(msml_max(tensor->buf.f32[i], 0.0f), 1.0f) * 255.0f;
        image_data[i] = (uint8_t)f32_u8;
    }
    int result = stbi_write_jpg(file_path, (int)width, (int)height, (int)channels, image_data, 100);
    msml_assert(result, "Failed to save tensor to image: %s", file_path);
    (*tensor->ctx->alloc_fn)(image_data, 0); /* Free image data */
    msml_log_info("Saved tensor to image: %s, width: %d, height: %d, channels: %d", file_path, (int)width, (int)height, (int)channels);
#else
    msml_panic("Image support is disabled. MSML must be compiled with MSML_ENABLE_IMAGE_SUPPORT defined.");
#endif
}
