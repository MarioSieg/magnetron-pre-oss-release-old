# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# MSML - Single header STB-style machine learning library in C99.
# MIT licensed.
# Python bindings for MSML.

from cffi import FFI
from enum import Enum

# Load shared library

MSML_LIB = '../bin/debug/libmsml.dylib'
ffi = FFI()
C = ffi.dlopen(MSML_LIB)

# Define constants
MAX_DIMS = 4
DIM_MAX = 0x7fffffffffffffff

# Define C types
ffi.cdef(f'''
    typedef struct msml_ctx_info_t msml_ctx_info_t;
    typedef struct msml_ctx_t msml_ctx_t;
    typedef int msml_dtype_t;
    typedef struct msml_tensor_t msml_tensor_t;
    
    msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info);
    void msml_ctx_destroy(msml_ctx_t* ctx);
    
    msml_tensor_t* msml_tensor_create(msml_ctx_t* ctx, msml_dtype_t type, const int64_t* dims, int64_t rank);
    msml_tensor_t* msml_tensor_create_1d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1);
    msml_tensor_t* msml_tensor_create_2d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2);
    msml_tensor_t* msml_tensor_create_3d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3);
    msml_tensor_t* msml_tensor_create_4d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4);
    void msml_tensor_print(const msml_tensor_t* tensor, bool with_data);
    void msml_tensor_set_name(msml_tensor_t* tensor, const char* name);
    const char* msml_tensor_get_name(const msml_tensor_t* tensor);
    int64_t msml_tensor_rank(const msml_tensor_t* tensor);
    const int64_t* msml_tensor_dims(const msml_tensor_t* tensor);
    const int64_t* msml_tensor_strides(const msml_tensor_t* tensor);
    msml_dtype_t msml_tensor_dtype(const msml_tensor_t* tensor);
    void* msml_tensor_buf(const msml_tensor_t* tensor);
    int64_t msml_tensor_buf_size(const msml_tensor_t* tensor);
    int64_t msml_tensor_num_rows(const msml_tensor_t* tensor);
    int64_t msml_tensor_num_cols(const msml_tensor_t* tensor);
    bool msml_tensor_is_scalar(const msml_tensor_t* tensor);
    bool msml_tensor_is_vector(const msml_tensor_t* tensor);
    bool msml_tensor_is_matrix(const msml_tensor_t* tensor);
    bool msml_tensor_is_higher_order_3d(const msml_tensor_t* tensor);
    void msml_tensor_virtual_to_physical_index(const msml_tensor_t* tensor, int64_t v_idx, int64_t(*p_idx)[{MAX_DIMS}]);
    int64_t msml_tensor_physical_to_virtual_index(const msml_tensor_t* tensor, const int64_t (*p_idx)[{MAX_DIMS}]);
    bool msml_tensor_is_contiguous(const msml_tensor_t* tensor);
''')


# Define Python wrapper classes

class Context:
    def __init__(self):
        self.ctx = C.msml_ctx_create(ffi.NULL)

    def __del__(self):

        C.msml_ctx_destroy(self.ctx)


class DType(Enum):
    F32 = 0


class Tensor:
    def __init__(self, ctx: Context, dtype: DType, name: str | None, dims: list[int]):
        assert 0 < len(dims) <= MAX_DIMS, 'Number of dimensions exceeds maximum'
        for dim in dims:
            assert DIM_MAX > dim > 0, 'Invalid dimension size'
        self.tensor = C.msml_tensor_create(ctx.ctx, dtype.value, dims, len(dims))
        if name is not None:
            self.set_name(name)

    def print(self, with_data: bool):
        C.msml_tensor_print(self.tensor, with_data)

    def set_name(self, name: str):
        C.msml_tensor_set_name(self.tensor, bytes(name, 'utf-8'))

    def get_name(self) -> str:
        return ffi.string(C.msml_tensor_get_name(self.tensor)).decode('utf-8')

    def rank(self) -> int:
        return C.msml_tensor_rank(self.tensor)

    def dims(self) -> list[int]:
        ptr = C.msml_tensor_dims(self.tensor)
        return [ptr[i] for i in range(MAX_DIMS)]

    def strides(self) -> list[int]:
        ptr = C.msml_tensor_strides(self.tensor)
        return [ptr[i] for i in range(MAX_DIMS)]

    def dtype(self) -> int:
        return C.msml_tensor_dtype(self.tensor)

    def buf(self) -> ffi.CData:
        return C.msml_tensor_buf(self.tensor)

    def buf_size(self) -> int:
        return C.msml_tensor_buf_size(self.tensor)

    def num_rows(self) -> int:
        return C.msml_tensor_num_rows(self.tensor)

    def num_cols(self) -> int:
        return C.msml_tensor_num_cols(self.tensor)

    def is_scalar(self) -> bool:
        return C.msml_tensor_is_scalar(self.tensor)

    def is_vector(self) -> bool:
        return C.msml_tensor_is_vector(self.tensor)

    def is_matrix(self) -> bool:
        return C.msml_tensor_is_matrix(self.tensor)

    def is_higher_order_3d(self) -> bool:
        return C.msml_tensor_is_higher_order_3d(self.tensor)

    def virtual_to_physical_index(self, v_idx: int) -> list[int]:
        p_idx = ffi.new(f'int64_t[{MAX_DIMS}]')
        C.msml_tensor_virtual_to_physical_index(self.tensor, v_idx, p_idx)
        return list(p_idx)

    def physical_to_virtual_index(self, p_idx: list[int]) -> int:
        assert len(p_idx) == MAX_DIMS
        return C.msml_tensor_physical_to_virtual_index(self.tensor, p_idx)

    def is_contiguous(self) -> bool:
        return C.msml_tensor_is_contiguous(self.tensor)


ctx = Context()
grad = Tensor(ctx, DType.F32, 'Gradients', [4, 4])
grad.print(True)
