# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# MSML - Single header STB-style machine learning library in C99.
# MIT licensed.
# Python bindings for MSML.

import weakref
from cffi import FFI
from enum import Enum

# Load shared library

MSML_LIB = '../bin/debug/libmsml.dylib'
ffi = FFI()
C = ffi.dlopen(MSML_LIB)

# Define constants
MAX_DIMS = 4
DIM_MAX = 0x7fffffffffffffff

# Define C types - keep in sync carefully with the C header file, only include what is needed
ffi.cdef(f'''
    typedef struct msml_ctx_info_t msml_ctx_info_t;
    typedef struct msml_ctx_t msml_ctx_t;
    typedef int msml_dtype_t;
    typedef int msml_desired_color_channels_t;
    typedef struct msml_tensor_t msml_tensor_t;
    
    msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info);
    void msml_ctx_destroy(msml_ctx_t* ctx);
    
    msml_tensor_t* msml_tensor_create(msml_ctx_t* ctx, msml_dtype_t type, const int64_t* dims, int64_t rank);
    msml_tensor_t* msml_tensor_create_1d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1);
    msml_tensor_t* msml_tensor_create_2d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2);
    msml_tensor_t* msml_tensor_create_3d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3);
    msml_tensor_t* msml_tensor_create_4d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4);
    void msml_tensor_copy_buffer_from(msml_tensor_t* tensor, const void* data, size_t size);
    void msml_tensor_set_zero(msml_tensor_t* tensor);
    void msml_tensor_set_one(msml_tensor_t* tensor);
    void msml_tensor_set(msml_tensor_t* tensor, float x);
    void msml_tensor_print(const msml_tensor_t* tensor, bool with_data);
    void msml_tensor_set_name(msml_tensor_t* tensor, const char* name);
    const char* msml_tensor_get_name(const msml_tensor_t* tensor);
    int64_t msml_tensor_rank(const msml_tensor_t* tensor);
    const int64_t* msml_tensor_dims(const msml_tensor_t* tensor);
    const int64_t* msml_tensor_strides(const msml_tensor_t* tensor);
    msml_dtype_t msml_tensor_dtype(const msml_tensor_t* tensor);
    void* msml_tensor_buf(const msml_tensor_t* tensor);
    int64_t msml_tensor_buf_size(const msml_tensor_t* tensor);
    int64_t msml_tensor_buf_len(const msml_tensor_t* tensor);
    int64_t msml_tensor_num_rows(const msml_tensor_t* tensor);
    int64_t msml_tensor_num_cols(const msml_tensor_t* tensor);
    bool msml_tensor_is_scalar(const msml_tensor_t* tensor);
    bool msml_tensor_is_vector(const msml_tensor_t* tensor);
    bool msml_tensor_is_matrix(const msml_tensor_t* tensor);
    bool msml_tensor_is_higher_order_3d(const msml_tensor_t* tensor);
    void msml_tensor_virtual_to_physical_index(const msml_tensor_t* tensor, int64_t v_idx, int64_t(*p_idx)[{MAX_DIMS}]);
    int64_t msml_tensor_physical_to_virtual_index(const msml_tensor_t* tensor, const int64_t (*p_idx)[{MAX_DIMS}]);
    bool msml_tensor_is_contiguous(const msml_tensor_t* tensor);
    
    msml_tensor_t* msml_tensor_create_from_image(msml_ctx_t* ctx, const char* file_path, msml_desired_color_channels_t channels, uint32_t resize_width, uint32_t resize_height);
    void msml_tensor_save_to_image(const msml_tensor_t* tensor, const char* file_path);
''')


# Define Python wrapper classes

class Context:
    def __init__(self):
        self.ctx = C.msml_ctx_create(ffi.NULL)
        # Use weak references to manage the lifecycle of tensors, as they are owned by the context
        self.allocated_tensors = weakref.WeakSet()

    def __del__(self):
        # Ensure tensors are cleaned up
        for tensor in list(self.allocated_tensors):
            tensor.__del__()
        C.msml_ctx_destroy(self.ctx)


class DType(Enum):
    F32 = 0


class DesiredColorChannels(Enum):
    AUTO = 0  # Automatically determine the number of color channels
    GRAY = 1  # Grayscale F32
    GRAY_A = 2  # Grayscale F32 with alpha F32
    RGB = 3  # R32G32B32
    RGBA = 4  # R32G32B32A32


class Tensor:
    def __init__(self, ctx: Context, dtype: DType, name: str | None, dims: list[int], internal_instance=None):
        if internal_instance is None:  # Create tensor from arguments if not instance provided
            assert 0 < len(dims) <= MAX_DIMS, 'Number of dimensions exceeds maximum'
            for dim in dims:
                assert DIM_MAX > dim > 0, 'Invalid dimension size'
            self.tensor = C.msml_tensor_create(ctx.ctx, dtype.value, dims, len(dims))
        else:  # If instance is provided, just assign it
            self.tensor = internal_instance
        if name is not None:
            self.set_name(name)
        ctx.allocated_tensors.add(self)  # Add the tensor to the context's weakly referenced set

    def __del__(self):
        if self.tensor is not None:
            self.tensor = None

    def set_zero(self):
        C.msml_tensor_set_zero(self.tensor)

    def set_one(self):
        C.msml_tensor_set_one(self.tensor)

    def set(self, x: float):
        C.msml_tensor_set(self.tensor, x)

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

    @staticmethod
    def empty(ctx: Context, dtype: DType, name: str | None, dims: list[int]):
        """Creates an empty tensor, with uninitialized data."""
        return Tensor(ctx, dtype, name, dims)

    @staticmethod
    def zeros(ctx: Context, dtype: DType, name: str | None, dims: list[int]):
        """Creates a tensor filled with zeros."""
        result = Tensor(ctx, dtype, name, dims)
        result.set_zero()
        return result

    @staticmethod
    def full(ctx: Context, dtype: DType, name: str | None, dims: list[int], fill_value: float):
        """Creates a tensor filled with a constant value."""
        result = Tensor(ctx, dtype, name, dims)
        if fill_value == 0.0:
            result.set_zero()
        elif fill_value == 1.0:
            result.set_one()
        else:
            result.set(fill_value)
        return result

    def image_width(self) -> int:
        return self.dims()[0]

    def image_height(self) -> int:
        return self.dims()[1]

    def image_channels(self) -> int:
        return self.dims()[2]

    @staticmethod
    def from_image(ctx: Context, name: str | None, file_path: str, desired_color_channels=DesiredColorChannels.AUTO,
                   resize_dims=(0, 0)):
        """Loads an image from a file and creates a tensor from it."""
        instance = C.msml_tensor_create_from_image(ctx.ctx, bytes(file_path, 'utf-8'), desired_color_channels.value,
                                                   resize_dims[0], resize_dims[1])
        return Tensor(ctx, DType.F32, name, [], internal_instance=instance)

    def save_to_image(self, file_path: str):
        assert self.rank() == 3, 'Tensor must be a 3D image tensor'
        channels: int = self.image_channels()
        assert channels == 1 or channels == 3 or channels == 4, 'Invalid number of color channels'
        C.msml_tensor_save_to_image(self.tensor, bytes(file_path, 'utf-8'))


ctx = Context()
img = Tensor.from_image(ctx, 'Cat', '../test_data/cat.jpg', resize_dims=(64, 64))
img.print(False)
img.save_to_image('cat_out.jpg')
