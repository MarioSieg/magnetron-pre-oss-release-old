# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# MSML - Single header STB-style machine learning library in C99.
# MIT licensed.
# Python bindings for MSML.
import math
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
    msml_tensor_t* msml_tensor_isomorphic_clone(msml_tensor_t* tensor);
    msml_tensor_t* msml_tensor_deep_clone(msml_tensor_t* tensor);
    void msml_tensor_copy_buffer_from(msml_tensor_t* tensor, const void* data, size_t size);
    void msml_tensor_fill_zero(msml_tensor_t* tensor);
    void msml_tensor_fill_one(msml_tensor_t* tensor);
    void msml_tensor_fill(msml_tensor_t* tensor, float x);
    void msml_tensor_fill_random(msml_tensor_t* tensor, float min, float max);
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
    float msml_tensor_get_scalar_physical_index(const msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3);
    void msml_tensor_set_scalar_physical_index(msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3, float x);
    float msml_tensor_get_scalar_virtual_index(const msml_tensor_t* tensor, int64_t v_idx);
    void msml_tensor_set_scalar_virtual_index(msml_tensor_t* tensor, int64_t v_idx, float x);
    void msml_tensor_save(const msml_tensor_t* tensor, const char* file_name);
    msml_tensor_t* msml_tensor_load(msml_ctx_t* ctx, const char* file_name);

    msml_tensor_t* msml_tensor_create_from_image(msml_ctx_t* ctx, const char* file_path, msml_desired_color_channels_t channels, uint32_t resize_width, uint32_t resize_height);
    void msml_tensor_save_to_image(const msml_tensor_t* tensor, const char* file_path);
''')


# Define Python wrapper classes

class Context:
    """Manages the MSML context and tensor lifecycles."""

    def __init__(self):
        self.ctx = C.msml_ctx_create(ffi.NULL)
        # Use weak references to manage the lifecycle of tensors, as they are owned by the context
        self.allocated_tensors = weakref.WeakSet()

    def __del__(self):
        """Ensure tensors are cleaned up when the context is destroyed."""
        for tensor in list(self.allocated_tensors):
            tensor.__del__()
        C.msml_ctx_destroy(self.ctx)


class DType(Enum):
    """Enumerates the supported data types for tensors."""
    F32 = 0


class DesiredColorChannels(Enum):
    """Enumerates the desired color channels when loading images."""
    AUTO = 0  # Automatically determine the number of color channels
    GRAY = 1  # Grayscale F32
    GRAY_A = 2  # Grayscale F32 with alpha F32
    RGB = 3  # R32G32B32
    RGBA = 4  # R32G32B32A32


class Tensor:
    """Represents a tensor in the MSML library."""

    def __init__(self, internal_instance=None):
        self.tensor = internal_instance
        ctx.allocated_tensors.add(self)  # Add the tensor to the context's weakly referenced set

    def __del__(self):
        """Destructor to release tensor resources."""
        if self.tensor is not None:
            self.tensor = None

    def _create_internal(self, ctx: Context, name: str | None, dtype: DType, dims: list[int]):
        assert 0 < len(dims) <= MAX_DIMS, 'Number of dimensions exceeds maximum'
        for dim in dims:
            assert DIM_MAX > dim > 0, 'Invalid dimension size'
        self.tensor = C.msml_tensor_create(ctx.ctx, dtype.value, dims, len(dims))
        if name is not None:
            self.set_name(name)

    def fill_zeros(self):
        """Sets all elements of the tensor to zero."""
        C.msml_tensor_fill_zero(self.tensor)

    def fill_ones(self):
        """Sets all elements of the tensor to one."""
        C.msml_tensor_fill_one(self.tensor)

    def fill(self, x: float):
        """Sets all elements of the tensor to x."""
        C.msml_tensor_fill(self.tensor, x)

    def fill_random(self, r_min: float = 0.0, r_max: float = 1.0):
        assert r_min < r_max
        """Sets all elements of the tensor to random values within [min, max]"""
        C.msml_tensor_fill_random(self.tensor, r_min, r_max)

    def print(self, with_data: bool):
        """Prints the tensor metadata and optionally its data."""
        C.msml_tensor_print(self.tensor, with_data)

    def set_name(self, name: str):
        """Sets a name for the tensor."""
        C.msml_tensor_set_name(self.tensor, bytes(name, 'utf-8'))

    def get_name(self) -> str:
        """Returns the name of the tensor."""
        return ffi.string(C.msml_tensor_get_name(self.tensor)).decode('utf-8')

    def rank(self) -> int:
        """Returns the rank (number of dimensions) of the tensor."""
        return C.msml_tensor_rank(self.tensor)

    def dims(self) -> list[int]:
        """Returns the dimensions of the tensor."""
        ptr = C.msml_tensor_dims(self.tensor)
        return [ptr[i] for i in range(MAX_DIMS)]

    def strides(self) -> list[int]:
        """Returns the strides of the tensor."""
        ptr = C.msml_tensor_strides(self.tensor)
        return [ptr[i] for i in range(MAX_DIMS)]

    def dtype(self) -> int:
        """Returns the data type of the tensor."""
        return C.msml_tensor_dtype(self.tensor)

    def buf(self) -> ffi.CData:
        """Returns the raw buffer of the tensor."""
        return C.msml_tensor_buf(self.tensor)

    def buf_size(self) -> int:
        """Returns the size of the tensor buffer in bytes."""
        return C.msml_tensor_buf_size(self.tensor)

    def num_rows(self) -> int:
        """Returns the number of rows in the tensor, assuming it's a matrix."""
        return C.msml_tensor_num_rows(self.tensor)

    def num_cols(self) -> int:
        """Returns the number of columns in the tensor, assuming it's a matrix."""
        return C.msml_tensor_num_cols(self.tensor)

    def is_scalar(self) -> bool:
        """Checks if the tensor is a scalar (0D tensor)."""
        return C.msml_tensor_is_scalar(self.tensor)

    def is_vector(self) -> bool:
        """Checks if the tensor is a vector (1D tensor)."""
        return C.msml_tensor_is_vector(self.tensor)

    def is_matrix(self) -> bool:
        """Checks if the tensor is a matrix (2D tensor)."""
        return C.msml_tensor_is_matrix(self.tensor)

    def is_higher_order_3d(self) -> bool:
        """Checks if the tensor is a higher-order 3D tensor."""
        return C.msml_tensor_is_higher_order_3d(self.tensor)

    def virtual_to_physical_index(self, v_idx: int) -> list[int]:
        """Converts a virtual index to a physical index."""
        p_idx = ffi.new(f'int64_t[{MAX_DIMS}]')
        C.msml_tensor_virtual_to_physical_index(self.tensor, v_idx, p_idx)
        return list(p_idx)

    def physical_to_virtual_index(self, p_idx: list[int]) -> int:
        """Converts a physical index to a virtual index."""
        assert len(p_idx) == MAX_DIMS
        return C.msml_tensor_physical_to_virtual_index(self.tensor, p_idx)

    def is_contiguous(self) -> bool:
        """Checks if the tensor is contiguous in memory."""
        return C.msml_tensor_is_contiguous(self.tensor)

    def get_scalar_physical_index(self, d0: int, d1: int, d2: int, d3: int) -> float:
        """Returns the scalar value at a physical index."""
        return C.msml_tensor_get_scalar_physical_index(self.tensor, d0, d1, d2, d3)

    def set_scalar_physical_index(self, d0: int, d1: int, d2: int, d3: int, x: float):
        """Sets the scalar value at a physical index."""
        C.msml_tensor_set_scalar_physical_index(self.tensor, d0, d1, d2, d3, x)

    def get_scalar_virtual_index(self, v_idx: int) -> float:
        """Returns the scalar value at a virtual index."""
        return C.msml_tensor_get_scalar_virtual_index(self.tensor, v_idx)

    def set_scalar_virtual_index(self, v_idx: int, x: float):
        """Sets the scalar value at a virtual index."""
        C.msml_tensor_set_scalar_virtual_index(self.tensor, v_idx, x)

    def image_width(self) -> int:
        """Returns the width of the image tensor. (Equals to the first dimension)"""
        return self.dims()[0]

    def image_height(self) -> int:
        """Returns the height of the image tensor. (Equals to the second dimension)"""
        return self.dims()[1]

    def image_channels(self) -> int:
        """Returns the number of color channels in the image tensor. (Equals to the third dimension)"""
        return self.dims()[2]

    def save(self, file_path: str):
        """Saves to tensor to a binary MSML file"""
        if not file_path.endswith('.msml'):
            file_path += '.msml'
        C.msml_tensor_save(self.tensor, bytes(file_path, 'utf-8'))

    def save_to_image(self, file_path: str):
        """Saves the tensor as an JPG image to a file."""
        assert self.rank() == 3, 'Tensor must be a 3D image tensor'
        channels: int = self.image_channels()
        assert channels in (1, 3, 4), 'Invalid number of color channels'
        C.msml_tensor_save_to_image(self.tensor, bytes(file_path, 'utf-8'))

    @staticmethod
    def empty(ctx: Context, dtype: DType, name: str | None, dims: list[int]):
        """Creates an empty tensor, with uninitialized data."""
        tensor = Tensor(None)
        tensor._create_internal(ctx, name, dtype, dims)
        return tensor

    @staticmethod
    def isomorphic_clone(tensor):
        """Create new empty tensor with same shape as input, but without cloning data."""
        return Tensor(C.msml_tensor_isomorphic_clone(tensor.tensor))

    @staticmethod
    def deep_clone(tensor):
        """Create new tensor with same shape and data as input."""
        return Tensor(C.msml_tensor_deep_clone(tensor.tensor))

    @staticmethod
    def zeros(ctx: Context, dtype: DType, name: str | None, dims: list[int]):
        """Creates a tensor filled with zeros."""
        tensor = Tensor(None)
        tensor._create_internal(ctx, name, dtype, dims)
        tensor.fill_zeros()
        return tensor

    @staticmethod
    def full(ctx: Context, dtype: DType, name: str | None, dims: list[int], fill_value: float):
        """Creates a tensor filled with a constant value."""
        tensor = Tensor(None)
        tensor._create_internal(ctx, name, dtype, dims)
        if fill_value == 0.0:
            tensor.fill_zeros()
        elif fill_value == 1.0:
            tensor.fill_ones()
        else:
            tensor.fill(fill_value)
        return tensor

    @staticmethod
    def random(ctx: Context, dtype: DType, name: str | None, dims: list[int], r_min: float = 0.0, r_max: float = 1.0):
        """Creates a tensor filled with zeros."""
        tensor = Tensor(None)
        tensor._create_internal(ctx, name, dtype, dims)
        tensor.fill_random(r_min, r_max)
        return tensor

    @staticmethod
    def load(ctx: Context, file_path: str):
        assert file_path.endswith('.msml'), 'File must be a MSML file'
        """Loads a tensor from a binary MSML file."""
        instance = C.msml_tensor_load(ctx.ctx, bytes(file_path, 'utf-8'))
        return Tensor(internal_instance=instance)

    @staticmethod
    def from_image(ctx: Context,
                   name: str | None,
                   file_path: str,
                   desired_color_channels=DesiredColorChannels.AUTO,
                   resize_to_dims: tuple[int, int] = (0, 0)):
        """Loads an image from a file and creates a tensor from it."""
        instance = C.msml_tensor_create_from_image(ctx.ctx, bytes(file_path, 'utf-8'), desired_color_channels.value,
                                                   resize_to_dims[0], resize_to_dims[1])
        return Tensor(internal_instance=instance)


ctx = Context()
img = Tensor.from_image(ctx, 'Cat', '../test_data/car.jpg')
for i in range(img.image_width()):
    img.set_scalar_virtual_index(i * 3 + 0, 0)
    img.set_scalar_virtual_index(i * 3 + 1, 0)
    img.set_scalar_virtual_index(i * 3 + 2, 1)
img.print(False)
img.save_to_image('cat_out.jpg')
img.save('cat.msml')
img2 = Tensor.random(ctx, DType.F32, 'Random Image', [4, 4, 3])
img2.save_to_image('random.jpg')
img2.save('random.msml')
