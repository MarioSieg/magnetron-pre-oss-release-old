# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# MSML - Single header STB-style machine learning library in C99.
# MIT licensed.
# Implements the core functionality of the MSML Python bindings. Requires the MSML shared library.

import platform
import weakref
import random

from cffi import FFI
from enum import Enum
from os.path import isfile
from ctypes.util import find_library

# Load shared library

BUILD_DIR = 'release'

msml_lib_locations: list[str] = []

if platform.system() == 'Windows':
    msml_lib_locations.append(f'../bin/{BUILD_DIR}/msml.dll')
elif platform.system() == 'Linux':
    msml_lib_locations.append(f'../../bin/{BUILD_DIR}/libmsml.so')
elif platform.system() == 'Darwin':
    msml_lib_locations.append(f'../bin/{BUILD_DIR}/libmsml.dylib')
else:
    raise RuntimeError('Unsupported platform')

MSML_LIB_PATH: str | None = None
for loc in msml_lib_locations:  # Try to find the shared library in the locations (mostly used for debug builds)
    if isfile(loc):
        MSML_LIB_PATH = loc
        break
    elif isfile(f'../{loc}'):  # Try to find the shared library in the parent directory
        MSML_LIB_PATH = f'../{loc}'
        break
if MSML_LIB_PATH is None:  # If not found, try to find the shared library in the system paths
    MSML_LIB_PATH = find_library('msml')
assert MSML_LIB_PATH is not None, 'MSML shared library not found'

ffi = FFI()
ffi.dlopen('m')
C = ffi.dlopen(MSML_LIB_PATH)

# Define constants
MAX_DIMS = 4
MAX_ARG_TENSORS = 2
DIM_MAX = 0x7fffffffffffffff

# Define C types - keep in sync carefully with the C header file, only include what is needed
ffi.cdef(f'''
    typedef struct msml_ctx_info_t msml_ctx_info_t;
    typedef struct msml_ctx_t msml_ctx_t;
    
    typedef int msml_prng_algorithm_t;
    typedef int msml_dtype_t;
    typedef int msml_desired_color_channels_t;
    typedef int msml_op_t;
    
    typedef struct msml_tensor_t msml_tensor_t;

    msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info);
    msml_ctx_t* msml_ctx_create2(size_t pool_chunk_size);
    size_t msml_ctx_total_memory(const msml_ctx_t* ctx);
    msml_prng_algorithm_t msml_ctx_get_prng_algorithm(const msml_ctx_t* ctx);
    void msml_ctx_set_prng_algorithm(msml_ctx_t* ctx, msml_prng_algorithm_t algorithm, uint64_t seed);
    void msml_ctx_destroy(msml_ctx_t* ctx);
    
    const char* msml_op_get_name(msml_op_t op);
    const char* msml_op_get_mnemonic(msml_op_t op);
    uint8_t msml_op_get_argcount(msml_op_t op);

    msml_ctx_t* msml_tensor_get_ctx(const msml_tensor_t* tensor);
    msml_tensor_t* msml_tensor_create(msml_ctx_t* ctx, msml_dtype_t type, const int64_t* shape, int64_t rank);
    msml_tensor_t* msml_tensor_create_1d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1);
    msml_tensor_t* msml_tensor_create_2d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2);
    msml_tensor_t* msml_tensor_create_3d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3);
    msml_tensor_t* msml_tensor_create_4d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4);
    msml_tensor_t* msml_tensor_get_arg(const msml_tensor_t* tensor, size_t slot);
    void msml_tensor_set_arg(msml_tensor_t* tensor, size_t slot, msml_tensor_t* arg);
    msml_op_t msml_tensor_get_op(const msml_tensor_t* tensor);
    void msml_tensor_set_op(msml_tensor_t* tensor, msml_op_t op);
    msml_tensor_t* msml_tensor_isomorphic_clone(msml_tensor_t* tensor);
    msml_tensor_t* msml_tensor_deep_clone(msml_tensor_t* tensor);
    void msml_tensor_copy_buffer_from(msml_tensor_t* tensor, const void* data, size_t size);
    void msml_tensor_fill(msml_tensor_t* tensor, float x);
    void msml_tensor_fill_random(msml_tensor_t* tensor, float min, float max);
    size_t msml_tensor_get_memory_usage(const msml_tensor_t* tensor);
    void msml_tensor_print(const msml_tensor_t* tensor, bool with_data);
    void msml_tensor_set_name(msml_tensor_t* tensor, const char* name);
    const char* msml_tensor_get_name(const msml_tensor_t* tensor);
    int64_t msml_tensor_rank(const msml_tensor_t* tensor);
    const int64_t* msml_tensor_shape(const msml_tensor_t* tensor);
    const int64_t* msml_tensor_strides(const msml_tensor_t* tensor);
    msml_dtype_t msml_tensor_dtype(const msml_tensor_t* tensor);
    void* msml_tensor_buf(const msml_tensor_t* tensor);
    float* msml_tensor_buf_f32(const msml_tensor_t* tensor);
    int64_t msml_tensor_buf_size(const msml_tensor_t* tensor);
    int64_t msml_tensor_buf_len(const msml_tensor_t* tensor);
    int64_t msml_tensor_num_rows(const msml_tensor_t* tensor);
    int64_t msml_tensor_num_cols(const msml_tensor_t* tensor);
    bool msml_tensor_is_scalar(const msml_tensor_t* tensor);
    bool msml_tensor_is_vector(const msml_tensor_t* tensor);
    bool msml_tensor_is_matrix(const msml_tensor_t* tensor);
    bool msml_tensor_is_higher_order_3d(const msml_tensor_t* tensor);
    bool msml_tensor_is_shape_eq(const msml_tensor_t* a, const msml_tensor_t* b);
    bool msml_tensor_are_strides_eq(const msml_tensor_t* a, const msml_tensor_t* b);
    bool msml_tensor_can_broadcast(const msml_tensor_t* a, const msml_tensor_t* b);
    void msml_tensor_virtual_to_physical_index(const msml_tensor_t* tensor, int64_t v_idx, int64_t(*p_idx)[{MAX_DIMS}]);
    int64_t msml_tensor_physical_to_virtual_index(const msml_tensor_t* tensor, const int64_t (*p_idx)[{MAX_DIMS}]);
    bool msml_tensor_is_contiguous(const msml_tensor_t* tensor);
    float msml_tensor_get_scalar_physical_index(const msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3);
    void msml_tensor_set_scalar_physical_index(msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3, float x);
    float msml_tensor_get_scalar_virtual_index(const msml_tensor_t* tensor, int64_t v_idx);
    void msml_tensor_set_scalar_virtual_index(msml_tensor_t* tensor, int64_t v_idx, float x);
    bool msml_tensor_eq(const msml_tensor_t* a, const msml_tensor_t* b);
    bool msml_tensor_isclose(const msml_tensor_t* a, const msml_tensor_t* b, float eps, double* percent_eq);
    void msml_tensor_evaluate(msml_tensor_t* tensor);
    
    void msml_tensor_save(const msml_tensor_t* tensor, const char* file_name);
    msml_tensor_t* msml_tensor_load(msml_ctx_t* ctx, const char* file_name);
    msml_tensor_t* msml_tensor_create_from_image(msml_ctx_t* ctx, const char* file_path, msml_desired_color_channels_t channels, uint32_t resize_width, uint32_t resize_height);
    void msml_tensor_save_to_image(const msml_tensor_t* tensor, const char* file_path);
''')


# Define Python wrapper classes

def humanize_memory_size(size: int) -> str:
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    unit = 0
    while size >= 1024 and unit < len(units) - 1:
        size /= 1024
        unit += 1
    return f'{size:.2f} {units[unit]}'


class PRNGAlgorithm(Enum):
    MERSENNE_TWISTER = 0  # Default - Mersenne Twister Generator
    PCG = 1  # Permuted Congruential Generator

    def __str__(self) -> str:
        match self:
            case PRNGAlgorithm.MERSENNE_TWISTER:
                return 'Mersenne Twister'
            case PRNGAlgorithm.PCG:
                return 'Permuted Congruential Generator'


class DType(Enum):
    """Enumerates the supported data types for tensors."""
    F32 = 0

    def __str__(self) -> str:
        match self:
            case DType.F32:
                return 'F32'


class DesiredColorChannels(Enum):
    """Enumerates the desired color channels when loading images."""
    AUTO = 0  # Automatically determine the number of color channels
    GRAY = 1  # Grayscale F32
    GRAY_A = 2  # Grayscale F32 with alpha F32
    RGB = 3  # R32G32B32
    RGBA = 4  # R32G32B32A32

    def __str__(self) -> str:
        match self:
            case DesiredColorChannels.AUTO:
                return 'Auto'
            case DesiredColorChannels.GRAY:
                return 'Grayscale'
            case DesiredColorChannels.GRAY_A:
                return 'Grayscale with Alpha'
            case DesiredColorChannels.RGB:
                return 'RGB'
            case DesiredColorChannels.RGBA:
                return 'RGBA'


class Operation(Enum):
    """A"""
    NOP = 0
    SOFTMAX = 1
    SOFTMAX_DV = 2
    SIGMOID = 3
    SIGMOID_DV = 4
    SILU = 5
    SILU_DV = 6
    TANH = 7
    TANH_DV = 8
    RELU = 9
    RELU_DV = 10
    GELU = 11
    GELU_DV = 12
    ADD = 13
    SUB = 14
    MUL = 15
    DIV = 16
    MATMUL = 17

    _COUNT = MATMUL + 1

    @property
    def name(self) -> str:
        assert self.value < self._COUNT.value
        return ffi.string(C.msml_op_get_name(self.value)).decode('utf-8')

    @property
    def mnemonic(self) -> str:
        assert self.value < self._COUNT.value
        return ffi.string(C.msml_op_get_mnemonic(self.value)).decode('utf-8')

    @property
    def argument_count(self) -> int:
        assert self.value < self._COUNT.value
        return C.msml_op_get_argcount(self.value)

    @property
    def is_unary(self) -> bool:
        return self.argument_count == 1

    @property
    def is_binary(self) -> bool:
        return self.argument_count == 2


class Context:
    """Manages the MSML context and tensor lifecycles."""

    def __init__(self, pool_chunk_size: int = 2 * (1 << 30)):  # Pool chunk size. Default: 2GiB
        self.ctx = C.msml_ctx_create2(pool_chunk_size)
        # Use weak references to manage the lifecycle of tensors, as they are owned by the context
        self.allocated_tensors = weakref.WeakSet()

    @property
    def total_memory(self) -> int:
        """Returns the total memory allocated in the context in bytes."""
        return C.msml_ctx_total_memory(self.ctx)

    @property
    def prng_algorithm(self) -> PRNGAlgorithm:
        """Returns the PRNG algorithm used by the context."""
        return PRNGAlgorithm(C.msml_ctx_get_prng_algorithm(self.ctx))

    @prng_algorithm.setter
    def prng_algorithm(self, algorithm: PRNGAlgorithm):
        """Sets the PRNG algorithm and seed for the context."""
        C.msml_ctx_set_prng_algorithm(self.ctx, algorithm.value, random.randint(0, 1 << 63))

    def __del__(self):
        """Ensure tensors are cleaned up when the context is destroyed."""
        for tensor in list(self.allocated_tensors):
            tensor.__del__()
        C.msml_ctx_destroy(self.ctx)


class Tensor:
    """Represents a tensor in the MSML library."""

    def __init__(self, internal_instance: ffi.CData | None = None) -> None:
        self.tensor = internal_instance
        #ctx.allocated_tensors.add(self)  # Add the tensor to the context's weakly referenced set

    def __del__(self) -> None:
        """Destructor to release tensor resources."""
        if self.tensor is not None:
            self.tensor = None

    def _create_internal(self, ctx: Context, shape: list[int], dtype: DType = DType.F32,
                         name: str | None = None) -> None:
        assert 0 < len(shape) <= MAX_DIMS, 'Number of dimensions exceeds maximum'
        for dim in shape:
            assert DIM_MAX > dim > 0, 'Invalid dimension size'
        self.tensor = C.msml_tensor_create(ctx.ctx, dtype.value, shape, len(shape))
        if name is not None:
            self.name = name

    def get_arg(self, slot: int) -> 'Tensor':
        return Tensor(C.msml_tensor_get_arg(self.tensor, slot))

    def set_arg(self, slot: int, tensor: 'Tensor') -> None:
        C.msml_tensor_set_arg(self.tensor, slot, tensor.tensor)

    def get_op(self) -> Operation:
        return Operation(C.msml_tensor_get_op(self.tensor))

    def set_op(self, op: Operation) -> None:
        C.msml_tensor_set_op(self.tensor, op.value)

    def set_op_with_args(self, op: Operation, *args) -> None:
        assert len(args) == op.argument_count, 'Argument count does not match required argument count for operation'
        for i in range(0, len(args)):
            assert isinstance(args[i], Tensor)
            self.set_arg(i, args[i])
        self.set_op(op)

    def eval(self) -> None:
        C.msml_tensor_evaluate(self.tensor)

    def fill(self, x: float) -> None:
        """Sets all elements of the tensor to x."""
        C.msml_tensor_fill(self.tensor, x)

    def fill_random(self, interval: (float, float) = (0.0, 1.0)) -> None:
        assert interval[0] < interval[1]
        """Sets all elements of the tensor to random values within [min, max]"""
        C.msml_tensor_fill_random(self.tensor, interval[0], interval[1])

    def print(self, with_data: bool) -> None:
        """Prints the tensor metadata and optionally its data."""
        C.msml_tensor_print(self.tensor, with_data)

    @property
    def name(self) -> str:
        """Returns the name of the tensor."""
        return ffi.string(C.msml_tensor_get_name(self.tensor)).decode('utf-8')

    @name.setter
    def name(self, name: str) -> None:
        """Sets a name for the tensor."""
        C.msml_tensor_set_name(self.tensor, bytes(name, 'utf-8'))

    @property
    def rank(self) -> int:
        """Returns the rank (number of dimensions) of the tensor."""
        return C.msml_tensor_rank(self.tensor)

    @property
    def shape(self) -> list[int]:
        """Returns the dimensions of the tensor."""
        ptr = C.msml_tensor_shape(self.tensor)
        return [ptr[i] for i in range(self.rank)]

    @property
    def strides(self) -> list[int]:
        """Returns the strides of the tensor."""
        ptr = C.msml_tensor_strides(self.tensor)
        return [ptr[i] for i in range(self.rank)]

    @property
    def dtype(self) -> DType:
        """Returns the data type of the tensor."""
        return DType(C.msml_tensor_dtype(self.tensor))

    @property
    def buf_size(self) -> int:
        """Returns the size of the tensor buffer in bytes."""
        return C.msml_tensor_buf_size(self.tensor)

    @property
    def num_elements(self) -> int:
        """Returns the size of the tensor buffer in bytes."""
        return C.msml_tensor_buf_len(self.tensor)

    def f32_data(self) -> list[float]:
        """Returns the data of the tensor buffer as a list of floats."""
        return ffi.unpack(C.msml_tensor_buf_f32(self.tensor), self.buf_size)

    @property
    def num_rows(self) -> int:
        """Returns the number of rows in the tensor, assuming it's a matrix."""
        return C.msml_tensor_num_rows(self.tensor)

    @property
    def num_cols(self) -> int:
        """Returns the number of columns in the tensor, assuming it's a matrix."""
        return C.msml_tensor_num_cols(self.tensor)

    @property
    def is_scalar(self) -> bool:
        """Checks if the tensor is a scalar (0D tensor)."""
        return C.msml_tensor_is_scalar(self.tensor)

    @property
    def is_vector(self) -> bool:
        """Checks if the tensor is a vector (1D tensor)."""
        return C.msml_tensor_is_vector(self.tensor)

    @property
    def is_matrix(self) -> bool:
        """Checks if the tensor is a matrix (2D tensor)."""
        return C.msml_tensor_is_matrix(self.tensor)

    @property
    def is_higher_order_3d(self) -> bool:
        """Checks if the tensor is a higher-order 3D tensor."""
        return C.msml_tensor_is_higher_order_3d(self.tensor)

    def is_shape_eq(self, other: 'Tensor') -> bool:
        """Checks if the shape is equal to another tensor."""
        return C.msml_tensor_is_shape_eq(self.tensor, other.tensor)

    def are_strides_eq(self, other: 'Tensor') -> bool:
        """Checks if the strides are equal to another tensor."""
        return C.msml_tensor_are_strides_eq(self.tensor, other.tensor)

    def can_broadcast(self, other: 'Tensor') -> bool:
        """Checks second tensor can be broadcasted into self."""
        return C.msml_tensor_can_broadcast(self.tensor, other.tensor)

    @property
    def image_width(self) -> int:
        """Returns the width of the image tensor. (Equals to the first dimension)"""
        return self.shape[0]

    @property
    def image_height(self) -> int:
        """Returns the height of the image tensor. (Equals to the second dimension)"""
        return self.shape[1]

    @property
    def image_channels(self) -> int:
        """Returns the number of color channels in the image tensor. (Equals to the third dimension)"""
        return self.shape[2]

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

    def set_scalar_physical_index(self, d0: int, d1: int, d2: int, d3: int, x: float) -> None:
        """Sets the scalar value at a physical index."""
        C.msml_tensor_set_scalar_physical_index(self.tensor, d0, d1, d2, d3, x)

    def get_scalar_virtual_index(self, v_idx: int) -> float:
        """Returns the scalar value at a virtual index."""
        return C.msml_tensor_get_scalar_virtual_index(self.tensor, v_idx)

    def set_scalar_virtual_index(self, v_idx: int, x: float) -> None:
        """Sets the scalar value at a virtual index."""
        C.msml_tensor_set_scalar_virtual_index(self.tensor, v_idx, x)

    def is_close(self, other: 'Tensor', eps: float = -1.0, print_eq_percent: bool = False) -> (bool, float):
        """Checks if the tensor is close to another tensor within a given epsilon."""
        """Returns a tuple with a boolean indicating if the tensors are close and the percentage of equal elements."""
        percent_eq = ffi.new(f'double[1]')
        is_eq: bool = C.msml_tensor_isclose(self.tensor, other.tensor, eps, percent_eq)
        if print_eq_percent:
            print(f'Tensors are close: {is_eq}, Percent equal: {percent_eq[0]:.2f}%')
        return is_eq, percent_eq[0]

    def save(self, file_path: str) -> None:
        """Saves to tensor to a binary MSML file"""
        if not file_path.endswith('.msml'):
            file_path += '.msml'
        C.msml_tensor_save(self.tensor, bytes(file_path, 'utf-8'))

    def save_to_image(self, file_path: str) -> None:
        """Saves the tensor as an JPG image to a file."""
        assert self.rank == 3, 'Tensor must be a 3D image tensor'
        channels: int = self.image_channels
        assert channels in (1, 3, 4), 'Invalid number of color channels'
        C.msml_tensor_save_to_image(self.tensor, bytes(file_path, 'utf-8'))

    @staticmethod
    def empty(ctx: Context, shape: list[int], dtype: DType = DType.F32, name: str | None = None) -> 'Tensor':
        """Creates an empty tensor, with uninitialized data."""
        tensor = Tensor(None)
        tensor._create_internal(ctx, shape, dtype, name)
        return tensor

    @staticmethod
    def isomorphic_clone(tensor) -> 'Tensor':
        """Create new empty tensor with same shape as input, but without cloning data."""
        return Tensor(C.msml_tensor_isomorphic_clone(tensor.tensor))

    @staticmethod
    def deep_clone(tensor) -> 'Tensor':
        """Create new tensor with same shape and data as input."""
        return Tensor(C.msml_tensor_deep_clone(tensor.tensor))

    @staticmethod
    def full(ctx: Context, shape: list[int], fill_value: float, dtype: DType = DType.F32,
             name: str | None = None) -> 'Tensor':
        """Creates a tensor filled with a constant value."""
        tensor = Tensor(None)
        tensor._create_internal(ctx, shape, dtype, name)
        tensor.fill(fill_value)
        return tensor

    @staticmethod
    def zeros(ctx: Context, shape: list[int], dtype: DType = DType.F32, name: str | None = None) -> 'Tensor':
        """Creates a tensor filled with zeros."""
        return Tensor.full(ctx, shape, 1.0, dtype, name)

    @staticmethod
    def random(ctx: Context, shape: list[int], interval: (float, float) = (0.0, 1.0), dtype: DType = DType.F32,
               name: str | None = None) -> 'Tensor':
        """Creates a tensor filled with random values within [min, max]."""
        tensor = Tensor(None)
        tensor._create_internal(ctx, shape, dtype, name)
        tensor.fill_random(interval)
        return tensor

    @staticmethod
    def load(ctx: Context, file_path: str) -> 'Tensor':
        assert file_path.endswith('.msml'), 'File must be a MSML file'
        """Loads a tensor from a binary MSML file."""
        instance = C.msml_tensor_load(ctx.ctx, bytes(file_path, 'utf-8'))
        return Tensor(internal_instance=instance)

    @staticmethod
    def from_image(ctx: Context,
                   name: str | None,
                   file_path: str,
                   desired_color_channels=DesiredColorChannels.AUTO,
                   resize_to_dims: tuple[int, int] = (0, 0)) -> 'Tensor':
        """Loads an image from a file and creates a tensor from it."""
        instance = C.msml_tensor_create_from_image(ctx.ctx, bytes(file_path, 'utf-8'), desired_color_channels.value,
                                                   resize_to_dims[0], resize_to_dims[1])
        tensor = Tensor(internal_instance=instance)
        if name is not None:
            tensor.name = name
        return tensor

    def softmax(self, derivative: bool = False) -> 'Tensor':
        result = self.isomorphic_clone(self)
        result.set_op_with_args(Operation.SOFTMAX_DV if derivative else Operation.SOFTMAX, self)
        return result

    def sigmoid(self, derivative: bool = False) -> 'Tensor':
        result = self.isomorphic_clone(self)
        result.set_op_with_args(Operation.SIGMOID_DV if derivative else Operation.SIGMOID, self)
        return result

    def silu(self, derivative: bool = False) -> 'Tensor':
        result = self.isomorphic_clone(self)
        result.set_op_with_args(Operation.SILU_DV if derivative else Operation.SILU, self)
        return result

    def tanh(self, derivative: bool = False) -> 'Tensor':
        result = self.isomorphic_clone(self)
        result.set_op_with_args(Operation.TANH_DV if derivative else Operation.TANH, self)
        return result

    def relu(self, derivative: bool = False) -> 'Tensor':
        result = self.isomorphic_clone(self)
        result.set_op_with_args(Operation.RELU_DV if derivative else Operation.RELU, self)
        return result

    def gelu(self, derivative: bool = False) -> 'Tensor':
        result = self.isomorphic_clone(self)
        result.set_op_with_args(Operation.GELU_DV if derivative else Operation.GELU, self)
        return result

    def __str__(self) -> str:
        fmt: str = f'Tensor {"?" if self.name == "" else self.name}, DType: {self.dtype}, Rank: {self.rank}, Shape: {self.shape}, Strides: {self.shape}, Mem: {humanize_memory_size(self.buf_size)}'
        return fmt

    def __add__(self, other: 'Tensor') -> 'Tensor':
        result = self.isomorphic_clone(self)
        result.set_op_with_args(Operation.ADD, self, other)
        return result

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        result = self.isomorphic_clone(self)
        result.set_op_with_args(Operation.SUB, self, other)
        return result

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        result = self.isomorphic_clone(self)
        result.set_op_with_args(Operation.MUL, self, other)
        return result

    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        result = self.isomorphic_clone(self)
        result.set_op_with_args(Operation.DIV, self, other)
        return result

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        result = self.isomorphic_clone(self)
        result.set_op_with_args(Operation.MATMUL, self, other)
        return result

    def __eq__(self, other: 'Tensor') -> bool:
        return C.msml_tensor_eq(self.tensor, other.tensor)
