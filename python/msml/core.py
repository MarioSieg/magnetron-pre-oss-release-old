# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

# To debug Python to C FFI calls:
# $ cp examples/perceptron.py tmp.py && gdb -ex r --args python3 tmp.py
# See also https://wiki.python.org/moin/DebuggingWithGdb

import random
import faulthandler
import weakref

from msml._lib_loader import load_native_msml_lib
from enum import Enum, auto

# Enable faulthandler for debugging
faulthandler.enable()

ffi, C = load_native_msml_lib()  # Load the native MSML shared library

# Define Python wrapper classes
MAX_DIMS: int = 4
MAX_ARG_TENSORS: int = 2
DIM_MAX: int = 0x7fffffffffffffff


def humanize_memory_size(size: int) -> str:
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    unit = 0
    while size >= (1 << 10) and unit < len(units) - 1:
        size /= (1 << 10)
        unit += 1
    return f'{size:.2f} {units[unit]}'


class PRNGAlgorithm(Enum):
    MERSENNE_TWISTER = 0  # Default - Mersenne Twister Generator
    PCG = auto()  # Permuted Congruential Generator


class DType(Enum):
    """Enumerates the supported data types for tensors."""
    F32 = 0


class DesiredColorChannels(Enum):
    """Enumerates the desired color channels when loading images."""
    AUTO = 0  # Automatically determine the number of color channels
    GRAY = auto()  # Grayscale F32
    GRAY_A = auto()  # Grayscale F32 with alpha F32
    RGB = auto()  # R32G32B32
    RGBA = auto()  # R32G32B32A32


class Operation(Enum):
    """All supported tensor operations."""
    NOP = 0
    CLONE = auto()
    VIEW = auto()
    TRANSPOSE = auto()
    STEP = auto()
    SOFTMAX = auto()
    SOFTMAX_DV = auto()
    SIGMOID = auto()
    SIGMOID_DV = auto()
    HARD_SIGMOID = auto()
    SILU = auto()
    SILU_DV = auto()
    TANH = auto()
    TANH_DV = auto()
    RELU = auto()
    RELU_DV = auto()
    GELU = auto()
    GELU_DV = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MATMUL = auto()

    _COUNT = auto()

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


class GraphEvalOrder(Enum):
    """Enumerates the order in which the graph should be evaluated."""
    FORWARD = 0  # Evaluate the graph in forward order (left-to-right)
    REVERSE = 1  # Evaluate the graph in reverse order (right-to-left)


class ExecutionMode(Enum):
    """"""
    EAGER = 0  # Execute operations immediately. (Dynamic computation graph, like PyTorch).
    DEFERRED = 1  # Build computation graph and execute later. (Static computation graph, like TensorFlow 1.0).


class Context:
    """Manages the MSML context and tensor lifecycles."""

    G: 'Context' = None  # Global context

    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.EAGER,
                 pool_chunk_size: int = 1 << 30):  # Pool chunk size. Default: 2GiB
        self.ctx = C.msml_ctx_create2(pool_chunk_size)
        self.execution_mode = execution_mode

    @property
    def execution_mode(self) -> ExecutionMode:
        """Returns the execution mode of the context."""
        return ExecutionMode(C.msml_ctx_get_exec_mode(self.ctx))

    @execution_mode.setter
    def execution_mode(self, mode: ExecutionMode):
        """Sets the execution mode of the context."""
        C.msml_ctx_set_exec_mode(self.ctx, mode.value)

    @property
    def prng_algorithm(self) -> PRNGAlgorithm:
        """Returns the PRNG algorithm used by the context."""
        return PRNGAlgorithm(C.msml_ctx_get_prng_algorithm(self.ctx))

    @prng_algorithm.setter
    def prng_algorithm(self, algorithm: PRNGAlgorithm):
        """Sets the PRNG algorithm and seed for the context."""
        C.msml_ctx_set_prng_algorithm(self.ctx, algorithm.value, random.randint(0, 1 << 63))

    @property
    def os_name(self) -> str:
        """Returns the name of the operating system."""
        return ffi.string(C.msml_ctx_get_os_name(self.ctx)).decode('utf-8')

    @property
    def cpu_name(self) -> str:
        """Returns the name of the CPU."""
        return ffi.string(C.msml_ctx_get_cpu_name(self.ctx)).decode('utf-8')

    @property
    def cpu_virtual_cores(self) -> int:
        """Returns the number of virtual cores of the CPU."""
        return C.msml_ctx_get_cpu_virtual_cores(self.ctx)

    @property
    def cpu_physical_cores(self) -> int:
        """Returns the number of physical cores of the CPU."""
        return C.msml_ctx_get_cpu_physical_cores(self.ctx)

    @property
    def cpu_sockets(self) -> int:
        """Returns the number of CPU sockets."""
        return C.msml_ctx_get_cpu_sockets(self.ctx)

    @property
    def physical_memory_total(self) -> int:
        """Returns the total physical memory in bytes."""
        return C.msml_ctx_get_physical_memory_total(self.ctx)

    @property
    def physical_memory_free(self) -> int:
        """Returns the free physical memory in bytes."""
        return C.msml_ctx_get_physical_memory_free(self.ctx)

    @property
    def physical_memory_used(self) -> int:
        """Returns the used physical memory in bytes."""
        return abs(self.physical_memory_total - self.physical_memory_free)

    @property
    def is_numa_system(self) -> bool:
        """Returns if the system is a NUMA system."""
        return C.msml_ctx_is_numa_system(self.ctx)

    @property
    def total_allocated_pool_memory(self) -> int:
        """Returns the total memory allocated in the context in bytes."""
        return C.msml_ctx_total_allocated_pool_memory(self.ctx)

    def __del__(self):
        C.msml_ctx_destroy(self.ctx)


Context.G = Context()  # Create the global context


class Tensor:
    """Represents a tensor in the MSML library."""

    def __init__(self, internal_instance: ffi.CData | None = None) -> None:
        self.tensor = internal_instance

    def __del__(self) -> None:
        """Destructor to release tensor resources."""
        if self.tensor is not None:
            self.tensor = None

    def _create_internal(self, ctx: Context, shape: list[int], dtype: DType = DType.F32,
                         name: str | None = None) -> None:
        assert 0 < len(shape) <= MAX_DIMS, 'Number of dimensions exceeds maximum'
        for dim in shape:
            assert DIM_MAX > dim > 0, 'Invalid dimension size'
        self.context_ref = weakref.ref(ctx)
        match len(shape):
            case 1:
                self.tensor = C.msml_tensor_create_1d(ctx.ctx, dtype.value, shape[0])
            case 2:
                self.tensor = C.msml_tensor_create_2d(ctx.ctx, dtype.value, shape[0], shape[1])
            case 3:
                self.tensor = C.msml_tensor_create_3d(ctx.ctx, dtype.value, shape[0], shape[1], shape[2])
            case 4:
                self.tensor = C.msml_tensor_create_4d(ctx.ctx, dtype.value, shape[0], shape[1], shape[2], shape[3])
            case _:
                raise RuntimeError('Invalid number of dimensions')
        if name is not None:
            self.name = name

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
        return ffi.unpack(C.msml_tensor_shape(self.tensor), self.rank)

    @property
    def strides(self) -> list[int]:
        """Returns the strides of the tensor."""
        return ffi.unpack(C.msml_tensor_strides(self.tensor), self.rank)

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
        return ffi.unpack(C.msml_tensor_buf_f32(self.tensor), self.num_elements)

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

    @property
    def is_transposed(self) -> bool:
        """Checks if the tensor is transposed."""
        return C.msml_tensor_is_transposed(self.tensor)

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
        is_eq: bool = C.msml_tensor_is_close(self.tensor, other.tensor, eps, percent_eq)
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
    def empty(shape: list[int], dtype: DType = DType.F32, ctx: Context = Context.G,
              name: str | None = None) -> 'Tensor':
        """Creates an empty tensor, with uninitialized data."""
        tensor = Tensor(None)
        tensor._create_internal(ctx, shape, dtype, name)
        return tensor

    @staticmethod
    def full(shape: list[int], fill_value: float, dtype: DType = DType.F32,
             ctx: Context = Context.G,
             name: str | None = None) -> 'Tensor':
        """Creates a tensor filled with a constant value."""
        tensor = Tensor(None)
        tensor._create_internal(ctx, shape, dtype, name)
        tensor.fill(fill_value)
        return tensor

    @staticmethod
    def with_data(shape: list[int], data: list[float], dtype: DType = DType.F32, ctx: Context = Context.G,
                  name: str | None = None) -> 'Tensor':
        """Creates a tensor with the given data."""
        tensor = Tensor(None)
        tensor._create_internal(ctx, shape, dtype, name)
        size: int = len(data) * ffi.sizeof('float')
        C.msml_tensor_copy_buffer_from(tensor.tensor, ffi.new(f'float[{len(data)}]', data), size)
        return tensor

    @staticmethod
    def zeros(shape: list[int], dtype: DType = DType.F32, ctx: Context = Context.G, name: str | None = None) -> 'Tensor':
        """Creates a tensor filled with zeros."""
        return Tensor.full(shape, 1.0, dtype, ctx, name)

    @staticmethod
    def random(shape: list[int], interval: (float, float) = (0.0, 1.0), dtype: DType = DType.F32, ctx: Context = Context.G,
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

    @staticmethod
    def _emit_op_tensor(op: Operation, *args) -> 'Tensor':
        assert len(args) == op.argument_count, f'{len(args)} != {op.argument_count}'
        tensors = ffi.new(f'msml_tensor_t*[{len(args)}]')
        for i, arg in enumerate(args):
            assert isinstance(arg, Tensor), 'Argument must be a tensor'
            tensors[i] = arg.tensor
        ctx: ffi.CData = C.msml_tensor_get_ctx(args[0].tensor)
        instance: ffi.CData = C.msml_tensor_operator(ctx, op.value, tensors, len(args))
        if instance == ffi.NULL:
            raise RuntimeError('Operation not possible')
        return Tensor(instance)

    def clone(self) -> 'Tensor':
        """Create new tensor with same shape and data as input. (deep clone)"""
        return self._emit_op_tensor(Operation.CLONE, self)

    def view(self) -> 'Tensor':
        """Create new tensor with same shape as input, and with data referencing into the input tensor's data. (shallow copy)"""
        return self._emit_op_tensor(Operation.VIEW, self)

    def transpose(self) -> 'Tensor':
        """Transposes the tensor."""
        return self._emit_op_tensor(Operation.TRANSPOSE, self)

    def step(self) -> 'Tensor':
        """Applies the heaviside step function to the tensor."""
        return self._emit_op_tensor(Operation.STEP, self)

    def softmax(self, derivative: bool = False) -> 'Tensor':
        """Applies the softmax function to the tensor."""
        return self._emit_op_tensor(Operation.SOFTMAX_DV if derivative else Operation.SOFTMAX, self)

    def sigmoid(self, derivative: bool = False) -> 'Tensor':
        """Applies the sigmoid function to the tensor."""
        return self._emit_op_tensor(Operation.SIGMOID_DV if derivative else Operation.SIGMOID, self)

    def hard_sigmoid(self) -> 'Tensor':
        """Applies the hard sigmoid function to the tensor."""
        return self._emit_op_tensor(Operation.HARD_SIGMOID, self)

    def silu(self, derivative: bool = False) -> 'Tensor':
        """Applies the SiLU function to the tensor."""
        return self._emit_op_tensor(Operation.SILU_DV if derivative else Operation.SILU, self)

    def tanh(self, derivative: bool = False) -> 'Tensor':
        """Applies the hyperbolic tangent function to the tensor."""
        return self._emit_op_tensor(Operation.TANH_DV if derivative else Operation.TANH, self)

    def relu(self, derivative: bool = False) -> 'Tensor':
        """Applies the ReLU function to the tensor."""
        return self._emit_op_tensor(Operation.RELU_DV if derivative else Operation.RELU, self)

    def gelu(self, derivative: bool = False) -> 'Tensor':
        """Applies the GELU function to the tensor."""
        return self._emit_op_tensor(Operation.GELU_DV if derivative else Operation.GELU, self)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        """Adds two tensors element-wise."""
        return self._emit_op_tensor(Operation.ADD, self, other)

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        """Subtracts two tensors element-wise."""
        return self._emit_op_tensor(Operation.SUB, self, other)

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        """Multiplies two tensors element-wise. (Hadamard product)"""
        return self._emit_op_tensor(Operation.MUL, self, other)

    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        """Divides two tensors element-wise."""
        return self._emit_op_tensor(Operation.DIV, self, other)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Multiplies two tensors using transposed matrix multiplication. Computes Rᵀ = A x Bᵀ instead of 'normal' R = A x B."""
        return self._emit_op_tensor(Operation.MATMUL, self, other)

    def __eq__(self, other: 'Tensor') -> bool:
        """Checks if two tensors are equal."""
        return C.msml_tensor_eq(self.tensor, other.tensor)

    def __str__(self) -> str:
        #fmt: str = f'Tensor {"?" if self.name == "" else self.name}, DType: {self.dtype}, Rank: {self.rank}, Shape: {self.shape}, Strides: {self.shape}, Mem: {humanize_memory_size(self.buf_size)}'
        #return fmt
        self.print(True)
        return ''
