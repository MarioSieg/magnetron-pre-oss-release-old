# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# Implements core functionality: Context, Tensors and Operations.

# To debug Python to C FFI calls:
# $ cp examples/perceptron.py tmp.py && gdb -ex r --args python3 tmp.py
# See also https://wiki.python.org/moin/DebuggingWithGdb

import random
import faulthandler
import weakref

from os.path import isfile
from msml._lib_loader import load_native_msml_lib
from enum import Enum, auto

# Enable faulthandler for debugging
faulthandler.enable()

ffi, C = load_native_msml_lib()  # Load the native MSML shared library

# Define Python wrapper classes
MAX_DIMS: int = 6
MAX_ARG_TENSORS: int = 2
MSML_MAX_OP_PARAMS: int = 6
DIM_MAX: int = ((1 << 64) - 1) >> 1


def humanize_memory_size(size: int) -> str:
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    unit = 0
    while size >= (1 << 10) and unit < len(units) - 1:
        size /= (1 << 10)
        unit += 1
    return f'{size:.2f} {units[unit]}'


def pack_color(r: int, g: int, b: int) -> int:
    return C.msml_pack_color_u8(r, g, b)


class PRNGAlgorithm(Enum):
    MERSENNE_TWISTER = 0  # Default - Mersenne Twister Generator
    PCG = auto()  # Permuted Congruential Generator


class DType(Enum):
    """Enumerates the supported data types for tensors."""
    F32 = 0


class ColorChannels(Enum):
    """Enumerates the desired color channels when loading images."""
    AUTO = 0  # Automatically determine the number of color channels
    GRAY = auto()  # Grayscale F32
    GRAY_A = auto()  # Grayscale F32 with alpha F32
    RGB = auto()  # R32G32B32
    RGBA = auto()  # R32G32B32A32


class Op(Enum):
    """All supported tensor operations."""
    NOP = 0
    CLONE = auto()
    VIEW = auto()
    TRANSPOSE = auto()
    PERMUTE = auto()
    MEAN = auto()
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


class OpParam:
    """Represents an operation parameter."""

    def __init__(self, value: int) -> None:
        self.value = value

    @staticmethod
    def int(x: int) -> 'OpParam':
        """Creates an integer operation parameter."""
        return OpParam(C.msml_op_param_int(x))

    @property
    def is_int(self) -> bool:
        """Returns if the operation parameter is an integer."""
        return C.msml_op_param_is_int(self.value)

    @property
    def unpack_int(self) -> int:
        """Returns the integer value of the operation parameter."""
        assert self.is_int
        return C.msml_op_param_unpack_int(self.value)


class GraphEvalOrder(Enum):
    """Order in which the computation graph should be evaluated. Applies to deferred execution mode only."""
    FORWARD = 0  # Evaluate the graph in forward order (left-to-right)
    REVERSE = 1  # Evaluate the graph in reverse order (right-to-left)


class ExecutionMode(Enum):
    """Execution modes for the MSML context."""
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
        self.ctx = ffi.NULL


Context.G = Context()  # Create the global context


class Tensor:
    """Represents a tensor in the MSML library."""

    def __init__(self, internal_instance: ffi.CData | None = None) -> None:
        self.tensor = internal_instance

    def __del__(self) -> None:
        """Destructor to release tensor resources."""
        self.tensor = ffi.NULL

    def _new(self, ctx: Context, shape: list[int], dtype: DType = DType.F32,
             name: str | None = None) -> None:
        assert 0 < len(shape) <= MAX_DIMS, f'Invalid number of dimensions: {len(shape)}'
        assert all(0 < dim <= DIM_MAX for dim in shape), 'Invalid dimension size'
        self.context_ref = weakref.ref(ctx)
        dispatch = {
            1: C.msml_tensor_create_1d,
            2: C.msml_tensor_create_2d,
            3: C.msml_tensor_create_3d,
            4: C.msml_tensor_create_4d,
            5: C.msml_tensor_create_5d,
            6: C.msml_tensor_create_6d
        }
        assert len(shape) in dispatch
        self.tensor = dispatch[len(shape)](ctx.ctx, dtype.value, *shape)
        self.name = f'Tensor {self.shape}' if name is None else name

    @staticmethod
    def operator(op: Op, params: list[OpParam] | None = None, *args) -> 'Tensor':
        c_para: ffi.CData
        c_para_ptr: ffi.CData = ffi.NULL
        if params is not None:
            assert 0 < len(params) <= MSML_MAX_OP_PARAMS, f'Invalid number of operation parameters: {len(params)}'
            param_vals = [param.value & ((1 << 64) - 1) for param in params] + [OpParam.int(0)] * (
                    MSML_MAX_OP_PARAMS - len(params))
            c_para = ffi.new(f'msml_op_param_t[{MSML_MAX_OP_PARAMS}]', param_vals)
            c_para_ptr = ffi.new(f'msml_op_param_t(*)[{MSML_MAX_OP_PARAMS}]', c_para)
        assert len(args) == op.argument_count, f'{len(args)} != {op.argument_count}'
        tensors: ffi.CData = ffi.new(f'msml_tensor_t*[{len(args)}]', [arg.tensor for arg in args])
        ctx: ffi.CData = C.msml_tensor_get_ctx(args[0].tensor)
        instance: ffi.CData = C.msml_tensor_operator(ctx, op.value, tensors, len(args), c_para_ptr)
        assert instance != ffi.NULL, 'Operation invalid'
        return Tensor(instance)

    def fill(self, x: float) -> None:
        """Sets all elements of the tensor to x."""
        C.msml_tensor_fill(self.tensor, x)

    def fill_random(self, interval: (float, float) = (0.0, 1.0)) -> None:
        """Sets all elements of the tensor to random values within [min, max]"""
        if interval[1] < interval[0]:
            interval = (interval[1], interval[0])
        C.msml_tensor_fill_random(self.tensor, interval[0], interval[1])

    def print(self, print_header: bool = False, print_data: bool = True) -> None:
        """Prints the tensor metadata and optionally its data."""
        C.msml_tensor_print(self.tensor, print_header, print_data)

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

    def data_as_f32(self) -> list[float]:
        """Returns the data of the tensor buffer as a list of floats."""
        assert self.dtype == DType.F32, 'Invalid data type'
        return ffi.unpack(C.msml_tensor_data_as_f32(self.tensor), self.num_elements)

    def unpack_scalar(self) -> float:
        return self.data_as_f32()[0]

    @property
    def data_size(self) -> int:
        """Returns the size of the tensor buffer in bytes."""
        return C.msml_tensor_data_size(self.tensor)

    @property
    def num_elements(self) -> int:
        """Returns the size of the tensor buffer in bytes."""
        return C.msml_tensor_num_elements(self.tensor)

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
    def is_volume(self) -> bool:
        """Checks if the tensor is a higher-order 3D tensor."""
        return C.msml_tensor_is_volume(self.tensor)

    @property
    def is_transposed(self) -> bool:
        """Checks if the tensor is transposed."""
        return C.msml_tensor_is_transposed(self.tensor)

    @property
    def is_permuted(self) -> bool:
        """Checks if the tensor is permuted."""
        return C.msml_tensor_is_permuted(self.tensor)

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
    def width(self) -> int:
        """Returns the width of the image tensor. (Equals to the first dimension)"""
        return self.shape[2]

    @property
    def height(self) -> int:
        """Returns the height of the image tensor. (Equals to the second dimension)"""
        return self.shape[1]

    @property
    def channels(self) -> int:
        """Returns the number of color channels in the image tensor. (Equals to the third dimension)"""
        return self.shape[0]

    @property
    def is_contiguous(self) -> bool:
        """Checks if the tensor is contiguous in memory."""
        return C.msml_tensor_is_contiguous(self.tensor)

    def is_close(self, other: 'Tensor', eps: float = -1.0, print_eq_percent: bool = False) -> (bool, float):
        """Checks if the tensor is close to another tensor within a given epsilon."""
        """Returns a tuple with a boolean indicating if the tensors are close and the percentage of equal elements."""
        percent_eq: ffi.CData = ffi.new('double[1]')
        is_eq: bool = C.msml_tensor_is_close(self.tensor, other.tensor, eps, percent_eq)
        if print_eq_percent:
            print(f'Tensors are close: {is_eq}, Percent equal: {percent_eq[0]:.2f}%')
        return is_eq, percent_eq[0]

    def image_draw_box(self, p1: (int, int), p2: (int, int), width: int = 2, rgb: int = pack_color(0xff, 0xff, 0xff)):
        assert p2[0] > p1[0] and p2[1] > p1[1] and width > 0
        C.msml_tensor_img_draw_box(self.tensor, p1[0], p1[1], p2[0], p2[1], width, rgb & 0xffffff)

    def save(self, file_path: str) -> None:
        """Saves to tensor to a binary MSML file"""
        if not file_path.endswith('.msml'):
            file_path += '.msml'
        C.msml_tensor_save(self.tensor, bytes(file_path, 'utf-8'))

    def save_image(self, file_path: str) -> None:
        """Saves the tensor as an JPG image to a file."""
        assert self.rank == 3, 'Tensor must be a 3D image tensor'
        assert self.channels in (1, 3, 4), 'Invalid number of color channels'
        C.msml_tensor_save_image(self.tensor, bytes(file_path, 'utf-8'))

    @staticmethod
    def empty(shape: list[int], dtype: DType = DType.F32, ctx: Context = Context.G,
              name: str | None = None) -> 'Tensor':
        """Creates an empty tensor, with uninitialized data."""
        tensor = Tensor(None)
        tensor._new(ctx, shape, dtype, name)
        return tensor

    @staticmethod
    def full(shape: list[int], fill_value: float, dtype: DType = DType.F32,
             ctx: Context = Context.G,
             name: str | None = None) -> 'Tensor':
        """Creates a tensor filled with a constant value."""
        tensor = Tensor(None)
        tensor._new(ctx, shape, dtype, name)
        tensor.fill(fill_value)
        return tensor

    @staticmethod
    def const(data: list[float], shape: list[int] | None = None, dtype: DType = DType.F32, ctx: Context = Context.G,
              name: str | None = None) -> 'Tensor':
        """Creates a tensor with the given data."""
        tensor = Tensor(None)
        tensor._new(ctx, shape if shape is not None else [len(data)], dtype, name)
        size: int = len(data) * ffi.sizeof('float')
        C.msml_tensor_copy_buffer_from(tensor.tensor, ffi.new(f'float[{len(data)}]', data), size)
        return tensor

    @staticmethod
    def zeros(shape: list[int], dtype: DType = DType.F32, ctx: Context = Context.G,
              name: str | None = None) -> 'Tensor':
        """Creates a tensor filled with zeros."""
        return Tensor.full(shape, 1.0, dtype, ctx, name)

    @staticmethod
    def random(shape: list[int], interval: (float, float) = (-1.0, 1.0), dtype: DType = DType.F32,
               ctx: Context = Context.G,
               name: str | None = None) -> 'Tensor':
        """Creates a tensor filled with random values within [min, max]."""
        tensor = Tensor(None)
        tensor._new(ctx, shape, dtype, name)
        tensor.fill_random(interval)
        return tensor

    @staticmethod
    def load(file_path: str, ctx: Context = Context.G) -> 'Tensor':
        assert file_path.endswith('.msml'), 'File must be a MSML file'
        """Loads a tensor from a binary MSML file."""
        instance = C.msml_tensor_load(ctx.ctx, bytes(file_path, 'utf-8'))
        return Tensor(internal_instance=instance)

    @staticmethod
    def load_image(file_path: str,
                   ctx: Context = Context.G,
                   name: str | None = None,
                   channels=ColorChannels.AUTO,
                   resize_to_dims: (int, int) = (0, 0)) -> 'Tensor':
        """Loads an image from a file and creates a tensor from it."""
        assert isfile(file_path), f'File not found: {file_path}'
        instance = C.msml_tensor_load_image(ctx.ctx, bytes(file_path, 'utf-8'), channels.value, resize_to_dims[0],
                                            resize_to_dims[1])
        tensor = Tensor(internal_instance=instance)
        if name is not None:
            tensor.name = name
        return tensor

    def clone(self) -> 'Tensor':
        """Create new tensor with same shape and data as input. (deep clone)"""
        return self.operator(Op.CLONE, None, self)

    def view(self) -> 'Tensor':
        """Create new tensor with same shape as input, and with data referencing into the input tensor's data. (shallow copy)"""
        return self.operator(Op.VIEW, None, self)

    def transpose(self) -> 'Tensor':
        """Transposes the tensor."""
        return self.operator(Op.TRANSPOSE, None, self)

    def permute(self, axes: list[int]) -> 'Tensor':
        """Permutes the tensor according to the given axes."""
        assert len(axes) == MAX_DIMS, f'Invalid number of axes: {axes}'
        for i in range(MAX_DIMS):
            assert 0 <= axes[i] < MAX_DIMS, f'Invalid axis: {axes[i]}'
            for j in range(i + 1, MAX_DIMS):  # All axes must be unique
                assert axes[i] != axes[j], f'Duplicate axis: {axes[i]}'
        return self.operator(Op.PERMUTE, [OpParam.int(axis) for axis in axes], self)

    def mean(self) -> 'Tensor':
        """Applies the arithmetic mean to the tensor and reduces to scalar."""
        return self.operator(Op.MEAN, None, self)

    def step(self) -> 'Tensor':
        """Applies the heaviside step function to the tensor."""
        return self.operator(Op.STEP, None, self)

    def softmax(self, derivative: bool = False) -> 'Tensor':
        """Applies the softmax function to the tensor."""
        return self.operator(Op.SOFTMAX_DV if derivative else Op.SOFTMAX, None, self)

    def sigmoid(self, derivative: bool = False) -> 'Tensor':
        """Applies the sigmoid function to the tensor."""
        return self.operator(Op.SIGMOID_DV if derivative else Op.SIGMOID, None, self)

    def hard_sigmoid(self) -> 'Tensor':
        """Applies the hard sigmoid function to the tensor."""
        return self.operator(Op.HARD_SIGMOID, None, self)

    def silu(self, derivative: bool = False) -> 'Tensor':
        """Applies the SiLU function to the tensor."""
        return self.operator(Op.SILU_DV if derivative else Op.SILU, None, self)

    def tanh(self, derivative: bool = False) -> 'Tensor':
        """Applies the hyperbolic tangent function to the tensor."""
        return self.operator(Op.TANH_DV if derivative else Op.TANH, None, self)

    def relu(self, derivative: bool = False) -> 'Tensor':
        """Applies the ReLU function to the tensor."""
        return self.operator(Op.RELU_DV if derivative else Op.RELU, None, self)

    def gelu(self, derivative: bool = False) -> 'Tensor':
        """Applies the GELU function to the tensor."""
        return self.operator(Op.GELU_DV if derivative else Op.GELU, None, self)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        """Adds two tensors element-wise."""
        return self.operator(Op.ADD, None, self, other)

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        """Subtracts two tensors element-wise."""
        return self.operator(Op.SUB, None, self, other)

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        """Multiplies two tensors element-wise. (Hadamard product)"""
        return self.operator(Op.MUL, None, self, other)

    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        """Divides two tensors element-wise."""
        return self.operator(Op.DIV, None, self, other)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Multiplies two tensors using matrix multiplication."""
        return self.operator(Op.MATMUL, None, self, other)

    def __eq__(self, other: 'Tensor') -> bool:
        """Checks if two tensors are equal."""
        return C.msml_tensor_eq(self.tensor, other.tensor)

    def __str__(self) -> str:
        self.print(True)
        return ''
